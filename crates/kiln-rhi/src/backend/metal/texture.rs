use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLDevice, MTLHeap, MTLPixelFormat, MTLResidencySet, MTLResource, MTLStorageMode, MTLTexture,
    MTLTextureDescriptor, MTLTextureType, MTLTextureUsage as MtlTextureUsage,
};

use super::as_allocation;
use super::device::{MetalDevice, MetalRetiredResource};
use crate::error::{RhiError, RhiResult};
use crate::texture::{
    Texture, TextureDesc, TextureSizeAlign, TextureUsage, TextureViewDesc, ViewKind,
};
use crate::types::{Format, GpuPtr, TextureDimension, TextureHandle, TextureId};

pub fn format_to_mtl(format: Format) -> MTLPixelFormat {
    match format {
        // Color
        Format::R8Unorm => MTLPixelFormat::R8Unorm,
        Format::R8G8Unorm => MTLPixelFormat::RG8Unorm,
        Format::R8G8B8A8Unorm => MTLPixelFormat::RGBA8Unorm,
        Format::R8G8B8A8Srgb => MTLPixelFormat::RGBA8Unorm_sRGB,
        Format::B8G8R8A8Unorm => MTLPixelFormat::BGRA8Unorm,
        Format::B8G8R8A8Srgb => MTLPixelFormat::BGRA8Unorm_sRGB,
        Format::R16Float => MTLPixelFormat::R16Float,
        Format::R16G16Float => MTLPixelFormat::RG16Float,
        Format::R16G16B16A16Float => MTLPixelFormat::RGBA16Float,
        Format::R32Float => MTLPixelFormat::R32Float,
        Format::R32G32Float => MTLPixelFormat::RG32Float,
        Format::R32G32B32A32Float => MTLPixelFormat::RGBA32Float,
        Format::R10G10B10A2Unorm => MTLPixelFormat::RGB10A2Unorm,
        Format::R11G11B10Float => MTLPixelFormat::RG11B10Float,
        Format::R16Uint => MTLPixelFormat::R16Uint,
        Format::R32Uint => MTLPixelFormat::R32Uint,
        Format::D16Unorm => MTLPixelFormat::Depth16Unorm,
        Format::D32Float => MTLPixelFormat::Depth32Float,
        Format::D24UnormS8Uint => MTLPixelFormat::Depth24Unorm_Stencil8,
        Format::D32FloatS8Uint => MTLPixelFormat::Depth32Float_Stencil8,
    }
}

pub fn mtl_to_format(mtl: MTLPixelFormat) -> Format {
    match mtl {
        MTLPixelFormat::R8Unorm => Format::R8Unorm,
        MTLPixelFormat::RG8Unorm => Format::R8G8Unorm,
        MTLPixelFormat::RGBA8Unorm => Format::R8G8B8A8Unorm,
        MTLPixelFormat::RGBA8Unorm_sRGB => Format::R8G8B8A8Srgb,
        MTLPixelFormat::BGRA8Unorm => Format::B8G8R8A8Unorm,
        MTLPixelFormat::BGRA8Unorm_sRGB => Format::B8G8R8A8Srgb,
        MTLPixelFormat::R16Float => Format::R16Float,
        MTLPixelFormat::RG16Float => Format::R16G16Float,
        MTLPixelFormat::RGBA16Float => Format::R16G16B16A16Float,
        MTLPixelFormat::R32Float => Format::R32Float,
        MTLPixelFormat::RG32Float => Format::R32G32Float,
        MTLPixelFormat::RGBA32Float => Format::R32G32B32A32Float,
        MTLPixelFormat::RGB10A2Unorm => Format::R10G10B10A2Unorm,
        MTLPixelFormat::RG11B10Float => Format::R11G11B10Float,
        MTLPixelFormat::Depth16Unorm => Format::D16Unorm,
        MTLPixelFormat::Depth32Float => Format::D32Float,
        MTLPixelFormat::Depth24Unorm_Stencil8 => Format::D24UnormS8Uint,
        MTLPixelFormat::Depth32Float_Stencil8 => Format::D32FloatS8Uint,
        MTLPixelFormat::R16Uint => Format::R16Uint,
        MTLPixelFormat::R32Uint => Format::R32Uint,
        other => panic!("MTLPixelFormat {other:?} has no kiln-rhi Format mapping"),
    }
}

fn texture_type(dimension: TextureDimension) -> MTLTextureType {
    match dimension {
        TextureDimension::D1 => MTLTextureType::Type1D,
        TextureDimension::D2 => MTLTextureType::Type2D,
        TextureDimension::D2Array => MTLTextureType::Type2DArray,
        TextureDimension::D3 => MTLTextureType::Type3D,
        TextureDimension::Cube => MTLTextureType::TypeCube,
        TextureDimension::CubeArray => MTLTextureType::TypeCubeArray,
    }
}

pub(crate) struct MetalTexture {
    pub(crate) texture: Retained<ProtocolObject<dyn MTLTexture>>,
    /// True when this entry is a view into another texture rather than a heap placement.
    pub(crate) is_view: bool,
}

impl MetalDevice {
    fn build_texture_descriptor(&self, desc: &TextureDesc) -> Retained<MTLTextureDescriptor> {
        let mtl_desc = MTLTextureDescriptor::new();

        let mut usage = MtlTextureUsage::empty();
        if desc.usage.contains(TextureUsage::SAMPLED) {
            usage |= MtlTextureUsage::ShaderRead;
        }
        if desc.usage.contains(TextureUsage::STORAGE) {
            usage |= MtlTextureUsage::ShaderRead | MtlTextureUsage::ShaderWrite;
        }
        if desc.usage.contains(TextureUsage::COLOR_ATTACHMENT) {
            usage |= MtlTextureUsage::RenderTarget;
        }
        if desc.usage.contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT) {
            usage |= MtlTextureUsage::RenderTarget;
        }
        if desc.usage.contains(TextureUsage::FORMAT_VIEW) {
            usage |= MtlTextureUsage::PixelFormatView;
        }

        unsafe {
            mtl_desc.setPixelFormat(format_to_mtl(desc.format));
            mtl_desc.setWidth(desc.width as usize);
            mtl_desc.setHeight(desc.height as usize);
            mtl_desc.setDepth(desc.depth as usize);
            mtl_desc.setMipmapLevelCount(desc.mip_levels as usize);
            // Metal counts cubes, not faces, so `array_layers` goes through unchanged for every
            // dimension. `TypeCube` requires exactly 1, which `TextureDesc::validate` enforces.
            mtl_desc.setArrayLength(desc.array_layers as usize);
            mtl_desc.setTextureType(texture_type(desc.dimension));
            mtl_desc.setSampleCount(desc.sample_count.count() as usize);
            mtl_desc.setUsage(usage);
            mtl_desc.setStorageMode(MTLStorageMode::Private);
        }

        mtl_desc
    }

    pub fn texture_size_align(&self, desc: &TextureDesc) -> RhiResult<TextureSizeAlign> {
        let mtl_desc = self.build_texture_descriptor(desc);
        let size_align = self
            .shared
            .device
            .heapTextureSizeAndAlignWithDescriptor(&mtl_desc);
        Ok(TextureSizeAlign {
            size: size_align.size as u64,
            align: size_align.align as u64,
        })
    }

    pub fn create_texture(
        &self,
        desc: &TextureDesc,
        texture_gpu: GpuPtr<u8>,
    ) -> RhiResult<Texture> {
        if texture_gpu.is_null() {
            return Err(RhiError::TextureCreation(
                "create_texture requires a non-null texture allocation address".into(),
            ));
        }

        let mtl_desc = self.build_texture_descriptor(desc);
        let address = texture_gpu.address;
        let (heap, heap_offset) = {
            let allocations = self.shared.allocations.borrow();
            let alloc = allocations
                .range(..=address)
                .next_back()
                .map(|(_, alloc)| alloc)
                .filter(|alloc| address - alloc.base.address < alloc.size)
                .ok_or_else(|| {
                    RhiError::TextureCreation(
                        format!("texture address {address:#x} is not inside any allocation").into(),
                    )
                })?;
            // Placement offsets are heap-relative. A bad placement returns nil just below.
            (
                alloc.heap.clone(),
                alloc.heap_offset + (address - alloc.base.address),
            )
        };

        let texture =
            unsafe { heap.newTextureWithDescriptor_offset(&mtl_desc, heap_offset as usize) }
                .ok_or_else(|| {
                    RhiError::TextureCreation("Metal placed texture allocation failed".into())
                })?;

        if let Some(label) = &desc.label {
            texture.setLabel(Some(&NSString::from_str(label)));
        }

        let resource_id = texture.gpuResourceID().to_raw();
        let id = TextureId(self.shared.textures.allocate_id()?);
        self.shared.textures.insert(
            id.0,
            MetalTexture {
                texture,
                is_view: false,
            },
            resource_id,
        );

        Ok(Texture {
            id,
            handle: TextureHandle::from_raw(resource_id),
            views: Vec::new(),
            desc: TextureDesc {
                label: None,
                ..*desc
            },
            _owner: None,
        })
    }

    pub fn destroy_texture(&self, texture: Texture) {
        if let Some(tex) = self.shared.textures.take(texture.id.0) {
            self.queue.release_resource(MetalRetiredResource::Texture {
                id: texture.id,
                texture: tex.texture,
                is_view: tex.is_view,
            });
        }
    }

    pub fn destroy_texture_view(&self, id: TextureId) {
        // Only claims views, and only once; a base texture keeps its slot.
        if let Some(texture) = self.shared.textures.take_if(id.0, |t| t.is_view) {
            self.queue.release_resource(MetalRetiredResource::Texture {
                id,
                texture: texture.texture,
                is_view: true,
            });
        }
    }

    /// Value to store in a [`TextureHandle`](crate::TextureHandle) root field for sampled view
    /// `id`. On Metal a `DescriptorHandle<Texture2D>` is the texture's `gpuResourceID`, which the
    /// shader uses directly.
    pub fn texture_handle_raw(&self, id: TextureId) -> u64 {
        self.shared
            .textures
            .with(id.0, |t| t.texture.gpuResourceID().to_raw())
            .expect("texture id has no live slot")
    }

    /// A Metal view inherits its usage from the source, so `_kind` makes no difference here.
    pub fn create_texture_view(
        &self,
        source: &Texture,
        view: &TextureViewDesc,
        _kind: ViewKind,
    ) -> RhiResult<TextureId> {
        let src_texture = self
            .shared
            .textures
            .with(source.id.0, |t| t.texture.clone())
            .ok_or_else(|| {
                RhiError::TextureCreation("create texture view: invalid source TextureId".into())
            })?;

        let src_format = format_to_mtl(view.format.unwrap_or_else(|| source.desc().format));
        let src_type = texture_type(source.desc().dimension);

        let src_mips = source.desc().mip_levels;
        // `newTextureViewWithPixelFormat:` slices are faces for a cube, images otherwise.
        let src_layers = source
            .desc()
            .dimension
            .face_count(source.desc().array_layers);

        let mip_start = view.base_mip as usize;
        let mip_count = view.resolved_mip_count(src_mips) as usize;
        let layer_start = view.base_layer as usize;
        let layer_count = view.resolved_layer_count(src_layers) as usize;

        let level_range = NSRange::new(mip_start, mip_count);
        let slice_range = NSRange::new(layer_start, layer_count);

        let view_texture = unsafe {
            src_texture
                .newTextureViewWithPixelFormat_textureType_levels_slices(
                    src_format,
                    src_type,
                    level_range,
                    slice_range,
                )
                .ok_or_else(|| {
                    RhiError::TextureCreation("Metal texture view creation failed".into())
                })?
        };

        // A view is not placed in a heap, so it needs its own residency entry.
        self.shared
            .residency_set
            .addAllocation(as_allocation(&view_texture));
        self.shared.residency_dirty.set(true);

        let id = match self.shared.textures.allocate_id() {
            Ok(id) => TextureId(id),
            Err(err) => {
                self.shared
                    .residency_set
                    .removeAllocation(as_allocation(&view_texture));
                self.shared.residency_dirty.set(true);
                return Err(err);
            }
        };
        let resource_id = view_texture.gpuResourceID().to_raw();
        self.shared.textures.insert(
            id.0,
            MetalTexture {
                texture: view_texture,
                is_view: true,
            },
            resource_id,
        );

        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mtl_to_format_is_the_inverse_of_format_to_mtl() {
        for &format in crate::types::ALL_FORMATS {
            assert_eq!(
                mtl_to_format(format_to_mtl(format)),
                format,
                "{format:?} does not survive the round trip through MTLPixelFormat"
            );
        }
    }

    /// Metal's `arrayLength` counts cubes while Vulkan's `arrayLayers` counts faces, so the
    /// conversion has to live in one place and be six-to-one for both cube dimensions.
    #[test]
    fn cube_dimensions_count_six_faces_per_cube() {
        assert_eq!(TextureDimension::Cube.face_count(1), 6);
        assert_eq!(TextureDimension::CubeArray.face_count(4), 24);
        assert_eq!(TextureDimension::D2Array.face_count(4), 4);
        assert_eq!(TextureDimension::D2.face_count(1), 1);
    }
}
