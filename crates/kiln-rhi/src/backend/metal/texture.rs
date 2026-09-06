use objc2_metal::MTLPixelFormat;

use crate::types::Format;

/// Convert RHI Format to MTLPixelFormat.
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
        // Depth
        Format::D16Unorm => MTLPixelFormat::Depth16Unorm,
        Format::D32Float => MTLPixelFormat::Depth32Float,
        Format::D24UnormS8Uint => MTLPixelFormat::Depth24Unorm_Stencil8,
        Format::D32FloatS8Uint => MTLPixelFormat::Depth32Float_Stencil8,
        Format::R16Uint => MTLPixelFormat::R16Uint,
        Format::R32Uint => MTLPixelFormat::R32Uint,
    }
}

/// Convert an `MTLPixelFormat` to the corresponding RHI format.
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

use objc2::rc::Retained;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLDevice, MTLHeap, MTLResidencySet, MTLStorageMode, MTLTexture, MTLTextureDescriptor,
    MTLTextureType, MTLTextureUsage as MtlTextureUsage,
};

use super::as_allocation;
use super::device::{METAL_BINDLESS_TEXTURE_CAPACITY, MetalDevice, MetalRetiredResource};
use crate::error::{RhiError, RhiResult};
use crate::queue::QueueInner;
use crate::texture::{Texture, TextureDesc, TextureSizeAlign, TextureUsage};
use crate::types::{GpuPtr, SampleCount, TextureDimension, TextureHandle, TextureId};

impl MetalDevice {
    /// Build the native `MTLTextureDescriptor` for a `TextureDesc`. Shared by
    /// `texture_size_align` and `create_texture` so the mapping lives in one place.
    fn build_texture_descriptor(&self, desc: &TextureDesc) -> Retained<MTLTextureDescriptor> {
        let mtl_desc = MTLTextureDescriptor::new();

        let texture_type = match desc.dimension {
            TextureDimension::D1 => MTLTextureType::Type1D,
            TextureDimension::D2 => MTLTextureType::Type2D,
            TextureDimension::D2Array => MTLTextureType::Type2DArray,
            TextureDimension::D3 => MTLTextureType::Type3D,
            TextureDimension::Cube => MTLTextureType::TypeCube,
            TextureDimension::CubeArray => MTLTextureType::TypeCubeArray,
        };

        let sample_count = match desc.sample_count {
            SampleCount::S1 => 1usize,
            SampleCount::S2 => 2,
            SampleCount::S4 => 4,
            SampleCount::S8 => 8,
            SampleCount::S16 => 16,
        };

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
            mtl_desc.setArrayLength(desc.array_layers as usize);
            mtl_desc.setTextureType(texture_type);
            mtl_desc.setSampleCount(sample_count);
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

    fn allocate_texture_id(&self) -> RhiResult<TextureId> {
        if let Some(id) = self.shared.free_texture_ids.borrow_mut().pop() {
            return Ok(id);
        }
        let next = self.shared.textures.borrow().len();
        if next >= METAL_BINDLESS_TEXTURE_CAPACITY {
            return Err(RhiError::TextureCreation(
                "Metal bindless texture heap exhausted".into(),
            ));
        }
        Ok(TextureId(next as u32))
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
        let size_align = self
            .shared
            .device
            .heapTextureSizeAndAlignWithDescriptor(&mtl_desc);
        let (heap, heap_offset) = {
            let allocations = self.shared.allocations.borrow();
            let alloc = allocations
                .range(..=texture_gpu.address)
                .next_back()
                .map(|(_, alloc)| alloc)
                .filter(|alloc| texture_gpu.address - alloc.base.address < alloc.size)
                .ok_or_else(|| {
                    RhiError::TextureCreation(format!(
                        "texture allocation address 0x{:x} was not returned by gpuMalloc",
                        texture_gpu.address
                    ))
                })?;

            let offset = texture_gpu.address - alloc.base.address;
            // `newTextureWithDescriptor:offset:` positions the texture within the heap, so the
            // allocation's own placement has to be added in; otherwise every allocation that is
            // not itself at heap offset 0 lands the texture on top of unrelated resources.
            let heap_offset = alloc.heap_offset + offset;
            if !heap_offset.is_multiple_of(size_align.align as u64) {
                return Err(RhiError::TextureCreation(format!(
                    "texture allocation address 0x{:x} has heap offset {heap_offset}, expected alignment {}",
                    texture_gpu.address, size_align.align
                )));
            }
            if size_align.size as u64 > alloc.size - offset {
                return Err(RhiError::TextureCreation(format!(
                    "texture allocation address 0x{:x} has {} bytes available, needs {}",
                    texture_gpu.address,
                    alloc.size - offset,
                    size_align.size
                )));
            }
            (alloc.heap.clone(), heap_offset)
        };

        let texture =
            unsafe { heap.newTextureWithDescriptor_offset(&mtl_desc, heap_offset as usize) }
                .ok_or_else(|| {
                    RhiError::TextureCreation("Metal placed texture allocation failed".into())
                })?;

        // Track the buffer for Metal 4 residency.
        self.shared
            .residency_set
            .addAllocation(as_allocation(&texture));
        self.shared.residency_dirty.set(true);

        if let Some(label) = &desc.label {
            use objc2_metal::MTLResource;
            let ns_label = NSString::from_str(label);
            texture.setLabel(Some(&ns_label));
        }

        let id = self.allocate_texture_id()?;
        let idx = id.0 as usize;
        let mut textures = self.shared.textures.borrow_mut();
        if textures.len() <= idx {
            textures.resize_with(idx + 1, || None);
        }
        textures[idx] = Some(texture.clone());
        drop(textures);
        let mut view_flags = self.texture_view_flags.borrow_mut();
        if view_flags.len() <= idx {
            view_flags.resize(idx + 1, false);
        }
        view_flags[idx] = false;
        Self::write_heap_slot(
            &self.shared.texture_heap,
            idx,
            texture.gpuResourceID().to_raw(),
        );

        Ok(Texture {
            id,
            handle: TextureHandle::from_raw(texture.gpuResourceID().to_raw()),
            views: Vec::new(),
            desc: desc.clone(),
            _owner: None,
        })
    }

    pub fn destroy_texture(&self, texture: Texture) {
        let retired = {
            let mut textures = self.shared.textures.borrow_mut();
            let idx = texture.id.0 as usize;
            if idx < textures.len() {
                textures[idx].take()
            } else {
                None
            }
        };
        if let Some(tex) = retired {
            if let Some(is_view) = self
                .texture_view_flags
                .borrow_mut()
                .get_mut(texture.id.0 as usize)
            {
                *is_view = false;
            }
            backend_expect!(&self.rhi_queue.inner, QueueInner::Metal).release_resource(
                MetalRetiredResource::Texture {
                    id: texture.id,
                    texture: tex,
                },
            );
        }
    }

    pub fn destroy_texture_view(&self, id: TextureId) {
        // Claim the view in one borrow, mirroring the Vulkan path.
        let was_view = self
            .texture_view_flags
            .borrow_mut()
            .get_mut(id.0 as usize)
            .is_some_and(|flag| std::mem::replace(flag, false));
        if !was_view {
            return;
        }
        let retired = self
            .shared
            .textures
            .borrow_mut()
            .get_mut(id.0 as usize)
            .and_then(Option::take);
        if let Some(texture) = retired {
            backend_expect!(&self.rhi_queue.inner, QueueInner::Metal)
                .release_resource(MetalRetiredResource::Texture { id, texture });
        }
    }

    pub fn create_sampled_view(
        &self,
        source: &Texture,
        view: &crate::texture::TextureViewDesc,
    ) -> RhiResult<TextureId> {
        self.create_view_internal(source, view)
    }

    pub fn create_storage_view(
        &self,
        source: &Texture,
        view: &crate::texture::TextureViewDesc,
    ) -> RhiResult<TextureId> {
        self.create_view_internal(source, view)
    }

    /// Value to store in a [`TextureHandle`](crate::TextureHandle) root field for sampled view
    /// `id`. On Metal a `DescriptorHandle<Texture2D>` is the texture's `gpuResourceID`, which the
    /// shader uses directly.
    pub fn texture_handle_raw(&self, id: TextureId) -> u64 {
        let textures = self.shared.textures.borrow();
        let texture = textures
            .get(id.0 as usize)
            .and_then(|t| t.as_ref())
            .expect("invalid TextureId");
        texture.gpuResourceID().to_raw()
    }

    fn create_view_internal(
        &self,
        source: &Texture,
        view: &crate::texture::TextureViewDesc,
    ) -> RhiResult<TextureId> {
        use crate::texture::{ALL_LAYERS, ALL_MIPS};
        use objc2_foundation::NSRange;

        let textures_borrow = self.shared.textures.borrow();
        let src_texture = textures_borrow
            .get(source.id.0 as usize)
            .and_then(|t| t.as_ref())
            .ok_or_else(|| {
                RhiError::TextureCreation("create texture view: invalid source TextureId".into())
            })?
            .clone();
        drop(textures_borrow);

        let src_format =
            super::texture::format_to_mtl(view.format.unwrap_or_else(|| source.desc().format));
        let src_type = match source.desc().dimension {
            TextureDimension::D1 => objc2_metal::MTLTextureType::Type1D,
            TextureDimension::D2 => {
                if source.desc().array_layers > 1 {
                    objc2_metal::MTLTextureType::Type2DArray
                } else {
                    objc2_metal::MTLTextureType::Type2D
                }
            }
            TextureDimension::D2Array => objc2_metal::MTLTextureType::Type2DArray,
            TextureDimension::D3 => objc2_metal::MTLTextureType::Type3D,
            TextureDimension::Cube => objc2_metal::MTLTextureType::TypeCube,
            TextureDimension::CubeArray => objc2_metal::MTLTextureType::TypeCubeArray,
        };

        let src_mips = source.desc().mip_levels;
        let src_layers = source.desc().array_layers;

        let mip_start = view.base_mip as usize;
        let mip_count = if view.mip_count == ALL_MIPS {
            (src_mips as usize).saturating_sub(mip_start)
        } else {
            view.mip_count as usize
        };
        let layer_start = view.base_layer as usize;
        let layer_count = if view.layer_count == ALL_LAYERS {
            (src_layers as usize).saturating_sub(layer_start)
        } else {
            view.layer_count as usize
        };

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

        // Views share the source allocation but still need residency tracking.
        self.shared
            .residency_set
            .addAllocation(as_allocation(&view_texture));
        self.shared.residency_dirty.set(true);

        let id = match self.allocate_texture_id() {
            Ok(id) => id,
            Err(err) => {
                self.shared
                    .residency_set
                    .removeAllocation(as_allocation(&view_texture));
                self.shared.residency_dirty.set(true);
                return Err(err);
            }
        };
        let idx = id.0 as usize;
        let resource_id = view_texture.gpuResourceID().to_raw();
        let mut textures = self.shared.textures.borrow_mut();
        if textures.len() <= idx {
            textures.resize_with(idx + 1, || None);
        }
        textures[idx] = Some(view_texture);
        drop(textures);
        let mut view_flags = self.texture_view_flags.borrow_mut();
        if view_flags.len() <= idx {
            view_flags.resize(idx + 1, false);
        }
        view_flags[idx] = true;
        Self::write_heap_slot(&self.shared.texture_heap, idx, resource_id);

        Ok(id)
    }
}
