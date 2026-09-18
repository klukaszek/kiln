//! Vulkan textures: creation, views, and bindless registration.

use ash::vk;

use super::device::{
    IMAGE_LAYOUT, VulkanDevice, format_has_stencil, format_to_vk, is_depth_format,
};
use super::queue::VulkanRetiredResource;

use crate::error::{RhiError, RhiResult};
use crate::texture::{Texture, TextureDesc, TextureSizeAlign, TextureUsage, ViewKind};
use crate::types::{Format, GpuPtr, TextureDimension, TextureHandle, TextureId};

/// A texture under construction, destroyed on drop unless
/// [`into_registered`](PartialTexture::into_registered) disarms it, so a failure part-way through
/// `create_texture` unwinds exactly what it built.
struct PartialTexture<'a> {
    device: &'a ash::Device,
    image: vk::Image,
    image_view: vk::ImageView,
    texture_id: Option<TextureId>,
    textures: &'a super::device::SharedTextures,
}

impl PartialTexture<'_> {
    /// Hand the view over to a registered texture, so `Drop` stops owning any of it.
    fn into_registered(mut self) -> vk::ImageView {
        let view = self.image_view;
        self.image = vk::Image::null();
        self.image_view = vk::ImageView::null();
        self.texture_id = None;
        view
    }
}

impl Drop for PartialTexture<'_> {
    fn drop(&mut self) {
        if let Some(id) = self.texture_id.take() {
            self.textures.recycle(id.0);
        }
        unsafe {
            if self.image_view != vk::ImageView::null() {
                self.device.destroy_image_view(self.image_view, None);
            }
            if self.image != vk::Image::null() {
                self.device.destroy_image(self.image, None);
            }
        }
    }
}

/// Vulkan texture stored in the bindless heap.
pub struct VulkanTexture {
    pub(crate) image: vk::Image,
    pub(crate) image_view: vk::ImageView,
    /// True when this entry is a view into another texture's image.
    /// On destruction, only `image_view` is freed; `image` belongs to the source.
    pub(crate) is_view: bool,
}

impl VulkanDevice {
    pub fn texture_size_align(&self, desc: &TextureDesc) -> RhiResult<TextureSizeAlign> {
        // Query image requirements before allocating backing memory.
        let (image, _, _) = self.create_image_for_desc(desc)?;
        let mem_reqs = unsafe { self.loaders.device.get_image_memory_requirements(image) };
        unsafe {
            self.loaders.device.destroy_image(image, None);
        }
        Ok(TextureSizeAlign {
            size: mem_reqs.size,
            align: mem_reqs.alignment,
        })
    }

    /// Build the `vk::ImageCreateInfo` for `desc` and create the image.
    /// Returns the image plus the effective `array_layers` (cube → ×6) and the format.
    fn create_image_for_desc(&self, desc: &TextureDesc) -> RhiResult<(vk::Image, u32, vk::Format)> {
        let vk_format = format_to_vk(desc.format);

        let mut usage = vk::ImageUsageFlags::empty();
        use crate::texture::TextureUsage;
        let pairs = [
            (TextureUsage::SAMPLED, vk::ImageUsageFlags::SAMPLED),
            (TextureUsage::STORAGE, vk::ImageUsageFlags::STORAGE),
            (
                TextureUsage::COLOR_ATTACHMENT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ),
            (
                TextureUsage::DEPTH_STENCIL_ATTACHMENT,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ),
            (
                TextureUsage::TRANSFER_SRC,
                vk::ImageUsageFlags::TRANSFER_SRC,
            ),
            (
                TextureUsage::TRANSFER_DST,
                vk::ImageUsageFlags::TRANSFER_DST,
            ),
        ];
        for (flag, vk_flag) in pairs {
            if desc.usage.contains(flag) {
                usage |= vk_flag;
            }
        }

        let image_type = match desc.dimension {
            TextureDimension::D1 => vk::ImageType::TYPE_1D,
            TextureDimension::D2
            | TextureDimension::D2Array
            | TextureDimension::Cube
            | TextureDimension::CubeArray => vk::ImageType::TYPE_2D,
            TextureDimension::D3 => vk::ImageType::TYPE_3D,
        };
        let samples = vk::SampleCountFlags::from_raw(desc.sample_count.count());
        // Vulkan counts cube *faces* as array layers; `array_layers` counts cubes.
        let array_layers = desc.dimension.face_count(desc.array_layers);
        let mut image_flags = match desc.dimension {
            TextureDimension::Cube | TextureDimension::CubeArray => {
                vk::ImageCreateFlags::CUBE_COMPATIBLE
            }
            _ => vk::ImageCreateFlags::empty(),
        };
        if desc.usage.contains(TextureUsage::FORMAT_VIEW) {
            image_flags |= vk::ImageCreateFlags::MUTABLE_FORMAT;
        }

        let image_info = vk::ImageCreateInfo::default()
            .flags(image_flags)
            .image_type(image_type)
            .format(vk_format)
            .extent(vk::Extent3D {
                width: desc.width,
                height: desc.height,
                depth: desc.depth,
            })
            .mip_levels(desc.mip_levels)
            .array_layers(array_layers)
            .samples(samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe {
            self.loaders
                .device
                .create_image(&image_info, None)
                .map_err(|e| RhiError::TextureCreation(e.into()))?
        };
        Ok((image, array_layers, vk_format))
    }

    /// Locate the caller's allocation for `texture_gpu` and validate it can back `image`.
    fn resolve_texture_placement(
        &self,
        texture_gpu: GpuPtr<u8>,
    ) -> RhiResult<(vk::DeviceMemory, u64)> {
        // Only the lookup is checked here. Alignment, available size and memory-type
        // compatibility are all conditions `vkBindImageMemory` already validates, and restating
        // them buys nothing the validation layer does not report more precisely.
        let allocations = self.allocations.borrow();
        let alloc = allocations
            .range(..=texture_gpu.address)
            .next_back()
            .map(|(_, alloc)| alloc)
            .filter(|alloc| texture_gpu.address - alloc.base.address < alloc.size)
            .ok_or_else(|| {
                RhiError::TextureCreation(
                    format!(
                        "texture allocation address 0x{:x} was not returned by gpuMalloc",
                        texture_gpu.address
                    )
                    .into(),
                )
            })?;

        let offset = texture_gpu.address - alloc.base.address;
        Ok((alloc.memory, alloc.memory_offset + offset))
    }

    /// Full-resource view of `image`, covering every mip and layer.
    /// The view description for a texture's default view. Attachments need a real `VkImageView`,
    /// while descriptors are written from this info directly, so both are derived from it.
    fn default_view_info(
        desc: &TextureDesc,
        image: vk::Image,
        array_layers: u32,
        vk_format: vk::Format,
    ) -> vk::ImageViewCreateInfo<'static> {
        let view_type = match desc.dimension {
            TextureDimension::D1 => vk::ImageViewType::TYPE_1D,
            TextureDimension::D2 => vk::ImageViewType::TYPE_2D,
            TextureDimension::D2Array => vk::ImageViewType::TYPE_2D_ARRAY,
            TextureDimension::D3 => vk::ImageViewType::TYPE_3D,
            TextureDimension::Cube => vk::ImageViewType::CUBE,
            TextureDimension::CubeArray => vk::ImageViewType::CUBE_ARRAY,
        };
        vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(view_type)
            .format(vk_format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: texture_aspect(desc.format),
                base_mip_level: 0,
                level_count: desc.mip_levels,
                base_array_layer: 0,
                layer_count: array_layers,
            })
    }

    fn create_default_view(&self, info: &vk::ImageViewCreateInfo<'_>) -> RhiResult<vk::ImageView> {
        unsafe { self.loaders.device.create_image_view(info, None) }
            .map_err(|e| RhiError::TextureCreation(e.into()))
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

        let (image, array_layers, vk_format) = self.create_image_for_desc(desc)?;
        // Everything created below is unwound by `partial` if any step fails; nothing else in
        // this function may return early without going through it.
        let mut partial = PartialTexture {
            device: &self.loaders.device,
            image,
            image_view: vk::ImageView::null(),
            texture_id: None,
            textures: &self.textures,
        };

        let (memory, memory_offset) = self.resolve_texture_placement(texture_gpu)?;
        unsafe {
            self.loaders
                .device
                .bind_image_memory(image, memory, memory_offset)
        }
        .map_err(|e| RhiError::TextureCreation(e.into()))?;

        let view_info = Self::default_view_info(desc, image, array_layers, vk_format);
        partial.image_view = self.create_default_view(&view_info)?;
        if let Some(label) = desc.label {
            self.set_object_name(image, label);
            self.set_object_name(partial.image_view, label);
        }

        let id = TextureId(self.textures.allocate_id()?);
        partial.texture_id = Some(id);

        let aspect = texture_aspect(desc.format);
        let transition_aspect = if format_has_stencil(desc.format) {
            aspect | vk::ImageAspectFlags::STENCIL
        } else {
            aspect
        };
        self.initialize_image_layout(image, transition_aspect, desc.mip_levels, array_layers)?;

        // One id is one slot, so the default view is one descriptor type: `SAMPLED` wins when
        // both are present, matching `Texture::gpu()`. Storage form comes from a separate view.
        if desc.usage.contains(TextureUsage::SAMPLED) {
            self.write_image_descriptor(id, &view_info, IMAGE_LAYOUT, false)?;
        } else if desc.usage.contains(TextureUsage::STORAGE) {
            self.write_image_descriptor(id, &view_info, vk::ImageLayout::GENERAL, true)?;
        }

        let image_view = partial.into_registered();
        self.textures.insert(
            id.0,
            VulkanTexture {
                image,
                image_view,
                is_view: false,
            },
        );
        Ok(Texture {
            id,
            handle: TextureHandle::from_raw(id.0 as u64),
            views: Vec::new(),
            desc: TextureDesc {
                label: None,
                ..*desc
            },
            _owner: None,
        })
    }

    pub fn destroy_texture(&self, texture: Texture) {
        let texture_id = texture.id;
        if let Some(texture) = self.textures.take(texture_id.0) {
            self.queue.release_resource(VulkanRetiredResource::Texture {
                id: texture_id,
                texture,
            });
        }
    }

    pub fn destroy_texture_view(&self, id: TextureId) {
        // Taking the entry is itself the claim, so a second call finds nothing to do. Only
        // entries that really are views are claimed; a base texture keeps its slot.
        let retired = self.textures.take_if(id.0, |t| t.is_view);
        if let Some(texture) = retired {
            self.queue
                .release_resource(VulkanRetiredResource::Texture { id, texture });
        }
    }

    /// Value to store in a [`TextureHandle`](crate::TextureHandle) root field for sampled view
    /// `id`. On Vulkan a `DescriptorHandle<Texture2D>` is the bindless heap index, so the handle
    /// is just the id widened to 64 bits.
    pub fn texture_handle_raw(&self, id: TextureId) -> u64 {
        id.0 as u64
    }

    pub fn create_texture_view(
        &self,
        source: &crate::texture::Texture,
        view: &crate::texture::TextureViewDesc,
        kind: ViewKind,
    ) -> RhiResult<TextureId> {
        let storage = kind == ViewKind::Storage;
        let (src_image, src_format, src_aspect, src_mip_levels, src_array_layers, src_view_type) = {
            let src_image = self
                .textures
                .with(source.id.0, |t| t.image)
                .ok_or_else(|| {
                    RhiError::Backend("create texture view: invalid source TextureId".into())
                })?;

            let fmt = format_to_vk(source.desc().format);
            let aspect = if is_depth_format(source.desc().format) {
                vk::ImageAspectFlags::DEPTH
            } else {
                vk::ImageAspectFlags::COLOR
            };
            let mips = source.desc().mip_levels;
            let layers = source
                .desc()
                .dimension
                .face_count(source.desc().array_layers);
            let vt = match source.desc().dimension {
                TextureDimension::D1 => vk::ImageViewType::TYPE_1D,
                TextureDimension::D2 => vk::ImageViewType::TYPE_2D,
                TextureDimension::D2Array => vk::ImageViewType::TYPE_2D_ARRAY,
                TextureDimension::D3 => vk::ImageViewType::TYPE_3D,
                TextureDimension::Cube => vk::ImageViewType::CUBE,
                TextureDimension::CubeArray => vk::ImageViewType::CUBE_ARRAY,
            };
            (src_image, fmt, aspect, mips, layers, vt)
        };

        let vk_format = view.format.map(format_to_vk).unwrap_or(src_format);

        let level_count = view.resolved_mip_count(src_mip_levels);
        let layer_count = view.resolved_layer_count(src_array_layers);

        let view_info = vk::ImageViewCreateInfo::default()
            .image(src_image)
            .view_type(src_view_type)
            .format(vk_format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: src_aspect,
                base_mip_level: view.base_mip as u32,
                level_count,
                base_array_layer: view.base_layer as u32,
                layer_count,
            });

        let image_view = unsafe {
            self.loaders
                .device
                .create_image_view(&view_info, None)
                .map_err(|e| {
                    RhiError::TextureCreation(format!("create texture view: {e}").into())
                })?
        };

        let texture_id = TextureId(self.textures.allocate_id()?);

        let layout = if storage {
            vk::ImageLayout::GENERAL
        } else {
            IMAGE_LAYOUT
        };
        if let Err(err) = self.write_image_descriptor(texture_id, &view_info, layout, storage) {
            unsafe { self.loaders.device.destroy_image_view(image_view, None) };
            self.textures.recycle(texture_id.0);
            return Err(err);
        }

        let vk_texture = VulkanTexture {
            image: vk::Image::null(),
            image_view,
            is_view: true,
        };

        self.textures.insert(texture_id.0, vk_texture);

        Ok(texture_id)
    }
}

/// Copies address one aspect; depth/stencil textures copy their depth plane.
pub(crate) fn texture_aspect(format: Format) -> vk::ImageAspectFlags {
    if is_depth_format(format) {
        vk::ImageAspectFlags::DEPTH
    } else {
        vk::ImageAspectFlags::COLOR
    }
}
