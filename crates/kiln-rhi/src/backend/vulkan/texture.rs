//! Vulkan textures: creation, views, and bindless registration.

use std::cell::Cell;

use ash::vk;

use super::device::{
    IMAGE_LAYOUT, VulkanDevice, format_has_stencil, format_to_vk, is_depth_format,
};
use super::queue::VulkanRetiredResource;

use crate::error::{RhiError, RhiResult};
use crate::queue::QueueInner;
use crate::texture::{Texture, TextureDesc, TextureSizeAlign, TextureUsage};
use crate::types::{
    Format, GpuPtr, MAX_BINDLESS_TEXTURES, SampleCount, TextureDimension, TextureHandle, TextureId,
};

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
        let mem_reqs = unsafe { self.device.get_image_memory_requirements(image) };
        unsafe {
            self.device.destroy_image(image, None);
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
        let samples = match desc.sample_count {
            SampleCount::S1 => vk::SampleCountFlags::TYPE_1,
            SampleCount::S2 => vk::SampleCountFlags::TYPE_2,
            SampleCount::S4 => vk::SampleCountFlags::TYPE_4,
            SampleCount::S8 => vk::SampleCountFlags::TYPE_8,
            SampleCount::S16 => vk::SampleCountFlags::TYPE_16,
        };
        // A single cubemap = 6 faces; n cubes = n × 6. CubeArray callers supply the total.
        let array_layers = match desc.dimension {
            TextureDimension::Cube => desc.array_layers * 6,
            _ => desc.array_layers,
        };
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
            self.device
                .create_image(&image_info, None)
                .map_err(|e| RhiError::TextureCreation(e.to_string()))?
        };
        Ok((image, array_layers, vk_format))
    }

    fn allocate_texture_id(&self) -> RhiResult<TextureId> {
        if let Some(id) = self.free_texture_ids.borrow_mut().pop() {
            return Ok(id);
        }
        let mut next = self.next_texture_id.borrow_mut();
        if *next >= MAX_BINDLESS_TEXTURES {
            return Err(RhiError::TextureCreation(
                "Vulkan bindless texture heap exhausted".into(),
            ));
        }
        let id = TextureId(*next);
        *next += 1;
        Ok(id)
    }

    fn recycle_texture_id(&self, id: TextureId) {
        self.free_texture_ids.borrow_mut().push(id);
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
                RhiError::TextureCreation(format!(
                    "texture allocation address 0x{:x} was not returned by gpuMalloc",
                    texture_gpu.address
                ))
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
        unsafe { self.device.create_image_view(info, None) }
            .map_err(|e| RhiError::TextureCreation(e.to_string()))
    }

    /// Register a freshly created texture in the bindless tables under `id`.
    fn register_texture(&self, id: TextureId, texture: VulkanTexture) {
        let index = id.0 as usize;
        let mut textures = self.textures.borrow_mut();
        if textures.len() <= index {
            textures.resize_with(index + 1, || None);
        }
        textures[index] = Some(texture);
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

        let image_view = Cell::new(vk::ImageView::null());
        let texture_id = Cell::new(None);
        let build = || -> RhiResult<Texture> {
            let (memory, memory_offset) = self.resolve_texture_placement(texture_gpu)?;
            unsafe { self.device.bind_image_memory(image, memory, memory_offset) }
                .map_err(|e| RhiError::TextureCreation(e.to_string()))?;

            let view_info = Self::default_view_info(desc, image, array_layers, vk_format);
            image_view.set(self.create_default_view(&view_info)?);
            let view = image_view.get();
            if let Some(label) = desc.label.as_deref() {
                self.set_object_name(image, label);
                self.set_object_name(view, label);
            }

            let id = self.allocate_texture_id()?;
            texture_id.set(Some(id));

            let aspect = texture_aspect(desc.format);
            let transition_aspect = if format_has_stencil(desc.format) {
                aspect | vk::ImageAspectFlags::STENCIL
            } else {
                aspect
            };
            self.initialize_image_layout(image, transition_aspect, desc.mip_levels, array_layers)?;

            if desc.usage.contains(TextureUsage::SAMPLED) {
                self.write_image_descriptor(id, &view_info, IMAGE_LAYOUT, false)?;
            }
            if desc.usage.contains(TextureUsage::STORAGE) {
                self.write_image_descriptor(id, &view_info, vk::ImageLayout::GENERAL, true)?;
            }

            self.register_texture(
                id,
                VulkanTexture {
                    image,
                    image_view: view,
                    is_view: false,
                },
            );
            Ok(Texture {
                id,
                handle: TextureHandle::from_raw(id.0 as u64),
                views: Vec::new(),
                desc: desc.clone(),
                _owner: None,
            })
        };

        build().inspect_err(|_| {
            if let Some(id) = texture_id.get() {
                self.recycle_texture_id(id);
            }
            unsafe {
                let view = image_view.get();
                if view != vk::ImageView::null() {
                    self.device.destroy_image_view(view, None);
                }
                self.device.destroy_image(image, None);
            }
        })
    }

    pub fn destroy_texture(&self, texture: Texture) {
        let texture_id = texture.id;
        let idx = texture_id.0 as usize;
        let retired = self
            .textures
            .borrow_mut()
            .get_mut(idx)
            .and_then(Option::take);
        if let Some(texture) = retired {
            backend_expect!(&self.queue.inner, QueueInner::Vulkan).release_resource(
                VulkanRetiredResource::Texture {
                    id: texture_id,
                    texture,
                },
            );
        }
    }

    pub fn destroy_texture_view(&self, id: TextureId) {
        let idx = id.0 as usize;
        // Taking the entry is itself the claim, so a second call finds nothing to do. Only
        // entries that really are views are claimed; a base texture keeps its slot.
        let retired = {
            let mut textures = self.textures.borrow_mut();
            match textures.get_mut(idx) {
                Some(slot) if slot.as_ref().is_some_and(|t| t.is_view) => slot.take(),
                _ => None,
            }
        };
        if let Some(texture) = retired {
            backend_expect!(&self.queue.inner, QueueInner::Vulkan)
                .release_resource(VulkanRetiredResource::Texture { id, texture });
        }
    }

    pub fn create_sampled_view(
        &self,
        source: &crate::texture::Texture,
        view: &crate::texture::TextureViewDesc,
    ) -> RhiResult<TextureId> {
        self.create_view_internal(source, view, false)
    }

    pub fn create_storage_view(
        &self,
        source: &crate::texture::Texture,
        view: &crate::texture::TextureViewDesc,
    ) -> RhiResult<TextureId> {
        self.create_view_internal(source, view, true)
    }

    /// Value to store in a [`TextureHandle`](crate::TextureHandle) root field for sampled view
    /// `id`. On Vulkan a `DescriptorHandle<Texture2D>` is the bindless heap index, so the handle
    /// is just the id widened to 64 bits.
    pub fn texture_handle_raw(&self, id: TextureId) -> u64 {
        id.0 as u64
    }

    fn create_view_internal(
        &self,
        source: &crate::texture::Texture,
        view: &crate::texture::TextureViewDesc,
        storage: bool,
    ) -> RhiResult<TextureId> {
        use crate::texture::{ALL_LAYERS, ALL_MIPS};

        let (src_image, src_format, src_aspect, src_mip_levels, src_array_layers, src_view_type) = {
            let textures = self.textures.borrow();
            let src = textures
                .get(source.id.0 as usize)
                .and_then(|t| t.as_ref())
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
            let layers = match source.desc().dimension {
                TextureDimension::Cube => source.desc().array_layers * 6,
                _ => source.desc().array_layers,
            };
            let vt = match source.desc().dimension {
                TextureDimension::D1 => vk::ImageViewType::TYPE_1D,
                TextureDimension::D2 => vk::ImageViewType::TYPE_2D,
                TextureDimension::D2Array => vk::ImageViewType::TYPE_2D_ARRAY,
                TextureDimension::D3 => vk::ImageViewType::TYPE_3D,
                TextureDimension::Cube => vk::ImageViewType::CUBE,
                TextureDimension::CubeArray => vk::ImageViewType::CUBE_ARRAY,
            };
            (src.image, fmt, aspect, mips, layers, vt)
        };

        let vk_format = view.format.map(format_to_vk).unwrap_or(src_format);

        let level_count = if view.mip_count == ALL_MIPS {
            src_mip_levels.saturating_sub(view.base_mip as u32)
        } else {
            view.mip_count as u32
        };
        let layer_count = if view.layer_count == ALL_LAYERS {
            src_array_layers.saturating_sub(view.base_layer as u32)
        } else {
            view.layer_count as u32
        };

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
            self.device
                .create_image_view(&view_info, None)
                .map_err(|e| RhiError::TextureCreation(format!("create texture view: {e}")))?
        };

        let texture_id = self.allocate_texture_id()?;

        let layout = if storage {
            vk::ImageLayout::GENERAL
        } else {
            IMAGE_LAYOUT
        };
        if let Err(err) = self.write_image_descriptor(texture_id, &view_info, layout, storage) {
            unsafe { self.device.destroy_image_view(image_view, None) };
            self.recycle_texture_id(texture_id);
            return Err(err);
        }

        let vk_texture = VulkanTexture {
            image: vk::Image::null(),
            image_view,
            is_view: true,
        };

        {
            let mut textures = self.textures.borrow_mut();
            if textures.len() <= texture_id.0 as usize {
                textures.resize_with(texture_id.0 as usize + 1, || None);
            }
            textures[texture_id.0 as usize] = Some(vk_texture);
        }

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
