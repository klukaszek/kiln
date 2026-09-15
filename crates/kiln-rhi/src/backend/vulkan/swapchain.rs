//! Vulkan swapchain: construction, recreation, and per-frame resources.

use std::cell::RefCell;
use std::rc::Rc;

use ash::vk;

use super::device::{VulkanDevice, find_memorytype_index, format_to_vk, vk_to_format};

use crate::error::{RhiError, RhiResult};
use crate::queue::QueueInner;
use crate::surface::{Surface, SurfaceInner};
use crate::swapchain::{Swapchain, SwapchainDesc, SwapchainInner};
use crate::types::Format;
use crate::types::MAX_FRAMES_IN_FLIGHT;

/// Vulkan swapchain wrapper.
pub struct VulkanSwapchain {
    pub(crate) swapchain: vk::SwapchainKHR,
    pub(crate) surface: vk::SurfaceKHR,
    pub(crate) images: Rc<[vk::Image]>,
    pub(crate) image_views: Rc<[vk::ImageView]>,
    pub(crate) format: Format,
    pub(crate) surface_format: vk::SurfaceFormatKHR,
    pub(crate) extent: vk::Extent2D,
    pub(crate) depth_image: vk::Image,
    pub(crate) depth_image_view: vk::ImageView,
    pub(crate) depth_image_memory: vk::DeviceMemory,
    pub(crate) present_complete_semaphores: Vec<vk::Semaphore>,
    pub(crate) rendering_complete_semaphores: Vec<vk::Semaphore>,
    /// Acquisitions survive abandoned recording so a retry does not re-signal its semaphore.
    pub(crate) acquired_images: RefCell<Vec<Option<u32>>>,
    pub(crate) in_flight_fences: Vec<vk::Fence>,
    pub(crate) in_flight_cmd_buffers: RefCell<Vec<vk::CommandBuffer>>,
    // Keep the loaders with the swapchain; the device must outlive it.
    pub(crate) device: ash::Device,
    pub(crate) swapchain_loader: ash::khr::swapchain::Device,
}

impl Drop for VulkanSwapchain {
    fn drop(&mut self) {
        unsafe {
            for &view in self.image_views.iter() {
                self.device.destroy_image_view(view, None);
            }
            self.device.destroy_image_view(self.depth_image_view, None);
            self.device.destroy_image(self.depth_image, None);
            self.device.free_memory(self.depth_image_memory, None);
            for &sem in &self.present_complete_semaphores {
                self.device.destroy_semaphore(sem, None);
            }
            for &sem in &self.rendering_complete_semaphores {
                self.device.destroy_semaphore(sem, None);
            }
            for &fence in &self.in_flight_fences {
                self.device.destroy_fence(fence, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }
}

/// Components produced by `build_swapchain_contents` (shared between create and recreate).
struct SwapchainContents {
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    extent: vk::Extent2D,
    depth_image: vk::Image,
    depth_image_view: vk::ImageView,
    depth_image_memory: vk::DeviceMemory,
    present_complete_semaphores: Vec<vk::Semaphore>,
    rendering_complete_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    in_flight_cmd_buffers: Vec<vk::CommandBuffer>,
}

impl VulkanDevice {
    pub fn create_swapchain(
        &self,
        surface: &Surface,
        desc: &SwapchainDesc,
    ) -> RhiResult<Swapchain> {
        let vk_surface = backend_expect!(&surface.inner, SurfaceInner::Vulkan).surface;

        let surface_formats = unsafe {
            self.surface_loader
                .get_physical_device_surface_formats(self.physical_device, vk_surface)
                .map_err(|e| RhiError::SwapchainCreation(e.to_string()))?
        };
        let desired_vk_format = format_to_vk(desc.format);
        let surface_format = surface_formats
            .iter()
            .find(|f| f.format == desired_vk_format)
            .cloned()
            .unwrap_or(surface_formats[0]);

        let SwapchainContents {
            swapchain,
            images,
            image_views,
            extent,
            depth_image,
            depth_image_view,
            depth_image_memory,
            present_complete_semaphores,
            rendering_complete_semaphores,
            in_flight_fences,
            in_flight_cmd_buffers,
        } = self.build_swapchain_contents(
            vk_surface,
            surface_format,
            desc,
            vk::SwapchainKHR::null(),
        )?;

        Ok(Swapchain {
            inner: SwapchainInner::Vulkan(Box::new(VulkanSwapchain {
                swapchain,
                surface: vk_surface,
                images: images.into(),
                image_views: image_views.into(),
                format: vk_to_format(surface_format.format),
                surface_format,
                extent,
                depth_image,
                depth_image_view,
                depth_image_memory,
                present_complete_semaphores,
                rendering_complete_semaphores,
                in_flight_fences,
                in_flight_cmd_buffers: RefCell::new(in_flight_cmd_buffers),
                acquired_images: RefCell::new(vec![None; MAX_FRAMES_IN_FLIGHT]),
                device: self.device.clone(),
                swapchain_loader: self.swapchain_loader.clone(),
            })),
            _owner: None,
        })
    }

    pub fn recreate_swapchain(
        &self,
        swapchain: &mut Swapchain,
        desc: &SwapchainDesc,
    ) -> RhiResult<()> {
        unsafe {
            self.device
                .device_wait_idle()
                .map_err(|e| RhiError::Backend(e.to_string()))?
        };

        // Nothing may reference the images about to be destroyed.
        self.flush_setup_barriers()?;

        let sc = backend_expect!(&mut swapchain.inner, SwapchainInner::Vulkan);

        let old_swapchain = sc.swapchain;
        let surface = sc.surface;
        let surface_format = sc.surface_format;

        let contents =
            self.build_swapchain_contents(surface, surface_format, desc, old_swapchain)?;

        unsafe {
            self.device.destroy_image_view(sc.depth_image_view, None);
            self.device.destroy_image(sc.depth_image, None);
            self.device.free_memory(sc.depth_image_memory, None);
            for &view in sc.image_views.iter() {
                self.device.destroy_image_view(view, None);
            }
        }
        {
            let mut cmd_buffers = sc.in_flight_cmd_buffers.borrow_mut();
            let to_free: Vec<_> = cmd_buffers
                .iter()
                .copied()
                .filter(|c| *c != vk::CommandBuffer::null())
                .collect();
            if !to_free.is_empty() {
                for command_buffer in to_free {
                    self.recycle_command_buffer(command_buffer);
                }
            }
            cmd_buffers.clear();
        }
        unsafe {
            for &sem in &sc.present_complete_semaphores {
                self.device.destroy_semaphore(sem, None);
            }
            for &sem in &sc.rendering_complete_semaphores {
                self.device.destroy_semaphore(sem, None);
            }
            for &fence in &sc.in_flight_fences {
                self.device.destroy_fence(fence, None);
            }
        }

        unsafe {
            self.swapchain_loader.destroy_swapchain(old_swapchain, None);
        }

        sc.swapchain = contents.swapchain;
        sc.images = contents.images.into();
        sc.image_views = contents.image_views.into();
        sc.extent = contents.extent;
        sc.depth_image = contents.depth_image;
        sc.depth_image_view = contents.depth_image_view;
        sc.depth_image_memory = contents.depth_image_memory;
        sc.present_complete_semaphores = contents.present_complete_semaphores;
        sc.rendering_complete_semaphores = contents.rendering_complete_semaphores;
        sc.in_flight_fences = contents.in_flight_fences;
        sc.in_flight_cmd_buffers = RefCell::new(contents.in_flight_cmd_buffers);
        sc.acquired_images.borrow_mut().fill(None);
        backend_expect!(&self.queue.inner, QueueInner::Vulkan)
            .frame_fence_armed
            .borrow_mut()
            .fill(false);

        Ok(())
    }

    fn build_swapchain_contents(
        &self,
        surface: vk::SurfaceKHR,
        surface_format: vk::SurfaceFormatKHR,
        desc: &SwapchainDesc,
        old_swapchain: vk::SwapchainKHR,
    ) -> RhiResult<SwapchainContents> {
        let caps = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, surface)
                .map_err(|e| RhiError::SwapchainCreation(e.to_string()))?
        };

        let mut image_count = desc.image_count.max(caps.min_image_count);
        if caps.max_image_count > 0 {
            image_count = image_count.min(caps.max_image_count);
        }

        let extent = if caps.current_extent.width == u32::MAX {
            vk::Extent2D {
                width: desc.width,
                height: desc.height,
            }
        } else {
            caps.current_extent
        };

        let pre_transform = if caps
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            caps.current_transform
        };

        let present_mode = if desc.vsync {
            vk::PresentModeKHR::FIFO
        } else {
            unsafe {
                self.surface_loader
                    .get_physical_device_surface_present_modes(self.physical_device, surface)
                    .map_err(|e| RhiError::SwapchainCreation(e.to_string()))?
            }
            .into_iter()
            .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO)
        };

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_color_space(surface_format.color_space)
            .image_format(surface_format.format)
            .image_extent(extent)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(pre_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .image_array_layers(1)
            .old_swapchain(old_swapchain);

        let swapchain = unsafe {
            self.swapchain_loader
                .create_swapchain(&create_info, None)
                .map_err(|e| RhiError::SwapchainCreation(e.to_string()))?
        };
        let images = unsafe {
            self.swapchain_loader
                .get_swapchain_images(swapchain)
                .map_err(|e| RhiError::SwapchainCreation(e.to_string()))?
        };
        let image_views = self.create_swapchain_image_views(&images, surface_format.format)?;
        let (depth_image, depth_image_view, depth_image_memory) =
            self.create_depth_buffer(extent.width, extent.height)?;

        let sem_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let mk_sem = || unsafe {
            self.device
                .create_semaphore(&sem_info, None)
                .map_err(|e| RhiError::SwapchainCreation(e.to_string()))
        };
        let mk_fence = || unsafe {
            self.device
                .create_fence(&fence_info, None)
                .map_err(|e| RhiError::SwapchainCreation(e.to_string()))
        };

        let mut present_complete_semaphores = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut in_flight_fences = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            present_complete_semaphores.push(mk_sem()?);
            in_flight_fences.push(mk_fence()?);
        }
        let mut rendering_complete_semaphores = Vec::with_capacity(images.len());
        for _ in 0..images.len() {
            rendering_complete_semaphores.push(mk_sem()?);
        }

        Ok(SwapchainContents {
            swapchain,
            images,
            image_views,
            extent,
            depth_image,
            depth_image_view,
            depth_image_memory,
            present_complete_semaphores,
            rendering_complete_semaphores,
            in_flight_fences,
            in_flight_cmd_buffers: vec![vk::CommandBuffer::null(); MAX_FRAMES_IN_FLIGHT],
        })
    }

    fn create_swapchain_image_views(
        &self,
        images: &[vk::Image],
        format: vk::Format,
    ) -> RhiResult<Vec<vk::ImageView>> {
        images
            .iter()
            .map(|&image| {
                let view_info = vk::ImageViewCreateInfo::default()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image(image);
                unsafe {
                    self.device
                        .create_image_view(&view_info, None)
                        .map_err(|e| RhiError::SwapchainCreation(e.to_string()))
                }
            })
            .collect()
    }

    fn create_depth_buffer(
        &self,
        width: u32,
        height: u32,
    ) -> RhiResult<(vk::Image, vk::ImageView, vk::DeviceMemory)> {
        let depth_image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let depth_image = unsafe {
            self.device
                .create_image(&depth_image_info, None)
                .map_err(|e| RhiError::SwapchainCreation(format!("Depth image: {e}")))?
        };

        let mem_reqs = unsafe { self.device.get_image_memory_requirements(depth_image) };
        let mem_index = find_memorytype_index(
            &mem_reqs,
            &self.device_memory_properties,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or_else(|| RhiError::AllocationFailed("No memory for depth".into()))?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_index);

        let depth_memory = unsafe {
            self.device
                .allocate_memory(&alloc_info, None)
                .map_err(|e| RhiError::AllocationFailed(e.to_string()))?
        };

        unsafe {
            self.device
                .bind_image_memory(depth_image, depth_memory, 0)
                .map_err(|e| RhiError::SwapchainCreation(e.to_string()))?;
        }

        self.initialize_image_layout(depth_image, vk::ImageAspectFlags::DEPTH, 1, 1)?;
        // Submit straight away rather than batching: this image is destroyed on the next
        // swapchain rebuild, which would invalidate the setup buffer while it still held this
        // barrier. Batching only pays off for the many images of a scene load, not for one depth
        // buffer per swapchain.
        self.flush_setup_barriers()?;

        let view_info = vk::ImageViewCreateInfo::default()
            .image(depth_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let depth_view = unsafe {
            self.device
                .create_image_view(&view_info, None)
                .map_err(|e| RhiError::SwapchainCreation(format!("Depth view: {e}")))?
        };

        Ok((depth_image, depth_view, depth_memory))
    }
}
