//! Vulkan swapchain: construction, recreation, and per-frame resources.

use std::cell::RefCell;
use std::rc::Rc;

use ash::vk;

use super::device::{VulkanDevice, format_to_vk, vk_to_format};

use super::surface::VulkanSurface;
use crate::error::{RhiError, RhiResult};
use crate::swapchain::{Swapchain, SwapchainDesc};
use crate::types::MAX_FRAMES_IN_FLIGHT;

/// Everything a swapchain owns per in-flight frame slot, indexed by `frame_index`.
///
/// The driver's image index is a separate axis: `images`, `image_views` and
/// `rendering_complete_semaphores` are indexed by it and are a different length. Keeping the two
/// sets apart is the point — a slot's semaphore and its image's semaphore are not interchangeable.
#[derive(Clone, Copy)]
pub(crate) struct FrameSlot {
    pub(crate) present_complete: vk::Semaphore,
    pub(crate) fence: vk::Fence,
    /// The image this slot acquired, held across abandoned recording so a retry does not
    /// re-signal `present_complete`.
    pub(crate) acquired_image: Option<u32>,
    /// The frame's command buffer, recycled on the slot's next acquire. It belongs to the
    /// device's pool, so `Drop` cannot hand it back.
    pub(crate) command_buffer: vk::CommandBuffer,
}

/// Vulkan swapchain wrapper.
pub struct VulkanSwapchain {
    pub(crate) swapchain: vk::SwapchainKHR,
    pub(crate) surface: vk::SurfaceKHR,
    pub(crate) images: Rc<[vk::Image]>,
    pub(crate) image_views: Rc<[vk::ImageView]>,
    pub(crate) surface_format: vk::SurfaceFormatKHR,
    /// Per image, not per frame slot.
    pub(crate) rendering_complete_semaphores: Box<[vk::Semaphore]>,
    pub(crate) frames: RefCell<[FrameSlot; MAX_FRAMES_IN_FLIGHT]>,
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
            for &sem in &self.rendering_complete_semaphores {
                self.device.destroy_semaphore(sem, None);
            }
            for slot in self.frames.borrow().iter() {
                self.device.destroy_semaphore(slot.present_complete, None);
                self.device.destroy_fence(slot.fence, None);
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
    rendering_complete_semaphores: Box<[vk::Semaphore]>,
    frames: [FrameSlot; MAX_FRAMES_IN_FLIGHT],
}

impl VulkanDevice {
    pub fn create_swapchain(
        &self,
        surface: &VulkanSurface,
        desc: &SwapchainDesc,
    ) -> RhiResult<Swapchain> {
        let vk_surface = surface.surface;

        let surface_formats = unsafe {
            self.surface_loader
                .get_physical_device_surface_formats(self.physical_device, vk_surface)
                .map_err(|e| RhiError::SwapchainCreation(e.into()))?
        };
        // Only formats the RHI can name are candidates: `Swapchain::format()` reports this back
        // to the application, which builds pipelines against it, so an unnameable format would
        // have to be guessed at and would then mismatch every attachment.
        let desired_vk_format = format_to_vk(desc.format);
        let surface_format = surface_formats
            .iter()
            .find(|f| f.format == desired_vk_format)
            .or_else(|| {
                surface_formats
                    .iter()
                    .find(|f| vk_to_format(f.format).is_some())
            })
            .copied()
            .ok_or_else(|| {
                RhiError::SwapchainCreation(
                    format!(
                        "the surface supports no format kiln-rhi can name (offered: {:?})",
                        surface_formats.iter().map(|f| f.format).collect::<Vec<_>>()
                    )
                    .into(),
                )
            })?;
        let format = vk_to_format(surface_format.format).ok_or_else(|| {
            RhiError::SwapchainCreation(
                format!(
                    "chosen surface format {:?} has no kiln-rhi Format",
                    surface_format.format
                )
                .into(),
            )
        })?;

        let contents = self.build_swapchain_contents(
            vk_surface,
            surface_format,
            desc,
            vk::SwapchainKHR::null(),
        )?;

        let extent = contents.extent;
        Ok(Swapchain::new(
            Box::new(self.assemble_swapchain(contents, vk_surface, surface_format)),
            format,
            [extent.width, extent.height],
        ))
    }

    /// Wrap freshly built contents; `Drop` owns tearing them down again.
    fn assemble_swapchain(
        &self,
        contents: SwapchainContents,
        surface: vk::SurfaceKHR,
        surface_format: vk::SurfaceFormatKHR,
    ) -> VulkanSwapchain {
        VulkanSwapchain {
            swapchain: contents.swapchain,
            surface,
            images: contents.images.into(),
            image_views: contents.image_views.into(),
            surface_format,
            rendering_complete_semaphores: contents.rendering_complete_semaphores,
            frames: RefCell::new(contents.frames),
            device: self.loaders.device.clone(),
            swapchain_loader: self.swapchain_loader.clone(),
        }
    }

    pub fn recreate_swapchain(
        &self,
        swapchain: &mut Swapchain,
        desc: &SwapchainDesc,
    ) -> RhiResult<()> {
        // SAFETY: everything below destroys swapchain-owned objects, so nothing may be in flight.
        unsafe { self.loaders.device.device_wait_idle() }
            .map_err(|e| RhiError::Backend(e.into()))?;

        // Nothing may reference the images about to be destroyed.
        self.flush_setup_barriers()?;

        let sc = &mut swapchain.inner;

        for slot in sc.frames.borrow_mut().iter_mut() {
            let command_buffer =
                std::mem::replace(&mut slot.command_buffer, vk::CommandBuffer::null());
            if command_buffer != vk::CommandBuffer::null() {
                self.recycle_command_buffer(command_buffer);
            }
        }

        let (surface, surface_format) = (sc.surface, sc.surface_format);
        let contents =
            self.build_swapchain_contents(surface, surface_format, desc, sc.swapchain)?;
        let extent = contents.extent;

        // Assigning drops the old swapchain, and its `Drop` destroys exactly the views,
        // semaphores, fences and handle that creating a new set replaced. The old handle is
        // still live until here, which is what `oldSwapchain` above requires.
        **sc = self.assemble_swapchain(contents, surface, surface_format);

        // The fences the old slots armed went with them.
        for frame in self.queue.frames.borrow_mut().iter_mut() {
            frame.armed = false;
        }

        // The surface may clamp the requested size, so report what was actually created.
        swapchain.extent = [extent.width, extent.height];
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
                .map_err(|e| RhiError::SwapchainCreation(e.into()))?
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
                    .map_err(|e| RhiError::SwapchainCreation(e.into()))?
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
                .map_err(|e| RhiError::SwapchainCreation(e.into()))?
        };
        let images = unsafe {
            self.swapchain_loader
                .get_swapchain_images(swapchain)
                .map_err(|e| RhiError::SwapchainCreation(e.into()))?
        };
        let image_views = self.create_swapchain_image_views(&images, surface_format.format)?;

        let sem_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let mk_sem = || unsafe {
            self.loaders
                .device
                .create_semaphore(&sem_info, None)
                .map_err(|e| RhiError::SwapchainCreation(e.into()))
        };
        let mk_fence = || unsafe {
            self.loaders
                .device
                .create_fence(&fence_info, None)
                .map_err(|e| RhiError::SwapchainCreation(e.into()))
        };

        let mut frames = [FrameSlot {
            present_complete: vk::Semaphore::null(),
            fence: vk::Fence::null(),
            acquired_image: None,
            command_buffer: vk::CommandBuffer::null(),
        }; MAX_FRAMES_IN_FLIGHT];
        for slot in &mut frames {
            slot.present_complete = mk_sem()?;
            slot.fence = mk_fence()?;
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
            rendering_complete_semaphores: rendering_complete_semaphores.into_boxed_slice(),
            frames,
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
                    self.loaders
                        .device
                        .create_image_view(&view_info, None)
                        .map_err(|e| RhiError::SwapchainCreation(e.into()))
                }
            })
            .collect()
    }
}
