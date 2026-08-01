use std::cell::RefCell;
use std::sync::Arc;

use crate::types::Format;
use ash::vk;

/// Vulkan swapchain wrapper.
pub struct VulkanSwapchain {
    pub(crate) swapchain: vk::SwapchainKHR,
    pub(crate) surface: vk::SurfaceKHR,
    pub(crate) images: Arc<[vk::Image]>,
    pub(crate) image_views: Arc<[vk::ImageView]>,
    pub(crate) format: Format,
    pub(crate) surface_format: vk::SurfaceFormatKHR,
    pub(crate) extent: vk::Extent2D,
    pub(crate) depth_image: vk::Image,
    pub(crate) depth_image_view: vk::ImageView,
    pub(crate) depth_image_memory: vk::DeviceMemory,
    pub(crate) present_complete_semaphores: Vec<vk::Semaphore>,
    pub(crate) rendering_complete_semaphores: Vec<vk::Semaphore>,
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
