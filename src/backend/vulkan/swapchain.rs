use std::cell::RefCell;

use crate::types::Format;
use ash::vk;

/// Vulkan swapchain wrapper.
pub struct VulkanSwapchain {
    pub(crate) swapchain: vk::SwapchainKHR,
    pub(crate) surface: vk::SurfaceKHR,
    pub(crate) images: Vec<vk::Image>,
    pub(crate) image_views: Vec<vk::ImageView>,
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
}
