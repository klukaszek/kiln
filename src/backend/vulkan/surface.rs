use ash::vk;

/// Vulkan surface wrapper.
pub struct VulkanSurface {
    pub(crate) surface: vk::SurfaceKHR,
}
