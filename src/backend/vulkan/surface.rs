use ash::vk;

/// Vulkan surface wrapper.
pub struct VulkanSurface {
    pub(crate) surface: vk::SurfaceKHR,
    // Clone of the surface loader so `Drop` is self-contained. The owning `VulkanDevice` must
    // outlive this (it destroys the `VkInstance`); the harness drops the device last.
    pub(crate) surface_loader: ash::khr::surface::Instance,
}

impl Drop for VulkanSurface {
    fn drop(&mut self) {
        unsafe {
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}
