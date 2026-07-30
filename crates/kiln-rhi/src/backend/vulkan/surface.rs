use ash::vk;

/// Vulkan surface wrapper.
pub struct VulkanSurface {
    pub(crate) surface: vk::SurfaceKHR,
    // Keep the loader with the surface; the device must outlive both.
    pub(crate) surface_loader: ash::khr::surface::Instance,
}

impl Drop for VulkanSurface {
    fn drop(&mut self) {
        unsafe {
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}
