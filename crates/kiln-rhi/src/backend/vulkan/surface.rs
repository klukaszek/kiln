use super::device::VulkanDevice;
use crate::error::{RhiError, RhiResult};
use crate::surface::{Surface, SurfaceDesc};
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

impl VulkanDevice {
    pub fn create_surface(&self, desc: &SurfaceDesc) -> RhiResult<Surface> {
        let factory =
            ash_window::SurfaceFactory::new(&self.entry, &self.instance, desc.display_handle)
                .map_err(|e| RhiError::SurfaceCreation(e.into()))?;
        let surface = unsafe {
            factory
                .create_surface(desc.window_handle, None)
                .map_err(|e| RhiError::SurfaceCreation(e.into()))?
        };

        Ok(Surface {
            inner: VulkanSurface {
                surface,
                surface_loader: self.surface_loader.clone(),
            },
            _owner: None,
        })
    }
}
