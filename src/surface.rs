use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

/// Description for creating a surface.
pub struct SurfaceDesc {
    pub display_handle: RawDisplayHandle,
    pub window_handle: RawWindowHandle,
}

/// Platform surface for presentation.
pub struct Surface {
    pub(crate) inner: SurfaceInner,
}

pub(crate) enum SurfaceInner {
    #[cfg(feature = "vulkan")]
    Vulkan(crate::backend::vulkan::surface::VulkanSurface),
    #[cfg(feature = "metal")]
    Metal(crate::backend::metal::surface::MetalSurface),
}
