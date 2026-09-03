//! Swapchain creation, configuration, and image acquisition.

use crate::types::Format;

/// Description for creating/recreating a swapchain.
#[derive(Clone, Debug)]
pub struct SwapchainDesc {
    pub width: u32,
    pub height: u32,
    pub format: Format,
    pub vsync: bool,
    pub image_count: u32,
}

impl Default for SwapchainDesc {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            format: Format::B8G8R8A8Srgb,
            vsync: false,
            image_count: 3,
        }
    }
}

/// Swapchain for presenting rendered frames.
pub struct Swapchain {
    pub(crate) inner: SwapchainInner,
    pub(crate) _owner: Option<std::rc::Rc<crate::device::DeviceInner>>,
}

pub(crate) enum SwapchainInner {
    #[cfg(feature = "vulkan")]
    Vulkan(Box<crate::backend::vulkan::swapchain::VulkanSwapchain>),
    #[cfg(feature = "metal")]
    Metal(Box<crate::backend::metal::swapchain::MetalSwapchain>),
}

impl Swapchain {
    pub fn format(&self) -> Format {
        backend_dispatch!(&self.inner, SwapchainInner, sc => sc.format)
    }

    pub fn extent(&self) -> [u32; 2] {
        match &self.inner {
            #[cfg(feature = "vulkan")]
            SwapchainInner::Vulkan(sc) => [sc.extent.width, sc.extent.height],
            #[cfg(feature = "metal")]
            SwapchainInner::Metal(sc) => sc.extent,
        }
    }
}

/// An acquired swapchain image, ready for rendering.
pub struct AcquiredImage {
    pub index: u32,
    pub format: Format,
    pub width: u32,
    pub height: u32,
}
