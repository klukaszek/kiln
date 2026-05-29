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
}

pub(crate) enum SwapchainInner {
    #[cfg(feature = "vulkan")]
    Vulkan(crate::backend::vulkan::swapchain::VulkanSwapchain),
    #[cfg(feature = "metal")]
    Metal(crate::backend::metal::swapchain::MetalSwapchain),
}

impl Swapchain {
    /// Get the swapchain color format.
    pub fn format(&self) -> Format {
        backend_dispatch!(&self.inner, SwapchainInner, sc => sc.format)
    }

    /// Get the swapchain extent [width, height].
    pub fn extent(&self) -> [u32; 2] {
        // Divergent per backend: Vulkan stores a `vk::Extent2D`, Metal a `[u32; 2]`.
        match &self.inner {
            #[cfg(feature = "vulkan")]
            SwapchainInner::Vulkan(sc) => [sc.extent.width, sc.extent.height],
            #[cfg(feature = "metal")]
            SwapchainInner::Metal(sc) => sc.extent,
        }
    }

    /// Get raw Vulkan swapchain image views for escape-hatch scenarios (e.g. ImGui framebuffers).
    #[cfg(feature = "vulkan")]
    pub fn vulkan_image_views(&self) -> &[ash::vk::ImageView] {
        match &self.inner {
            SwapchainInner::Vulkan(sc) => &sc.image_views,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    /// Get the Vulkan swapchain extent for escape-hatch scenarios.
    #[cfg(feature = "vulkan")]
    pub fn vulkan_extent(&self) -> ash::vk::Extent2D {
        match &self.inner {
            SwapchainInner::Vulkan(sc) => sc.extent,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

/// An acquired swapchain image, ready for rendering.
pub struct AcquiredImage {
    /// Index into the swapchain images.
    pub index: u32,
    /// The format of the acquired image.
    pub format: Format,
    /// Width of the image.
    pub width: u32,
    /// Height of the image.
    pub height: u32,
}
