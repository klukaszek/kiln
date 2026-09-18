//! Swapchain creation, configuration, and image acquisition.

use crate::types::Format;

/// Description for creating/recreating a swapchain.
///
/// Deliberately has no `Default`: the extent has to come from the window, and a default of some
/// arbitrary size is a silent mismatch rather than a convenience. Use [`new`](Self::new).
#[derive(Clone, Debug)]
pub struct SwapchainDesc {
    pub width: u32,
    pub height: u32,
    pub format: Format,
    pub vsync: bool,
    pub image_count: u32,
}

impl SwapchainDesc {
    /// A swapchain of `width` x `height` with the usual settings: an sRGB BGRA surface, no vsync,
    /// and triple buffering.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            format: Format::B8G8R8A8Srgb,
            vsync: false,
            image_count: 3,
        }
    }
}

/// Swapchain for presenting rendered frames.
///
/// `format` and `extent` are the resolved ones — a surface may not support what the descriptor
/// asked for — and live here rather than in each backend, which stored its own copy.
pub struct Swapchain {
    pub(crate) inner: SwapchainInner,
    pub(crate) format: Format,
    pub(crate) extent: [u32; 2],
    pub(crate) _owner: Option<std::rc::Rc<crate::device::DeviceInner>>,
}

backend_enum!(SwapchainInner { vulkan: Box<crate::backend::vulkan::swapchain::VulkanSwapchain>, metal: Box<crate::backend::metal::swapchain::MetalSwapchain> });

impl Swapchain {
    pub(crate) fn new(inner: SwapchainInner, format: Format, extent: [u32; 2]) -> Self {
        Self {
            inner,
            format,
            extent,
            _owner: None,
        }
    }

    pub fn format(&self) -> Format {
        self.format
    }

    pub fn extent(&self) -> [u32; 2] {
        self.extent
    }
}
