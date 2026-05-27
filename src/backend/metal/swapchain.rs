use std::cell::RefCell;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDrawable, MTLTexture};
use objc2_quartz_core::CAMetalLayer;

use crate::types::Format;

pub struct MetalSwapchain {
    pub(crate) layer: Retained<CAMetalLayer>,
    pub(crate) format: Format,
    pub(crate) extent: [u32; 2],
    pub(crate) depth_texture: Retained<ProtocolObject<dyn MTLTexture>>,
    /// Current drawable acquired for this frame (set by acquire_image, consumed by present).
    pub(crate) current_drawable: RefCell<Option<Retained<ProtocolObject<dyn MTLDrawable>>>>,
    /// Texture from the current drawable for rendering.
    pub(crate) current_drawable_texture: RefCell<Option<Retained<ProtocolObject<dyn MTLTexture>>>>,
}
