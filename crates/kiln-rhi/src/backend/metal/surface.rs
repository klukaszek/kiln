use objc2::rc::Retained;
use objc2_quartz_core::CAMetalLayer;

pub struct MetalSurface {
    pub(crate) layer: Retained<CAMetalLayer>,
}
