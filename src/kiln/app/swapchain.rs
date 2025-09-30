pub use crate::metal::MTLPixelFormat as PixelFormatReexport;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;

#[derive(Copy, Clone, Debug)]
pub enum PresentMode { Fifo, Immediate }

#[derive(Copy, Clone, Debug)]
pub enum ColorSpace { SRGB, DisplayP3, ExtendedSRGB }

#[derive(Copy, Clone, Debug)]
pub struct SwapchainConfig {
    pub pixel_format: crate::metal::MTLPixelFormat,
    pub framebuffer_only: bool,
    pub max_drawables: u32,
    pub present_mode: PresentMode,
    pub colorspace: ColorSpace,
    pub wants_edr: bool,
}
impl Default for SwapchainConfig { fn default() -> Self { Self { pixel_format: crate::metal::MTLPixelFormat::BGRA8Unorm, framebuffer_only: true, max_drawables: 3, present_mode: PresentMode::Fifo, colorspace: ColorSpace::SRGB, wants_edr: false } } }

pub trait RenderSurface {
    fn current_mtl4_render_pass_descriptor(&self) -> Option<Retained<objc2_metal::MTL4RenderPassDescriptor>>;
    fn current_drawable(&self) -> Option<Retained<ProtocolObject<dyn objc2_quartz_core::CAMetalDrawable>>>;
    fn device(&self) -> Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>;
    fn color_pixel_format(&self) -> crate::metal::MTLPixelFormat;
}
