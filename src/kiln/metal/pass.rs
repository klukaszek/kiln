use objc2::rc::Retained;

use super::{MTLClearColor, MTLLoadAction};

pub struct RenderPass { pub(crate) raw: Retained<objc2_metal::MTL4RenderPassDescriptor> }
impl RenderPass {
    pub fn from_surface_current<T: super::MTLDrawableSource + ?Sized>(surface: &T) -> Option<Self> {
        surface.current_mtl4_render_pass_descriptor().map(|raw| Self { raw })
    }
    pub fn set_clear(&self, color: MTLClearColor, load: MTLLoadAction) {
        unsafe {
            let ca0 = self.raw.colorAttachments().objectAtIndexedSubscript(0);
            ca0.setLoadAction(load);
            ca0.setClearColor(color);
        }
    }
}
