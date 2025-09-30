use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDrawable;

pub struct Drawable { pub(crate) raw: Retained<ProtocolObject<dyn objc2_quartz_core::CAMetalDrawable>> }
impl Drawable {
    pub fn from_surface_current<T: super::MTLDrawableSource + ?Sized>(surface: &T) -> Option<Self> {
        surface.current_drawable().map(|raw| Self { raw })
    }
    pub fn present(&self) { self.raw.present() }
}
