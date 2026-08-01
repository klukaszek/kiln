use std::cell::RefCell;
use std::rc::Rc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDrawable, MTLTexture};
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};

use crate::types::Format;

/// The layer plus the drawable the current frame has taken from it. Shared with the frame's
/// command buffer so the drawable is taken at first encode rather than up front: `nextDrawable`
/// blocks once the pool is empty, so holding one across the whole CPU frame widens the stall.
pub(crate) struct MetalDrawableSlot {
    pub(crate) layer: Retained<CAMetalLayer>,
    drawable: RefCell<Option<Retained<ProtocolObject<dyn MTLDrawable>>>>,
    texture: RefCell<Option<Retained<ProtocolObject<dyn MTLTexture>>>>,
}

pub(crate) type SharedDrawableSlot = Rc<MetalDrawableSlot>;

impl MetalDrawableSlot {
    pub(crate) fn new(layer: Retained<CAMetalLayer>) -> Self {
        Self {
            layer,
            drawable: RefCell::new(None),
            texture: RefCell::new(None),
        }
    }

    /// This frame's drawable texture, taking a drawable on first call. `None` when the layer has
    /// none to give; the caller skips the attachment and the next frame retries.
    pub(crate) fn texture(&self) -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
        if let Some(texture) = self.texture.borrow().as_ref() {
            return Some(texture.clone());
        }
        let drawable = self.layer.nextDrawable()?;
        let texture = drawable.texture();
        let drawable: Retained<ProtocolObject<dyn MTLDrawable>> =
            ProtocolObject::from_retained(drawable);
        *self.texture.borrow_mut() = Some(texture.clone());
        *self.drawable.borrow_mut() = Some(drawable);
        Some(texture)
    }

    /// The drawable taken this frame, if any.
    pub(crate) fn current(&self) -> Option<Retained<ProtocolObject<dyn MTLDrawable>>> {
        self.drawable.borrow().clone()
    }

    /// Return this frame's drawable to the layer's pool.
    pub(crate) fn release(&self) {
        let _ = self.drawable.borrow_mut().take();
        let _ = self.texture.borrow_mut().take();
    }
}

pub struct MetalSwapchain {
    pub(crate) drawable: SharedDrawableSlot,
    pub(crate) format: Format,
    pub(crate) extent: [u32; 2],
}
