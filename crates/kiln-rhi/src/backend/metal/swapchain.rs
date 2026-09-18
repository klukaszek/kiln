use std::cell::RefCell;
use std::rc::Rc;

use super::device::MetalDevice;
use super::surface::MetalSurface;
use super::texture::mtl_to_format;
use crate::error::RhiResult;
use crate::swapchain::{Swapchain, SwapchainDesc};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTL4CommandQueue;
use objc2_metal::{MTLDrawable, MTLTexture};
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};

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

/// Apply a [`SwapchainDesc`] to the layer a swapchain presents through.
///
/// Create and recreate set exactly the same four properties, so they share this rather than
/// keeping two copies that have to be edited together.
pub(crate) fn configure_layer(layer: &CAMetalLayer, desc: &crate::swapchain::SwapchainDesc) {
    use objc2_core_foundation::CGSize;

    layer.setPixelFormat(super::texture::format_to_mtl(desc.format));
    layer.setDisplaySyncEnabled(desc.vsync);
    // Metal allows 2 or 3 drawables; anything else is clamped rather than refused.
    layer.setMaximumDrawableCount(desc.image_count.clamp(2, 3) as usize);
    layer.setDrawableSize(CGSize {
        width: desc.width as f64,
        height: desc.height as f64,
    });
}

pub struct MetalSwapchain {
    pub(crate) drawable: SharedDrawableSlot,
}

impl MetalDevice {
    pub fn create_swapchain(
        &self,
        surface: &MetalSurface,
        desc: &SwapchainDesc,
    ) -> RhiResult<Swapchain> {
        let layer = &surface.layer;

        // Drawables live in the layer's own residency set; adding it to the queue is what makes
        // them resident. The layer outlives every swapchain built from it, so this happens once.
        self.queue.queue.addResidencySet(&layer.residencySet());

        super::swapchain::configure_layer(layer, desc);
        let format = mtl_to_format(layer.pixelFormat());

        Ok(Swapchain::new(
            Box::new(MetalSwapchain {
                drawable: Rc::new(MetalDrawableSlot::new(layer.clone())),
            }),
            format,
            [desc.width, desc.height],
        ))
    }

    pub fn recreate_swapchain(
        &self,
        swapchain: &mut Swapchain,
        desc: &SwapchainDesc,
    ) -> RhiResult<()> {
        self.wait_idle();
        swapchain.inner.drawable.release();
        let layer = &swapchain.inner.drawable.layer;
        super::swapchain::configure_layer(layer, desc);
        swapchain.format = mtl_to_format(layer.pixelFormat());
        swapchain.extent = [desc.width, desc.height];
        Ok(())
    }
}
