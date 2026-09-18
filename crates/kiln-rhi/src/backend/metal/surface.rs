//! Metal surface: the `CAMetalLayer` a swapchain presents through.

use objc2::rc::Retained;
use objc2_metal::MTLPixelFormat;
use objc2_quartz_core::CAMetalLayer;
use raw_window_handle::RawWindowHandle;

use super::device::MetalDevice;
use crate::error::{RhiError, RhiResult};
use crate::surface::{Surface, SurfaceDesc};

/// The `CAMetalLayer` a swapchain is built from. Lives beside `MetalDrawableSlot`, which holds
/// the same layer: a Metal surface is nothing but that layer.
pub struct MetalSurface {
    pub(crate) layer: Retained<CAMetalLayer>,
}

impl MetalDevice {
    pub fn create_surface(&self, desc: &SurfaceDesc) -> RhiResult<Surface> {
        let layer = match desc.window_handle {
            RawWindowHandle::AppKit(handle) => unsafe {
                use objc2::msg_send;
                use objc2::runtime::{AnyObject, Bool};

                let ns_view = handle.ns_view.as_ptr().cast::<AnyObject>();

                let layer = CAMetalLayer::new();
                layer.setDevice(Some(&self.shared.device));
                layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm_sRGB);
                layer.setFramebufferOnly(true);
                layer.setOpaque(true);

                let _: () = msg_send![ns_view, setWantsLayer: Bool::YES];
                let layer_ptr = objc2::rc::Retained::as_ptr(&layer)
                    .cast::<AnyObject>()
                    .cast_mut();
                let _: () = msg_send![ns_view, setLayer: layer_ptr];

                layer
            },
            _ => {
                return Err(RhiError::SurfaceCreation(
                    "Only AppKit windows are supported for Metal".into(),
                ));
            }
        };

        Ok(Surface {
            inner: MetalSurface { layer },
            _owner: None,
        })
    }
}
