use crate::kiln::swapchain::{SwapchainConfig, ColorSpace, PresentMode};

#[cfg(target_os = "macos")]
use objc2_metal_kit::MTKView;
#[cfg(target_os = "macos")]
use objc2_core_graphics::{CGColorSpace, kCGColorSpaceDisplayP3, kCGColorSpaceSRGB, kCGColorSpaceExtendedSRGB};

#[cfg(target_os = "macos")]
pub unsafe fn apply_swapchain_to_mtkview(view: &MTKView, sc: &SwapchainConfig) {
    unsafe { view.setColorPixelFormat(sc.pixel_format); }
    unsafe { view.setFramebufferOnly(sc.framebuffer_only); }
    if matches!(sc.present_mode, PresentMode::Immediate) {
        unsafe { view.setPreferredFramesPerSecond(120); }
    }
    let name = unsafe { match sc.colorspace { ColorSpace::SRGB => kCGColorSpaceSRGB, ColorSpace::DisplayP3 => kCGColorSpaceDisplayP3, ColorSpace::ExtendedSRGB => kCGColorSpaceExtendedSRGB } };
    if let Some(cs) = unsafe { CGColorSpace::with_name(Some(name)) } { unsafe { view.setColorspace(Some(&*cs)); } }
}

#[cfg(target_os = "macos")]
use objc2_core_foundation::CGSize;
#[cfg(target_os = "macos")]
use objc2_quartz_core::CAMetalLayer;

#[cfg(target_os = "macos")]
pub unsafe fn apply_swapchain_to_metal_layer(layer: &CAMetalLayer, drawable_width: f64, drawable_height: f64, sc: &SwapchainConfig) {
    unsafe { layer.setPixelFormat(sc.pixel_format); }
    unsafe { layer.setFramebufferOnly(sc.framebuffer_only); }
    unsafe { layer.setMaximumDrawableCount(sc.max_drawables as usize); }
    unsafe { layer.setDisplaySyncEnabled(matches!(sc.present_mode, PresentMode::Fifo)); }
    unsafe { layer.setAllowsNextDrawableTimeout(false); }
    let name = unsafe { match sc.colorspace { ColorSpace::SRGB => kCGColorSpaceSRGB, ColorSpace::DisplayP3 => kCGColorSpaceDisplayP3, ColorSpace::ExtendedSRGB => kCGColorSpaceExtendedSRGB } };
    if let Some(cs) = unsafe { CGColorSpace::with_name(Some(name)) } { unsafe { layer.setColorspace(Some(&*cs)); } }
    unsafe { layer.setWantsExtendedDynamicRangeContent(sc.wants_edr); }
    unsafe { layer.setDrawableSize(CGSize { width: drawable_width, height: drawable_height }); }
}

#[cfg(feature = "winit")]
use winit::event_loop::ActiveEventLoop;
#[cfg(feature = "winit")]
use core::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "winit")]
static EXIT_REQUESTED: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "winit")]
pub fn request_app_exit(event_loop: &ActiveEventLoop) { if !EXIT_REQUESTED.swap(true, Ordering::SeqCst) { event_loop.exit(); } }

