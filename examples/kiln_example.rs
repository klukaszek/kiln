//! Kiln example entry-point — unified. Backend is selected by crate features.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use kiln::kiln;

fn main() {
    let cfg = kiln::app::RunConfig::new("kiln triangle");
    let mut renderer: Option<kiln::renderer::Renderer> = None;
    kiln::app::run(
        cfg,
        Box::new(move |surface: &dyn kiln::swapchain::RenderSurface, t| {
            if renderer.is_none() {
                renderer = Some(kiln::renderer::Renderer::new(
                    surface,
                    kiln::swapchain::SwapchainConfig::default(),
                ));
            }
            if let Some(r) = renderer.as_ref() {
                r.draw_frame(surface, t);
            }
        }),
    );
}
