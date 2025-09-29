//! Kiln example entry-point — unified. Backend is selected by crate features.
#![deny(unsafe_op_in_unsafe_fn)]

use kiln::kiln::renderer::Renderer;
use kiln::kiln::swapchain::{RenderSurface, SwapchainConfig};
use kiln::kiln::{app, shader};

// A minimal ExampleApp demonstrating:
// - Loading a Metal shader via kiln::shader::from_source
// - Building a pipeline via kiln::shader::pipeline_state
// - Drawing a triangle via kiln::renderer using the shared draw loop
struct TriangleApp {
    renderer: Option<Renderer>,
}
impl TriangleApp {
    fn new() -> Self {
        Self { renderer: None }
    }
}

impl app::KilnApp for TriangleApp {
    fn title(&self) -> &str {
        "kiln triangle"
    }
    fn init(&mut self, surface: &dyn RenderSurface) {
        // Demonstrate shader compilation helpers
        let device = surface.device();
        let msl = include_str!("../src/shaders/metal4_triangle.metal");
        let lib = shader::from_source(&device, "example_lib", msl).expect("compile shader lib");
        let _pso = shader::pipeline_state(
            &device,
            &lib,
            "vertex_main",
            "fragment_main",
            surface.color_pixel_format(),
        )
        .expect("pipeline state");

        // Use the shared renderer to draw
        self.renderer = Some(Renderer::new(surface, SwapchainConfig::default()));
    }
    fn update(&mut self, dt: f32) {
        // println!("Delta Time: {}", dt);
    }
    fn draw(&mut self, surface: &dyn RenderSurface, t: f32) {
        if let Some(r) = self.renderer.as_ref() {
            r.draw_frame(surface, t);
        }
    }
    fn quit(&mut self) {}
}

fn main() {
    app::run_app(TriangleApp::new());
}
