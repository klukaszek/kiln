//! Kiln example entry-point — unified. Backend is selected by crate features.

use kiln::renderer::swapchain::{RenderSurface, SwapchainConfig};
use kiln::renderer::Renderer;
use kiln::{app, gfx};
use objc2_metal::MTLResourceOptions;

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
        // Load shader and build pipeline state via kiln::gfx::shader
        let device = surface.device();
        let msl = include_str!("../src/shaders/metal4_triangle.metal");
        let lib =
            gfx::shader::from_source(&device, "example_lib", msl).expect("compile shader lib");
        let pso = gfx::shader::pipeline_state(
            &device,
            &lib,
            "vertex_main",
            "fragment_main",
            surface.color_pixel_format(),
        )
        .expect("pipeline state");

        // Create vertex buffer for a triangle via kiln::gfx helpers
        let verts: [gfx::VertexInput; 3] = [
            gfx::VertexInput {
                position: gfx::PackedFloat3::new(-f32::sqrt(3.0) / 4.0, -0.25, 0.0),
                color: gfx::PackedFloat3::new(1.0, 0.0, 0.0),
            },
            gfx::VertexInput {
                position: gfx::PackedFloat3::new(f32::sqrt(3.0) / 4.0, -0.25, 0.0),
                color: gfx::PackedFloat3::new(0.0, 1.0, 0.0),
            },
            gfx::VertexInput {
                position: gfx::PackedFloat3::new(0.0, 0.5, 0.0),
                color: gfx::PackedFloat3::new(0.0, 0.0, 1.0),
            },
        ];
        let vbuf = gfx::new_buffer_with_bytes(
            &device,
            &verts,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );

        // Construct renderer with provided PSO + vertex buffer
        self.renderer = Some(Renderer::new(
            surface,
            SwapchainConfig::default(),
            pso,
            vbuf,
        ));
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
