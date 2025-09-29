//! Instanced triangle example using `kiln::metal` facade.

use kiln::metal::MTLResourceOptions;
use kiln::renderer::swapchain::{RenderSurface, SwapchainConfig};
use kiln::renderer::Renderer;
use kiln::{app, metal};

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
        "kiln triangle instanced"
    }
    fn init(&mut self, surface: &dyn RenderSurface) {
        let dev = metal::Device::from_surface(surface);
        let msl = include_str!("../../src/shaders/metal4_triangle.metal");
        let lib = dev
            .compile_library_from_source("example_lib", msl)
            .expect("compile shader lib");
        let pso = dev
            .pipeline_builder(&lib)
            .vertex("vertex_main")
            .fragment("fragment_main")
            .color_format(surface.color_pixel_format())
            .build()
            .expect("pipeline state");

        let verts: [metal::VertexInput; 3] = [
            metal::VertexInput {
                position: metal::PackedFloat3::new(-f32::sqrt(3.0) / 4.0, -0.25, 0.0),
                color: metal::PackedFloat3::new(1.0, 0.0, 0.0),
            },
            metal::VertexInput {
                position: metal::PackedFloat3::new(f32::sqrt(3.0) / 4.0, -0.25, 0.0),
                color: metal::PackedFloat3::new(0.0, 1.0, 0.0),
            },
            metal::VertexInput {
                position: metal::PackedFloat3::new(0.0, 0.5, 0.0),
                color: metal::PackedFloat3::new(0.0, 0.0, 1.0),
            },
        ];
        let vbuf =
            dev.vertex_buffer_from_slice(&verts, MTLResourceOptions::CPUCacheModeDefaultCache);

        let mut r = Renderer::new(surface, SwapchainConfig::default(), pso, vbuf);
        r.set_instance_count(100);
        self.renderer = Some(r);
    }
    fn update(&mut self, _dt: f32) {}
    fn draw(&mut self, surface: &dyn RenderSurface, t: f32) {
        if let Some(r) = self.renderer.as_mut() {
            r.draw_frame(surface, t);
        }
    }
    fn quit(&mut self) {}
}

fn main() {
    app::run_app(TriangleApp::new());
}
