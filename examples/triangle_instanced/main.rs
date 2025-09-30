//! Instanced triangle example using zero-copy `kiln::metal` facade.

use kiln::metal::{MTLPrimitiveType, MTLRenderStages, MTLResourceOptions};
use kiln::metal::MTLDrawableSource as RenderSurface;
use kiln::{app, metal};

struct TriangleApp {
    pipeline: Option<metal::PipelineState>,
    vbuf: Option<metal::Vertex<metal::VertexInput>>,
    scene: Option<metal::Uniform<metal::SceneProperties>>,
    args: Option<metal::ArgumentBuffer>,
    instances: usize,
}
impl TriangleApp {
    fn new() -> Self {
        Self {
            pipeline: None,
            vbuf: None,
            scene: None,
            args: None,
            instances: 100,
        }
    }
}

impl app::KilnApp for TriangleApp {
    fn title(&self) -> &str {
        "kiln triangle instanced"
    }
    fn init(&mut self, surface: &dyn RenderSurface) {
        let dev = metal::Device::from_surface(surface);

        let msl = include_str!("shader.metal");
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
        let scene = dev.uniform_buffer_with_len::<metal::SceneProperties>(
            1,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );
        let args = dev.new_argument_buffer(2, 0);

        self.pipeline = Some(pso);
        self.vbuf = Some(vbuf);
        self.scene = Some(scene);
        self.args = Some(args);
    }
    fn update(&mut self, _dt: f32) {}
    fn draw(&mut self, encoder: &kiln::metal::RenderEncoder, t: f32) {
        let (pso, vbuf, scene, args) = match (&self.pipeline, &self.vbuf, &self.scene, &self.args) {
            (Some(p), Some(vb), Some(sc), Some(at)) => (p, vb, sc, at),
            _ => return,
        };

        let _ = scene.write_one(0, &metal::SceneProperties { time: t });
        args.bind2(0, scene, 1, vbuf);

        encoder.set_pipeline(pso);
        encoder.set_argument_table_at_stages(args, MTLRenderStages::Vertex);
        encoder.draw_primitives_instanced(MTLPrimitiveType::Triangle, 0, 3, self.instances);
    }
    fn quit(&mut self) {}
}

fn main() {
    app::run_app(TriangleApp::new());
}
