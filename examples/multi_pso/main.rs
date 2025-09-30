//! Multi-PSO example: draw two triangles with two pipelines in one frame.

use kiln::mtl::{DrawableSource as RenderSurface, ResourceOptions, RenderStages, PrimitiveType, PackedFloat3, SceneProperties, VertexInput};
use kiln::{app, mtl as metal};

struct MultiPsoApp {
    pso_a: Option<metal::PipelineState>,
    pso_b: Option<metal::PipelineState>,
    vbuf_a: Option<metal::Vertex<VertexInput>>,
    vbuf_b: Option<metal::Vertex<VertexInput>>,
    scene: Option<metal::Uniform<SceneProperties>>,
    args_a: Option<metal::ArgumentBuffer>,
    args_b: Option<metal::ArgumentBuffer>,
}
impl MultiPsoApp { fn new() -> Self { Self { pso_a: None, pso_b: None, vbuf_a: None, vbuf_b: None, scene: None, args_a: None, args_b: None } } }

impl app::KilnApp for MultiPsoApp {
    fn title(&self) -> &str { "kiln multi-pso" }
    fn init(&mut self, surface: &dyn RenderSurface) {
        let dev = metal::Device::from_surface(surface);
        let msl_a = include_str!("shader.metal");
        let msl_b = include_str!("shader.metal");
        let lib_a = dev.compile_library_from_source("lib_a", msl_a).expect("compile lib a");
        let lib_b = dev.compile_library_from_source("lib_b", msl_b).expect("compile lib b");
        let pso_a = dev.pipeline_builder(&lib_a)
            .vertex("vertex_main")
            .fragment("fragment_main")
            .color_format(surface.color_pixel_format())
            .build()
            .expect("pso a");
        let pso_b = dev.pipeline_builder(&lib_b)
            .vertex("vertex_main")
            .fragment("fragment_main")
            .color_format(surface.color_pixel_format())
            .build()
            .expect("pso b");

        // Two distinct triangle vertex sets
        let verts_a: [VertexInput; 3] = [
            VertexInput { position: PackedFloat3::new(-0.8, -0.2, 0.0), color: PackedFloat3::new(1.0, 0.2, 0.2) },
            VertexInput { position: PackedFloat3::new(-0.2, -0.2, 0.0), color: PackedFloat3::new(1.0, 0.6, 0.2) },
            VertexInput { position: PackedFloat3::new(-0.5,  0.5, 0.0), color: PackedFloat3::new(1.0, 1.0, 0.2) },
        ];
        let verts_b: [VertexInput; 3] = [
            VertexInput { position: PackedFloat3::new( 0.2, -0.2, 0.0), color: PackedFloat3::new(0.2, 0.6, 1.0) },
            VertexInput { position: PackedFloat3::new( 0.8, -0.2, 0.0), color: PackedFloat3::new(0.2, 0.2, 1.0) },
            VertexInput { position: PackedFloat3::new( 0.5,  0.5, 0.0), color: PackedFloat3::new(0.2, 1.0, 1.0) },
        ];
        let vbuf_a = dev.vertex_buffer_from_slice(&verts_a, ResourceOptions::CPUCacheModeDefaultCache);
        let vbuf_b = dev.vertex_buffer_from_slice(&verts_b, ResourceOptions::CPUCacheModeDefaultCache);
        let scene = dev.uniform_buffer_with_len::<SceneProperties>(1, ResourceOptions::CPUCacheModeDefaultCache);
        let args_a = dev.new_argument_buffer(2, 0);
        let args_b = dev.new_argument_buffer(2, 0);

        self.pso_a = Some(pso_a);
        self.pso_b = Some(pso_b);
        self.vbuf_a = Some(vbuf_a);
        self.vbuf_b = Some(vbuf_b);
        self.scene = Some(scene);
        self.args_a = Some(args_a);
        self.args_b = Some(args_b);
    }
    fn update(&mut self, _dt: f32) {}
    fn draw(&mut self, encoder: &kiln::metal::RenderEncoder, t: f32) {
        let (pso_a, pso_b, vbuf_a, vbuf_b, scene, args_a, args_b) = match (&self.pso_a, &self.pso_b, &self.vbuf_a, &self.vbuf_b, &self.scene, &self.args_a, &self.args_b) {
            (Some(a), Some(b), Some(va), Some(vb), Some(sc), Some(aa), Some(ab)) => (a, b, va, vb, sc, aa, ab),
            _ => return,
        };
        let _ = scene.write_one(0, &SceneProperties { time: t });

        // Draw first triangle with PSO A
        args_a.bind2(0, scene, 1, vbuf_a);
        encoder.set_pipeline(pso_a);
        encoder.set_argument_table_at_stages(args_a, RenderStages::Vertex);
        encoder.draw_primitives(PrimitiveType::Triangle, 0, 3);

        // Draw second triangle with PSO B
        args_b.bind2(0, scene, 1, vbuf_b);
        encoder.set_pipeline(pso_b);
        encoder.set_argument_table_at_stages(args_b, RenderStages::Vertex);
        encoder.draw_primitives(PrimitiveType::Triangle, 0, 3);
    }
    fn quit(&mut self) {}
}

fn main() { app::run_app(MultiPsoApp::new()); }
