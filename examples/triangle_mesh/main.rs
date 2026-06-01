//! Windowed mesh-shader triangle.
//!
//! The windowed counterpart to `tests/mesh.rs`'s `mesh_interpolated_triangle`: a mesh
//! shader emits one inset triangle with per-vertex RGB colours, presented to a real
//! window through the RHI's surface + swapchain. Exits cleanly if the device doesn't
//! support mesh shaders.
//!
//! Run with: `cargo run --example triangle_mesh` (needs `slangc` on PATH).

#[path = "../common/mod.rs"]
mod common;

use common::Example;
use kiln_rhi::{
    ColorTarget, CommandBuffer, Cull, Device, Format, MeshletPso, MeshletPsoDesc, SampleCount,
    ShaderStage, Topology,
};

// Same shader as the headless test. Note the digit-free `COLOR` varying semantic:
// Slang lowers a mesh output `COLOR0` and a fragment input `COLOR0` to mismatched
// Metal user-attribute names, so the indexed form fails PSO linking.
const TRI_BODY: &str = /*slang*/
    r#"
struct VOut { float4 pos : SV_Position; float4 color : COLOR; };

[shader("mesh")]
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void msMain(out vertices VOut verts[3], out indices uint3 tris[1])
{
    SetMeshOutputCounts(3, 1);
    VOut a; a.pos = float4( 0.0,  0.8, 0, 1); a.color = float4(1, 0, 0, 1); // top   = red
    VOut b; b.pos = float4( 0.8, -0.8, 0, 1); b.color = float4(0, 1, 0, 1); // right = green
    VOut c; c.pos = float4(-0.8, -0.8, 0, 1); c.color = float4(0, 0, 1, 1); // left  = blue
    verts[0] = a; verts[1] = b; verts[2] = c;
    tris[0] = uint3(0, 1, 2);
}

[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return i.color; }
"#;

struct TriangleMesh {
    pso: MeshletPso,
}

impl Example for TriangleMesh {
    fn new(device: &Device, color_format: Format) -> Self {
        let ms = common::compile(device, TRI_BODY, "msMain", ShaderStage::Mesh);
        let fs = common::compile(device, TRI_BODY, "fsMain", ShaderStage::Pixel);

        let pso = device
            .create_meshlet_pso(
                &MeshletPsoDesc {
                    topology: Topology::TriangleList,
                    color_targets: vec![ColorTarget::new(color_format)],
                    depth_format: None,
                    stencil_format: None,
                    sample_count: SampleCount::S1,
                    alpha_to_coverage: false,
                    cull: Cull::None,
                    support_dual_source_blending: false,
                    blendstate: None,
                    root_constant_size: 16,
                    label: Some("triangle-mesh".into()),
                },
                &ms,
                &fs,
            )
            .unwrap_or_else(|e| {
                eprintln!("mesh shaders unsupported on this device: {e}");
                std::process::exit(0);
            });

        Self { pso }
    }

    fn render(&mut self, cmd: &mut CommandBuffer, _extent: [u32; 2]) {
        cmd.set_meshlet_pipeline(&self.pso);
        cmd.draw_meshlets(None, None, 1, 1, 1);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    common::run::<TriangleMesh>("Kiln · mesh triangle", [0.05, 0.05, 0.08, 1.0])
}
