//! Windowed graphics-pipeline triangle.
//!
//! The windowed counterpart to `tests/graphics.rs`'s `graphics_interpolated_triangle`:
//! the same backend-agnostic Slang shader (an inset triangle with per-vertex RGB
//! colours interpolated across the face), but presented to a real window through the
//! RHI's surface + swapchain instead of read back from an offscreen texture.
//!
//! Run with: `cargo run -p triangle-graphics` (needs `slangc` on PATH).

use kiln_app::{Example, FrameCtx};
use kiln_rhi::{
    ColorTarget, CommandBuffer, Cull, Device, Format, GpuPtr, GraphicsPso, GraphicsPsoDesc,
    SampleCount, ShaderStage, Topology,
};

// Same shader as the headless test: positions and per-vertex colours are static in
// the vertex shader; the fragment shader just passes the interpolated colour through.
const TRI_BODY: &str = /* slang */
    r#"
struct VOut { float4 pos : SV_Position; float4 color : COLOR; };

static const float2 POS[3] = { float2(0.0, 0.8), float2(0.8, -0.8), float2(-0.8, -0.8) };
static const float4 COL[3] = { float4(1,0,0,1), float4(0,1,0,1), float4(0,0,1,1) };

[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID)
{
    VOut o; o.pos = float4(POS[vid], 0.0, 1.0); o.color = COL[vid]; return o;
}

[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return i.color; }
"#;

struct TriangleGraphics {
    pso: GraphicsPso,
}

impl Example for TriangleGraphics {
    fn new(device: &Device, color_format: Format) -> Self {
        let vs = kiln_rhi::compiler::compile(device, TRI_BODY, "vsMain", ShaderStage::Vertex, &[])
            .expect("shader compilation");
        let fs = kiln_rhi::compiler::compile(device, TRI_BODY, "fsMain", ShaderStage::Pixel, &[])
            .expect("shader compilation");

        let pso = device
            .create_graphics_pso(
                &GraphicsPsoDesc {
                    topology: Topology::TriangleList,
                    // Must match the swapchain's colour format, not a fixed RGBA8.
                    color_targets: vec![ColorTarget::new(color_format)],
                    depth_format: None,
                    sample_count: SampleCount::S1,
                    cull: Cull::None,
                    label: Some("triangle-graphics".into()),
                    ..Default::default()
                },
                &vs,
                &fs,
            )
            .expect("create_graphics_pso");

        Self { pso }
    }

    fn render(&mut self, _ctx: &FrameCtx, cmd: &mut CommandBuffer) {
        cmd.set_pipeline(&self.pso);
        cmd.draw(GpuPtr::<u8>::NULL, 3, 1, 0, 0);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Dark grey clear so the inset triangle's black corners are distinguishable.
    kiln_app::run::<TriangleGraphics>("Kiln · graphics triangle", [0.05, 0.05, 0.08, 1.0])
}
