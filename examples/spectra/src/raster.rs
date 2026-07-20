//! Mesh-shader raster preview: draws the triangle soup with a camera headlight.
//!
//! This is the fallback view when the path tracer can't be built (no ray-query
//! support), and a quick sanity check that the loaded geometry and camera are sound.

use glam::Vec4;
use kiln_rhi::{
    ColorTarget, CommandBuffer, CompareOp, Cull, DepthFlags, DepthStencilState, Device, Format,
    GpuAddress, MeshletPso, MeshletPsoDesc, SampleCount, ShaderStage, Topology, gpu_struct,
};

use kiln_app::FrameCtx;

use crate::frame_arena::FrameArenas;
use crate::scene::gpu::GpuGeometry;
use crate::scene::{Scene, Vertex};

/// Triangles per meshlet workgroup. 64 × 3 = 192 mesh-output vertices, within the 256 cap.
const TRIS_PER_MESHLET: u32 = 64;

gpu_struct! {
    /// Pointer-first draw root. `view_proj` is carried as four `float4` rows of a row-vector
    /// matrix so the shader never depends on Slang's matrix storage layout.
    pub struct Root {
        vp0: Vec4,
        vp1: Vec4,
        vp2: Vec4,
        vp3: Vec4,
        cam_pos: Vec4,
        verts: GpuAddress as "Vertex*",
        tri_count: u32,
        _pad: u32,
    }
}

// One Slang source for both stages. `Vertex` then `Root` declarations are prepended so the
// `gpu_struct!` host layouts and the shader stay in lockstep. Digit-free varying semantics
// (`COLOR`, `NORMAL`, `WORLD`) — Slang lowers indexed forms to mismatched Metal attributes.
const BODY: &str = /*slang*/
    r#"
struct VOut {
    float4 pos    : SV_Position;
    float3 world  : WORLD;
    float3 nrm    : NORMAL;
    float3 color  : COLOR;
};

[shader("mesh")]
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void msMain(uint3 gid : SV_GroupID,
            out vertices VOut verts[192],
            out indices uint3 tris[64],
            uniform Root* r)
{
    uint base = gid.x * 64u;
    uint remaining = r.tri_count - base;
    uint count = remaining < 64u ? remaining : 64u;
    SetMeshOutputCounts(count * 3u, count);

    for (uint t = 0u; t < count; t++) {
        for (uint k = 0u; k < 3u; k++) {
            uint vi = (base + t) * 3u + k;
            Vertex v = r.verts[vi];
            float4 p = v.pos;
            VOut o;
            // Row-vector transform: clip = p · view_proj, view_proj split into rows.
            o.pos   = p.x * r.vp0 + p.y * r.vp1 + p.z * r.vp2 + p.w * r.vp3;
            o.world = v.pos.xyz;
            o.nrm   = v.normal.xyz;
            o.color = v.color.xyz;
            verts[t * 3u + k] = o;
        }
        tris[t] = uint3(t * 3u, t * 3u + 1u, t * 3u + 2u);
    }
}

[shader("fragment")]
float4 fsMain(VOut i, uniform Root* r) : SV_Target
{
    float3 N = normalize(i.nrm);
    float3 L = normalize(r.cam_pos.xyz - i.world);   // camera headlight
    float lambert = abs(dot(N, L));                  // two-sided so no wall goes black
    float3 lit = i.color * (0.2 + 0.8 * lambert);
    return float4(lit, 1.0);
}
"#;

pub struct RasterPreview {
    pso: MeshletPso,
    frame_arenas: FrameArenas,
    tri_count: u32,
    num_meshlets: u32,
}

impl RasterPreview {
    pub fn build(
        device: &Device,
        color_format: Format,
        geometry: &GpuGeometry,
    ) -> anyhow::Result<Self> {
        let src = format!("{}{}{}", Vertex::SLANG, Root::SLANG, BODY);
        let ms = kiln_rhi::compiler::compile(device, &src, "msMain", ShaderStage::Mesh);
        let fs = kiln_rhi::compiler::compile(device, &src, "fsMain", ShaderStage::Pixel);

        let pso = device.create_meshlet_pso(
            &MeshletPsoDesc {
                topology: Topology::TriangleList,
                color_targets: vec![ColorTarget::new(color_format)],
                depth_format: Some(Format::D32Float),
                stencil_format: None,
                sample_count: SampleCount::S1,
                alpha_to_coverage: false,
                // No culling: the box is viewed from inside, and this keeps every
                // wall/light visible regardless of authored winding.
                cull: Cull::None,
                support_dual_source_blending: false,
                blendstate: None,
                root_constant_size: 16,
                label: Some("spectral-raster".into()),
            },
            &ms,
            &fs,
        )?;

        let tri_count = geometry.triangle_count;
        let num_meshlets = tri_count.div_ceil(TRIS_PER_MESHLET);
        eprintln!(
            "raster preview: {} vertices, {tri_count} triangles, {num_meshlets} meshlets",
            tri_count * 3
        );

        let frame_arenas = FrameArenas::new(device, 4096, "spectral-raster-arena")?;

        Ok(Self {
            pso,
            frame_arenas,
            tri_count,
            num_meshlets,
        })
    }

    pub fn render(
        &mut self,
        ctx: &FrameCtx,
        cmd: &mut CommandBuffer,
        scene: &Scene,
        geometry: &GpuGeometry,
    ) {
        self.frame_arenas.reset(ctx.slot);
        let aspect = ctx.extent.x as f32 / ctx.extent.y.max(1) as f32;
        let vp = scene.view_proj_rows(aspect);
        let cam = scene.camera_pos();

        let root = self.frame_arenas.upload(
            ctx.slot,
            &Root {
                vp0: vp[0],
                vp1: vp[1],
                vp2: vp[2],
                vp3: vp[3],
                cam_pos: cam.extend(1.0),
                verts: geometry.vertex_buffer.gpu(),
                tri_count: self.tri_count,
                _pad: 0,
            },
        );

        cmd.set_meshlet_pipeline(&self.pso);
        cmd.set_depth_stencil_state(&DepthStencilState {
            depth_mode: DepthFlags::READ | DepthFlags::WRITE,
            depth_test: CompareOp::Less,
            stencil_read_mask: 0,
            stencil_write_mask: 0,
            ..Default::default()
        });
        cmd.draw_meshlets(root, root, self.num_meshlets, 1, 1);
    }

    pub fn destroy(self, device: &Device) {
        self.frame_arenas.destroy(device);
    }
}
