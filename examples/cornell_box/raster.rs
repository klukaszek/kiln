//! Mesh-shader raster preview: draws the triangle soup with a camera headlight.
//!
//! This is the fallback view when the path tracer can't be built (no ray-query
//! support), and a quick sanity check that the loaded geometry and camera are sound.

use glam::Vec4;
use kiln_rhi::{
    BufferDesc, BumpAllocator, ColorTarget, CommandBuffer, CompareOp, Cull, DepthFlags,
    DepthStencilState, Device, Format, GpuAddress, MAX_FRAMES_IN_FLIGHT, MemoryType, MeshletPso,
    MeshletPsoDesc, SampleCount, ShaderStage, Topology, gpu_struct,
};

use crate::common::{self, FrameCtx};
use crate::scene::gpu::GpuScene;
use crate::scene::{Scene, Vertex};

/// Triangles per meshlet workgroup. 64 × 3 = 192 mesh-output vertices, within the 256 cap.
const TRIS_PER_MESHLET: u32 = 64;

gpu_struct! {
    /// Pointer-first draw root. `view_proj` is carried as four `float4` rows of a row-vector
    /// matrix so the shader never depends on Slang's matrix storage layout.
    pub struct Root {
        vp0: Vec4 as "float4",
        vp1: Vec4 as "float4",
        vp2: Vec4 as "float4",
        vp3: Vec4 as "float4",
        cam_pos: Vec4 as "float4",
        verts: GpuAddress as "Vertex*",
        tri_count: u32 as "uint",
        _pad: u32 as "uint",
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
    /// One transient-argument arena per frame in flight, reset when its slot records.
    frame_arenas: [BumpAllocator; MAX_FRAMES_IN_FLIGHT],
    tri_count: u32,
    num_meshlets: u32,
}

impl RasterPreview {
    /// Build the preview pipeline. Exits cleanly if the device lacks mesh shaders.
    pub fn build(device: &Device, color_format: Format, scene: &Scene) -> Self {
        let src = format!("{}{}{}", Vertex::SLANG, Root::SLANG, BODY);
        let ms = common::compile(device, &src, "msMain", ShaderStage::Mesh);
        let fs = common::compile(device, &src, "fsMain", ShaderStage::Pixel);

        let pso = device
            .create_meshlet_pso(
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
                    label: Some("cornell-box".into()),
                },
                &ms,
                &fs,
            )
            .unwrap_or_else(|e| {
                eprintln!("mesh shaders unsupported on this device: {e}");
                std::process::exit(0);
            });

        let tri_count = scene.triangle_count();
        let num_meshlets = tri_count.div_ceil(TRIS_PER_MESHLET);
        eprintln!(
            "cornell box: {} vertices, {tri_count} triangles, {num_meshlets} meshlets",
            scene.vertices.len()
        );

        // Per-slot bump arenas for the transient draw root, so a recording frame
        // never overwrites the root an in-flight frame still reads.
        let frame_arenas = std::array::from_fn(|slot| {
            BumpAllocator::new(
                device
                    .create_buffer(&BufferDesc {
                        size: 4096,
                        memory: MemoryType::Default,
                        label: Some(format!("cornell-raster-arena-{slot}")),
                    })
                    .expect("create raster arena"),
            )
        });

        Self {
            pso,
            frame_arenas,
            tri_count,
            num_meshlets,
        }
    }

    pub fn render(
        &mut self,
        ctx: &FrameCtx,
        cmd: &mut CommandBuffer,
        scene: &Scene,
        gpu_scene: &GpuScene,
    ) {
        let arena = &mut self.frame_arenas[ctx.slot];
        arena.reset();
        let aspect = ctx.extent.x as f32 / ctx.extent.y.max(1) as f32;
        let vp = scene.view_proj_rows(aspect);
        let cam = scene.camera_pos();

        let root = arena
            .alloc(std::mem::size_of::<Root>() as u64, 16)
            .expect("raster root from frame arena");
        root.upload(&Root {
            vp0: vp[0],
            vp1: vp[1],
            vp2: vp[2],
            vp3: vp[3],
            cam_pos: cam.extend(1.0),
            verts: gpu_scene.vertex_buffer.gpu(),
            tri_count: self.tri_count,
            _pad: 0,
        })
        .expect("upload root");

        cmd.set_meshlet_pipeline(&self.pso);
        cmd.set_depth_stencil_state(&DepthStencilState {
            depth_mode: DepthFlags::READ | DepthFlags::WRITE,
            depth_test: CompareOp::Less,
            stencil_read_mask: 0,
            stencil_write_mask: 0,
            ..Default::default()
        });
        cmd.draw_meshlets(root.gpu, root.gpu, self.num_meshlets, 1, 1);
    }
}
