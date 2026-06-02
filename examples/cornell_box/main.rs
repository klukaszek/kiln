//! Windowed Cornell box, rendered through the mesh-shader path.
//!
//! Loads `examples/assets/cornell-box.usda` with the pure-Rust `openusd` crate, bakes its
//! 16 quad meshes into a world-space triangle soup tagged with each surface's material
//! colour, and draws them with a mesh shader. The scene's authored `Camera` prim drives the
//! view/projection; a simple camera-headlight shades the boxes so depth reads correctly. A
//! harness-owned depth buffer (opted into via [`Example::depth_format`]) resolves occlusion.
//!
//! The first step toward spectrally path tracing this scene — for now it just rasterizes the
//! geometry. Exits cleanly if the device doesn't support mesh shaders.
//!
//! Run with: `cargo run --example cornell_box -- --spp 1024 --samples-per-frame 16`
//! (needs `slangc` on PATH).

#[path = "../common/mod.rs"]
mod common;
mod ray_scene;
mod scene;

use common::Example;
use kiln_rhi::{
    BufferDesc, BumpAllocator, ColorTarget, CommandBuffer, CompareOp, Cull, DepthFlags,
    DepthStencilState, Device, Format, GpuAddress, GpuAllocation, MemoryType, MeshletPso,
    MeshletPsoDesc, SampleCount, ShaderStage, Topology, gpu_struct,
};

const ASSET: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/examples/assets/cornell-box.usda"
);

/// Triangles per meshlet workgroup. 64 × 3 = 192 mesh-output vertices, within the 256 cap.
const TRIS_PER_MESHLET: u32 = 64;

gpu_struct! {
    /// Per-vertex render data. Declared first so the `Vertex*` in `Root` resolves.
    pub struct Vertex {
        pos: [f32; 4] as "float4",    // world position, w = 1
        normal: [f32; 4] as "float4", // world normal,   w = 0
        color: [f32; 4] as "float4",  // linear RGB,      w = 1
    }
}

gpu_struct! {
    /// Pointer-first draw root. `view_proj` is carried as four `float4` rows of a row-vector
    /// matrix so the shader never depends on Slang's matrix storage layout.
    pub struct Root {
        vp0: [f32; 4] as "float4",
        vp1: [f32; 4] as "float4",
        vp2: [f32; 4] as "float4",
        vp3: [f32; 4] as "float4",
        cam_pos: [f32; 4] as "float4",
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

struct CornellBox {
    pso: MeshletPso,
    vbuf: GpuAllocation,
    bump: BumpAllocator,
    scene: scene::Scene,
    _ray_scene: Option<ray_scene::RayScene>,
    tri_count: u32,
    num_meshlets: u32,
}

impl Example for CornellBox {
    fn depth_format() -> Option<Format> {
        Some(Format::D32Float)
    }

    fn new(device: &Device, color_format: Format) -> Self {
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
                    // No culling for the first render: the box is viewed from inside, and
                    // this keeps every wall/light visible regardless of authored winding.
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

        let scene = scene::load(ASSET).unwrap_or_else(|e| {
            eprintln!("failed to load {ASSET}: {e}");
            std::process::exit(1);
        });
        let tri_count = scene.triangle_count();
        let num_meshlets = tri_count.div_ceil(TRIS_PER_MESHLET);
        eprintln!(
            "cornell box: {} vertices, {tri_count} triangles, {num_meshlets} meshlets",
            scene.vertices.len()
        );

        // Persistent vertex buffer (the geometry is static). A plain CPU-visible allocation
        // the mesh shader reads through its `Vertex*` root pointer.
        let vbuf = device
            .malloc(
                (scene.vertices.len() * std::mem::size_of::<Vertex>()) as u64,
                MemoryType::Default,
            )
            .expect("alloc vertex buffer");
        vbuf.upload_slice(&scene.vertices).expect("upload vertices");

        let ray_scene = match ray_scene::RayScene::build(device, color_format, &scene, &vbuf) {
            Ok(ray_scene) => Some(ray_scene),
            Err(e) => {
                eprintln!("cornell ray scene disabled: {e}");
                None
            }
        };

        // Per-frame bump allocator for the transient draw root.
        let bump = BumpAllocator::new(
            device
                .create_buffer(&BufferDesc {
                    size: 64 * 1024,
                    memory: MemoryType::Default,
                    label: Some("cornell-bump".into()),
                })
                .expect("create bump buffer"),
        );

        Self {
            pso,
            vbuf,
            bump,
            scene,
            _ray_scene: ray_scene,
            tri_count,
            num_meshlets,
        }
    }

    fn pre_render(&mut self, device: &Device, cmd: &mut CommandBuffer, extent: [u32; 2]) {
        if let Some(ray_scene) = &mut self._ray_scene {
            ray_scene.pre_render(device, cmd, &self.scene, &self.vbuf, extent);
        }
    }

    fn render(&mut self, cmd: &mut CommandBuffer, extent: [u32; 2]) {
        if let Some(ray_scene) = &mut self._ray_scene {
            ray_scene.render(cmd, extent);
            return;
        }

        self.bump.reset();
        let aspect = extent[0] as f32 / extent[1].max(1) as f32;
        let vp = self.scene.view_proj_rows(aspect);
        let cam = self.scene.camera_pos();

        let root = self
            .bump
            .alloc(std::mem::size_of::<Root>() as u64, 16)
            .expect("bump root");
        root.upload(&Root {
            vp0: vp[0],
            vp1: vp[1],
            vp2: vp[2],
            vp3: vp[3],
            cam_pos: [cam[0], cam[1], cam[2], 1.0],
            verts: self.vbuf.gpu(),
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_config();
    ray_scene::set_target_spp(config.target_spp);
    ray_scene::set_samples_per_frame(config.samples_per_frame);
    common::run::<CornellBox>("Kiln · Cornell box", [0.02, 0.02, 0.03, 1.0])
}

struct Config {
    target_spp: u32,
    samples_per_frame: u32,
}

fn parse_config() -> Config {
    let mut args = std::env::args().skip(1);
    let default_target_spp = ray_scene::default_target_spp();
    let default_samples_per_frame = ray_scene::default_samples_per_frame();
    let mut config = Config {
        target_spp: default_target_spp,
        samples_per_frame: default_samples_per_frame,
    };

    while let Some(arg) = args.next() {
        if arg == "--spp" {
            let Some(value) = args.next() else {
                eprintln!("--spp requires a positive integer value");
                std::process::exit(2);
            };
            config.target_spp = parse_positive_u32("--spp", &value);
            continue;
        }

        if let Some(value) = arg.strip_prefix("--spp=") {
            config.target_spp = parse_positive_u32("--spp", value);
            continue;
        }

        if arg == "--samples-per-frame" || arg == "--spf" {
            let Some(value) = args.next() else {
                eprintln!("{arg} requires a positive integer value");
                std::process::exit(2);
            };
            config.samples_per_frame = parse_positive_u32(&arg, &value);
            continue;
        }

        if let Some(value) = arg.strip_prefix("--samples-per-frame=") {
            config.samples_per_frame = parse_positive_u32("--samples-per-frame", value);
            continue;
        }

        if let Some(value) = arg.strip_prefix("--spf=") {
            config.samples_per_frame = parse_positive_u32("--spf", value);
            continue;
        }

        if arg == "-h" || arg == "--help" {
            print_help(default_target_spp, default_samples_per_frame);
            std::process::exit(0);
        }

        eprintln!("unknown argument: {arg}");
        print_help(default_target_spp, default_samples_per_frame);
        std::process::exit(2);
    }

    config
}

fn parse_positive_u32(flag: &str, value: &str) -> u32 {
    match value.parse::<u32>() {
        Ok(value) if value > 0 => value,
        _ => {
            eprintln!("{flag} must be a positive integer, got {value:?}");
            std::process::exit(2);
        }
    }
}

fn print_help(default_spp: u32, default_samples_per_frame: u32) {
    eprintln!("Usage: cargo run --example cornell_box -- [--spp N] [--samples-per-frame N]");
    eprintln!("  --spp N    progressive render target samples per pixel (default: {default_spp})");
    eprintln!(
        "  --samples-per-frame N, --spf N    path samples accumulated per frame (default: {default_samples_per_frame})"
    );
}
