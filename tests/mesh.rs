//! Headless mesh-shader path test (timed), driven by a backend-agnostic Slang shader.
//!
//! A mesh shader emits a full-screen triangle; the pixel shader colours it from a
//! pointer-first root. Renders to an offscreen texture and verifies via readback.
//! Skips if the device doesn't support mesh shaders.

mod common;

use kiln_rhi::gpu_struct;
use kiln_rhi::{
    BufferDesc, BumpAllocator, ColorAttachment, ColorTarget, Cull, Device, Format, GpuAddress,
    LoadOp, MemoryType, MeshletPso, MeshletPsoDesc, RenderPassDesc, RenderTarget, SampleCount,
    ShaderModule, ShaderStage, StageFlags, StoreOp, TextureDesc, TextureDimension, TextureUsage,
    Topology,
};

gpu_struct! {
    pub struct Root {
        color: [f32; 4] as "float4",
    }
}

const MESH_BODY: &str = /*slang*/
    r#"
struct VOut { float4 pos : SV_Position; };

[shader("mesh")]
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void msMain(out vertices VOut verts[3], out indices uint3 tris[1])
{
    SetMeshOutputCounts(3, 1);
    VOut a; a.pos = float4(-1.0, -1.0, 0.0, 1.0);
    VOut b; b.pos = float4( 3.0, -1.0, 0.0, 1.0);
    VOut c; c.pos = float4(-1.0,  3.0, 0.0, 1.0);
    verts[0] = a; verts[1] = b; verts[2] = c;
    tris[0] = uint3(0, 1, 2);
}

[shader("fragment")]
float4 fsMain(VOut i, uniform Root* r) : SV_Target { return r.color; }
"#;

const SIZE: u32 = 64;

#[test]
fn mesh_fullscreen_color() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };

    let src = format!("{}{}", Root::SLANG, MESH_BODY);
    let Some(ms) = common::compile_shader_or_skip(&device, &src, "msMain", ShaderStage::Mesh)
    else {
        return;
    };
    let Some(fs) = common::compile_shader_or_skip(&device, &src, "fsMain", ShaderStage::Pixel)
    else {
        return;
    };

    let pso = match device.create_meshlet_pso(
        &MeshletPsoDesc {
            topology: Topology::TriangleList,
            color_targets: vec![ColorTarget::new(Format::R8G8B8A8Unorm)],
            depth_format: None,
            stencil_format: None,
            sample_count: SampleCount::S1,
            alpha_to_coverage: false,
            cull: Cull::None,
            support_dual_source_blending: false,
            blendstate: None,
            root_constant_size: 16,
            label: Some("mesh".into()),
        },
        &ms,
        &fs,
    ) {
        Ok(pso) => pso,
        Err(e) => {
            eprintln!("skipping: mesh shaders unsupported on this device ({e})");
            return;
        }
    };

    let tex_desc = TextureDesc {
        width: SIZE,
        height: SIZE,
        depth: 1,
        mip_levels: 1,
        array_layers: 1,
        format: Format::R8G8B8A8Unorm,
        dimension: TextureDimension::D2,
        sample_count: SampleCount::S1,
        usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::TRANSFER_SRC,
        label: Some("rt".into()),
    };
    let sa = device.texture_size_align(&tex_desc).expect("size_align");
    let tex_mem = device
        .malloc_aligned(sa.size, sa.align, MemoryType::GpuOnly)
        .expect("rt mem");
    let texture = device
        .create_texture(&tex_desc, tex_mem.gpu())
        .expect("create_texture");

    // Deliberately a raw `malloc` for the root (the dual-pointer primitive); the other
    // mesh tests source per-draw data from the bump allocator.
    let root = device
        .malloc(std::mem::size_of::<Root>() as u64, MemoryType::Default)
        .expect("root");
    root.upload(&Root {
        color: [0.0, 1.0, 0.0, 1.0],
    })
    .expect("upload root");
    let readback = device
        .malloc((SIZE * SIZE * 4) as u64, MemoryType::Readback)
        .expect("readback");

    common::timed("mesh draw full-screen triangle · submit+wait", || {
        let mut cmd = device.create_command_buffer().expect("cmd");
        cmd.begin_render_pass(&RenderPassDesc {
            color_attachments: vec![ColorAttachment {
                target: RenderTarget::Texture(texture.id()),
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                clear_color: [0.0, 0.0, 0.0, 1.0],
            }],
            depth_attachment: None,
            render_area: [0, 0, SIZE, SIZE],
        });
        cmd.set_meshlet_pipeline(&pso);
        cmd.set_viewport(0.0, 0.0, SIZE as f32, SIZE as f32, 0.0, 1.0);
        cmd.set_scissor(0, 0, SIZE, SIZE);
        cmd.draw_meshlets(root.gpu(), root.gpu(), 1, 1, 1);
        cmd.end_render_pass();

        cmd.barrier(StageFlags::RASTER_COLOR_OUT, StageFlags::TRANSFER);
        cmd.copy_from_texture(readback.gpu(), tex_mem.gpu(), &texture);
        cmd.barrier(StageFlags::TRANSFER, StageFlags::ALL_COMMANDS);
        cmd.end();
        let queue = device.queue();
        queue.submit(cmd).expect("submit");
        queue.wait_idle();
    });

    let pixels = readback.as_slice::<u8>().expect("read readback");
    // Dump the render so it can be eyeballed (written before asserting, so a bad
    // render still leaves an image to inspect).
    common::save_rgba_png("mesh_fullscreen_color", SIZE, SIZE, pixels);
    for px in 0..(SIZE * SIZE) as usize {
        let (r, g, b, a) = (
            pixels[px * 4],
            pixels[px * 4 + 1],
            pixels[px * 4 + 2],
            pixels[px * 4 + 3],
        );
        assert_eq!(
            (r, g, b, a),
            (0, 255, 0, 255),
            "pixel {px} not green: ({r},{g},{b},{a})"
        );
    }

    device.free(root);
    device.free(readback);
}

// ---------------------------------------------------------------------------
// Shared helpers for the mesh-shader tests below.
// ---------------------------------------------------------------------------

/// Build a meshlet PSO with one RGBA8 colour target and no culling, or `None`
/// (skip) if the device doesn't support mesh shaders.
fn make_meshlet_pso(
    device: &Device,
    ms: &ShaderModule,
    fs: &ShaderModule,
    root_constant_size: u32,
    label: &str,
) -> Option<MeshletPso> {
    match device.create_meshlet_pso(
        &MeshletPsoDesc {
            topology: Topology::TriangleList,
            color_targets: vec![ColorTarget::new(Format::R8G8B8A8Unorm)],
            depth_format: None,
            stencil_format: None,
            sample_count: SampleCount::S1,
            alpha_to_coverage: false,
            cull: Cull::None,
            support_dual_source_blending: false,
            blendstate: None,
            root_constant_size,
            label: Some(label.into()),
        },
        ms,
        fs,
    ) {
        Ok(pso) => Some(pso),
        Err(e) => {
            eprintln!("skipping: mesh shaders unsupported on this device ({e})");
            None
        }
    }
}

/// Dispatch `groups` meshlet workgroups of `pso` into a fresh `size`×`size` RGBA8
/// texture (cleared to opaque black) and read the result back to CPU bytes. `root`
/// is bound for both the mesh and pixel stages. The texture is transient; only the
/// returned pixels outlive the call.
fn render_meshlets(
    device: &Device,
    pso: &MeshletPso,
    root: GpuAddress,
    size: u32,
    groups: [u32; 3],
) -> Vec<u8> {
    let tex_desc = TextureDesc {
        width: size,
        height: size,
        depth: 1,
        mip_levels: 1,
        array_layers: 1,
        format: Format::R8G8B8A8Unorm,
        dimension: TextureDimension::D2,
        sample_count: SampleCount::S1,
        usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::TRANSFER_SRC,
        label: Some("rt".into()),
    };
    let sa = device.texture_size_align(&tex_desc).expect("size_align");
    let tex_mem = device
        .malloc_aligned(sa.size, sa.align, MemoryType::GpuOnly)
        .expect("rt mem");
    let texture = device
        .create_texture(&tex_desc, tex_mem.gpu())
        .expect("create_texture");
    let readback = device
        .malloc((size * size * 4) as u64, MemoryType::Readback)
        .expect("readback");

    let mut cmd = device.create_command_buffer().expect("cmd");
    cmd.begin_render_pass(&RenderPassDesc {
        color_attachments: vec![ColorAttachment {
            target: RenderTarget::Texture(texture.id()),
            load_op: LoadOp::Clear,
            store_op: StoreOp::Store,
            clear_color: [0.0, 0.0, 0.0, 1.0],
        }],
        depth_attachment: None,
        render_area: [0, 0, size, size],
    });
    cmd.set_meshlet_pipeline(pso);
    cmd.set_viewport(0.0, 0.0, size as f32, size as f32, 0.0, 1.0);
    cmd.set_scissor(0, 0, size, size);
    cmd.draw_meshlets(root, root, groups[0], groups[1], groups[2]);
    cmd.end_render_pass();

    cmd.barrier(StageFlags::RASTER_COLOR_OUT, StageFlags::TRANSFER);
    cmd.copy_from_texture(readback.gpu(), tex_mem.gpu(), &texture);
    cmd.barrier(StageFlags::TRANSFER, StageFlags::ALL_COMMANDS);
    cmd.end();
    let queue = device.queue();
    queue.submit(cmd).expect("submit");
    queue.wait_idle();

    let pixels = readback.as_slice::<u8>().expect("read readback").to_vec();
    device.free(readback);
    pixels
}

/// A per-test bump allocator over CPU-mapped memory — the doc's preferred source for
/// transient per-draw arguments. Caller releases it with
/// `device.destroy_buffer(bump.into_buffer())` after the draw has completed.
fn test_bump(device: &Device) -> BumpAllocator {
    let buffer = device
        .create_buffer(&BufferDesc {
            size: 64 * 1024,
            memory: MemoryType::Default,
            label: Some("test-bump".into()),
        })
        .expect("create_buffer");
    BumpAllocator::new(buffer)
}

// ---------------------------------------------------------------------------
// Clip-space orientation: Kiln normalizes every backend to Y-up NDC, so a quad
// in the top-left NDC quadrant must land in the top-left of the read-back image.
// This is the regression guard for the Vulkan negative-viewport-height flip; it
// would fail on a Y-down (un-normalized) backend.
// ---------------------------------------------------------------------------

const ORIENT_BODY: &str = /*slang*/
    r#"
struct VOut { float4 pos : SV_Position; };

[shader("mesh")]
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void msMain(out vertices VOut verts[4], out indices uint3 tris[2])
{
    SetMeshOutputCounts(4, 2);
    // Top-left NDC quadrant in a Y-up clip space: x in [-1, 0], y in [0, 1].
    VOut a; a.pos = float4(-1.0, 0.0, 0.0, 1.0);
    VOut b; b.pos = float4( 0.0, 0.0, 0.0, 1.0);
    VOut c; c.pos = float4(-1.0, 1.0, 0.0, 1.0);
    VOut d; d.pos = float4( 0.0, 1.0, 0.0, 1.0);
    verts[0] = a; verts[1] = b; verts[2] = c; verts[3] = d;
    tris[0] = uint3(0, 1, 2);
    tris[1] = uint3(2, 1, 3);
}

[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return float4(1.0, 1.0, 1.0, 1.0); }
"#;

#[test]
fn mesh_clip_space_is_y_up() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };

    let Some(ms) =
        common::compile_shader_or_skip(&device, ORIENT_BODY, "msMain", ShaderStage::Mesh)
    else {
        return;
    };
    let Some(fs) =
        common::compile_shader_or_skip(&device, ORIENT_BODY, "fsMain", ShaderStage::Pixel)
    else {
        return;
    };
    let Some(pso) = make_meshlet_pso(&device, &ms, &fs, 16, "orient") else {
        return;
    };

    const SIZE: u32 = 128;
    // Shader ignores the root, but `draw_meshlets` still needs a valid pointer — hand
    // it a transient bump allocation rather than a dedicated malloc.
    let mut bump = test_bump(&device);
    let root = bump.alloc(16, 16).expect("bump root");
    let pixels = common::timed("mesh clip-space orientation · submit+wait", || {
        render_meshlets(&device, &pso, root.gpu, SIZE, [1, 1, 1])
    });
    common::save_rgba_png("mesh_clip_space_is_y_up", SIZE, SIZE, &pixels);

    // White only in the top-left quadrant (rows < SIZE/2 && cols < SIZE/2); black
    // elsewhere. The quad edges fall on the half-pixel boundaries, so no pixel
    // centre is ambiguous.
    let half = SIZE / 2;
    for y in 0..SIZE {
        for x in 0..SIZE {
            let i = ((y * SIZE + x) * 4) as usize;
            let (r, g, b) = (pixels[i], pixels[i + 1], pixels[i + 2]);
            let expect_white = x < half && y < half;
            if expect_white {
                assert!(
                    r > 250 && g > 250 && b > 250,
                    "top-left should be white at ({x},{y}): ({r},{g},{b})"
                );
            } else {
                assert!(
                    r < 5 && g < 5 && b < 5,
                    "outside top-left should be black at ({x},{y}): ({r},{g},{b})"
                );
            }
        }
    }

    device.destroy_buffer(bump.into_buffer());
}

// ---------------------------------------------------------------------------
// Meshlet grid: a 4×4 dispatch where each meshlet (SV_GroupID.xy) emits a quad
// covering its own NDC cell, coloured from its group id. Verifies 2D meshlet
// dispatch, per-group placement, and that each meshlet rasterizes a clean cell —
// none of which the single full-screen triangle exercises. Bigger target so each
// cell is a comfortable 64×64 px.
// ---------------------------------------------------------------------------

const GRID: u32 = 4;
const GRID_SIZE: u32 = 256;

gpu_struct! {
    pub struct GridCfg {
        dim: u32 as "uint", // grid is dim × dim meshlets
    }
}

const GRID_BODY: &str = /*slang*/
    r#"
// NB: digit-free `COLOR` semantic. Slang lowers a mesh output `COLOR0` to
// `[[user(COLOR0)]]` but a fragment input `COLOR0` to `[[user(COLOR)]]`, so the
// indexed form fails to link across separately-compiled mesh/fragment modules.
struct VOut { float4 pos : SV_Position; float4 color : COLOR; };

[shader("mesh")]
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void msMain(uint3 gid : SV_GroupID,
            out vertices VOut verts[4],
            out indices uint3 tris[2],
            uniform GridCfg* cfg)
{
    SetMeshOutputCounts(4, 2);
    float g = float(cfg.dim);
    float cell = 2.0 / g;
    float x0 = -1.0 + float(gid.x) * cell;
    float y0 = -1.0 + float(gid.y) * cell;
    float x1 = x0 + cell;
    float y1 = y0 + cell;
    // Encode the meshlet's grid coordinates so each cell is uniquely identifiable.
    float4 col = float4((float(gid.x) + 0.5) / g, (float(gid.y) + 0.5) / g, 0.0, 1.0);
    VOut a; a.pos = float4(x0, y0, 0, 1); a.color = col;
    VOut b; b.pos = float4(x1, y0, 0, 1); b.color = col;
    VOut c; c.pos = float4(x0, y1, 0, 1); c.color = col;
    VOut d; d.pos = float4(x1, y1, 0, 1); d.color = col;
    verts[0] = a; verts[1] = b; verts[2] = c; verts[3] = d;
    tris[0] = uint3(0, 1, 2);
    tris[1] = uint3(2, 1, 3);
}

[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return i.color; }
"#;

#[test]
fn mesh_meshlet_grid() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };

    let src = format!("{}{}", GridCfg::SLANG, GRID_BODY);
    let Some(ms) = common::compile_shader_or_skip(&device, &src, "msMain", ShaderStage::Mesh)
    else {
        return;
    };
    let Some(fs) = common::compile_shader_or_skip(&device, &src, "fsMain", ShaderStage::Pixel)
    else {
        return;
    };
    let Some(pso) = make_meshlet_pso(&device, &ms, &fs, 16, "grid") else {
        return;
    };

    // Per-draw config from the bump allocator (the mesh shader reads grid dim).
    let mut bump = test_bump(&device);
    let cfg = bump
        .alloc(std::mem::size_of::<GridCfg>() as u64, 16)
        .expect("bump cfg");
    cfg.upload(&GridCfg { dim: GRID }).expect("upload cfg");

    let pixels = common::timed("meshlet grid (4×4) · submit+wait", || {
        render_meshlets(&device, &pso, cfg.gpu, GRID_SIZE, [GRID, GRID, 1])
    });
    common::save_rgba_png("mesh_meshlet_grid", GRID_SIZE, GRID_SIZE, &pixels);

    // Each meshlet (gx, gy) fills one cell. In the normalized Y-up clip space,
    // gid.y = 0 is the bottom NDC cell → the *bottom* rows of the image, so the
    // pixel-row band for gy is (GRID-1-gy). gid.x = 0 is the left cell as usual.
    let cell = GRID_SIZE / GRID; // 64 px
    let to_u8 = |k: u32| (((k as f32 + 0.5) / GRID as f32) * 255.0).round() as i32;
    let near = |a: u8, b: i32| (a as i32 - b).abs() <= 2;

    for gy in 0..GRID {
        for gx in 0..GRID {
            let px = gx * cell + cell / 2;
            let py = (GRID - 1 - gy) * cell + cell / 2;
            let i = ((py * GRID_SIZE + px) * 4) as usize;
            let (r, g, b) = (pixels[i], pixels[i + 1], pixels[i + 2]);
            let (er, eg) = (to_u8(gx), to_u8(gy));
            assert!(
                near(r, er) && near(g, eg) && near(b, 0),
                "cell ({gx},{gy}) at px ({px},{py}): got ({r},{g},{b}), expected (~{er},~{eg},~0)"
            );
        }
    }

    device.destroy_buffer(bump.into_buffer());
}

// ---------------------------------------------------------------------------
// Interpolated triangle: one inset triangle with per-vertex RGB colours.
// Verifies attribute interpolation (barycentric gradient) and rasterization
// coverage (image corners stay at the clear colour). Orientation-independent:
// every check is symmetric about the image centre, so it holds on any backend.
// ---------------------------------------------------------------------------

const TRI_BODY: &str = /*slang*/
    r#"
// NB: digit-free `COLOR` semantic. Slang lowers a mesh output `COLOR0` to
// `[[user(COLOR0)]]` but a fragment input `COLOR0` to `[[user(COLOR)]]`, so the
// indexed form fails to link across separately-compiled mesh/fragment modules.
struct VOut { float4 pos : SV_Position; float4 color : COLOR; };

[shader("mesh")]
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void msMain(out vertices VOut verts[3], out indices uint3 tris[1])
{
    SetMeshOutputCounts(3, 1);
    // Inset so the image corners are never covered (Y-up clip space).
    VOut a; a.pos = float4( 0.0,  0.8, 0, 1); a.color = float4(1, 0, 0, 1); // top   = red
    VOut b; b.pos = float4( 0.8, -0.8, 0, 1); b.color = float4(0, 1, 0, 1); // right = green
    VOut c; c.pos = float4(-0.8, -0.8, 0, 1); c.color = float4(0, 0, 1, 1); // left  = blue
    verts[0] = a; verts[1] = b; verts[2] = c;
    tris[0] = uint3(0, 1, 2);
}

[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return i.color; }
"#;

#[test]
fn mesh_interpolated_triangle() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };

    let Some(ms) = common::compile_shader_or_skip(&device, TRI_BODY, "msMain", ShaderStage::Mesh)
    else {
        return;
    };
    let Some(fs) = common::compile_shader_or_skip(&device, TRI_BODY, "fsMain", ShaderStage::Pixel)
    else {
        return;
    };
    let Some(pso) = make_meshlet_pso(&device, &ms, &fs, 16, "tri") else {
        return;
    };

    const SIZE: u32 = 128;
    // Per-vertex colours come from the shader; the root is unused but still required,
    // so source it transiently from the bump allocator.
    let mut bump = test_bump(&device);
    let root = bump.alloc(16, 16).expect("bump root");
    let pixels = common::timed("interpolated triangle · submit+wait", || {
        render_meshlets(&device, &pso, root.gpu, SIZE, [1, 1, 1])
    });
    common::save_rgba_png("mesh_interpolated_triangle", SIZE, SIZE, &pixels);

    let at = |x: u32, y: u32| {
        let i = ((y * SIZE + x) * 4) as usize;
        (pixels[i], pixels[i + 1], pixels[i + 2])
    };

    // 1) All four image corners are outside the inset triangle → clear (black).
    for (x, y) in [(0, 0), (SIZE - 1, 0), (0, SIZE - 1), (SIZE - 1, SIZE - 1)] {
        let (r, g, b) = at(x, y);
        assert!(r < 5 && g < 5 && b < 5, "corner ({x},{y}) not clear: ({r},{g},{b})");
    }

    // 2) Image centre (NDC origin) is inside the triangle; by symmetry its colour
    //    is 0.5·red + 0.25·green + 0.25·blue ≈ (128, 64, 64). This pins down the
    //    barycentric interpolation, not just "something was drawn".
    let (cr, cg, cb) = at(SIZE / 2, SIZE / 2);
    let near = |a: u8, b: i32| (a as i32 - b).abs() <= 6;
    assert!(
        near(cr, 128) && near(cg, 64) && near(cb, 64),
        "centre interpolation off: got ({cr},{cg},{cb}), expected ~(128,64,64)"
    );

    // 3) Each vertex colour is reached somewhere (interpolation endpoints): the
    //    most-red / most-green / most-blue pixels are strongly that colour. Found
    //    by scan, so this is independent of where each vertex landed.
    let (mut max_r, mut max_g, mut max_b) = ((0u8, 0u8, 0u8), (0u8, 0u8, 0u8), (0u8, 0u8, 0u8));
    for y in 0..SIZE {
        for x in 0..SIZE {
            let p = at(x, y);
            if p.0 > max_r.0 { max_r = p; }
            if p.1 > max_g.1 { max_g = p; }
            if p.2 > max_b.2 { max_b = p; }
        }
    }
    assert!(
        max_r.0 > 200 && max_r.1 < 80 && max_r.2 < 80,
        "no red-dominant pixel (vertex colour not interpolated): {max_r:?}"
    );
    assert!(
        max_g.1 > 200 && max_g.0 < 80 && max_g.2 < 80,
        "no green-dominant pixel: {max_g:?}"
    );
    assert!(
        max_b.2 > 200 && max_b.0 < 80 && max_b.1 < 80,
        "no blue-dominant pixel: {max_b:?}"
    );

    device.destroy_buffer(bump.into_buffer());
}
