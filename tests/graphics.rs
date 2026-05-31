//! Headless graphics path test (timed), driven by a backend-agnostic Slang shader.
//!
//! Renders a full-screen triangle into an offscreen RGBA8 texture, colouring every pixel
//! from a pointer-first root struct, then reads the texture back and verifies it. Exercises
//! the render-pass path (begin/end), graphics PSO creation, and the two-stage root binding.

mod common;

use kiln_rhi::gpu_struct;
use kiln_rhi::{
    BufferDesc, BumpAllocator, ColorAttachment, ColorTarget, Cull, Device, Format, GpuAddress,
    GraphicsPso, GraphicsPsoDesc, LoadOp, MemoryType, RenderPassDesc, RenderTarget, SampleCount,
    ShaderModule, ShaderStage, StageFlags, StoreOp, TextureDesc, TextureDimension, TextureUsage,
    Topology,
};

// Shared host/device root: a single colour, used by the pixel shader.
gpu_struct! {
    pub struct Root {
        color: [f32; 4] as "float4",
    }
}

// Full-screen triangle in the vertex shader (covers all of NDC regardless of clip-space Y),
// constant colour from the root in the pixel shader.
const GFX_BODY: &str = /*slang*/
    r#"
struct VOut { float4 pos : SV_Position; };

[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID)
{
    float2 p = float2(float((vid << 1) & 2), float(vid & 2));
    VOut o;
    o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

[shader("fragment")]
float4 fsMain(VOut i, uniform Root* r) : SV_Target
{
    return r.color;
}
"#;

const SIZE: u32 = 64;

#[test]
fn graphics_fullscreen_color() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };

    let src = format!("{}{}", Root::SLANG, GFX_BODY);
    let Some(vs) = common::compile_shader_or_skip(&device, &src, "vsMain", ShaderStage::Vertex)
    else {
        return;
    };
    let Some(fs) = common::compile_shader_or_skip(&device, &src, "fsMain", ShaderStage::Pixel)
    else {
        return;
    };

    let pso = common::timed("create_graphics_pso", || {
        device
            .create_graphics_pso(
                &GraphicsPsoDesc {
                    topology: Topology::TriangleList,
                    color_targets: vec![ColorTarget::new(Format::R8G8B8A8Unorm)],
                    depth_format: None,
                    sample_count: SampleCount::S1,
                    root_constant_size: 16,
                    cull: Cull::None,
                    ..Default::default()
                },
                &vs,
                &fs,
            )
            .expect("create_graphics_pso")
    });

    // Offscreen color target.
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

    // Root color + readback buffer. This test deliberately uses a raw `malloc` for the
    // root (the dual-pointer primitive); the other render tests source per-draw data
    // from the bump allocator. See `graphics_root_from_bump_allocator`.
    let root = device
        .malloc(std::mem::size_of::<Root>() as u64, MemoryType::Default)
        .expect("root");
    root.upload(&Root {
        color: [1.0, 0.0, 0.0, 1.0],
    })
    .expect("upload root");
    let readback = device
        .malloc((SIZE * SIZE * 4) as u64, MemoryType::Readback)
        .expect("readback");

    common::timed("render full-screen triangle · submit+wait", || {
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
        cmd.set_graphics_pipeline(&pso);
        cmd.set_viewport(0.0, 0.0, SIZE as f32, SIZE as f32, 0.0, 1.0);
        cmd.set_scissor(0, 0, SIZE, SIZE);
        // Vertex shader ignores the root; pixel shader reads the color. Same pointer for both.
        cmd.draw(root.gpu(), root.gpu(), 3, 1, 0, 0);
        cmd.end_render_pass();

        cmd.barrier(StageFlags::RASTER_COLOR_OUT, StageFlags::TRANSFER);
        cmd.copy_from_texture(readback.gpu(), tex_mem.gpu(), &texture);
        cmd.barrier(StageFlags::TRANSFER, StageFlags::ALL_COMMANDS);
        cmd.end();
        let queue = device.queue();
        queue.submit(cmd).expect("submit");
        queue.wait_idle();
    });

    // Every pixel should be opaque red.
    let pixels = readback.as_slice::<u8>().expect("read readback");
    // Dump the render so it can be eyeballed (written before asserting, so a bad
    // render still leaves an image to inspect).
    common::save_rgba_png("graphics_fullscreen_color", SIZE, SIZE, pixels);
    for px in 0..(SIZE * SIZE) as usize {
        let (r, g, b, a) = (
            pixels[px * 4],
            pixels[px * 4 + 1],
            pixels[px * 4 + 2],
            pixels[px * 4 + 3],
        );
        assert_eq!(
            (r, g, b, a),
            (255, 0, 0, 255),
            "pixel {px} not red: ({r},{g},{b},{a})"
        );
    }

    device.free(root);
    device.free(readback);
}

// ---------------------------------------------------------------------------
// Shared helpers for the graphics-pipeline tests below.
// ---------------------------------------------------------------------------

/// Build a graphics PSO with one RGBA8 colour target, no depth, no culling.
fn make_graphics_pso(
    device: &Device,
    vs: &ShaderModule,
    fs: &ShaderModule,
    root_constant_size: u32,
    label: &str,
) -> GraphicsPso {
    device
        .create_graphics_pso(
            &GraphicsPsoDesc {
                topology: Topology::TriangleList,
                color_targets: vec![ColorTarget::new(Format::R8G8B8A8Unorm)],
                depth_format: None,
                sample_count: SampleCount::S1,
                root_constant_size,
                cull: Cull::None,
                label: Some(label.into()),
                ..Default::default()
            },
            vs,
            fs,
        )
        .expect("create_graphics_pso")
}

/// Draw `vertex_count × instance_count` into a fresh `size`×`size` RGBA8 texture
/// (cleared to opaque black) and read the result back to CPU bytes. `root` is bound
/// for both the vertex and pixel stages. The texture is transient.
fn render_draw(
    device: &Device,
    pso: &GraphicsPso,
    root: impl Into<Option<GpuAddress>>,
    size: u32,
    vertex_count: u32,
    instance_count: u32,
) -> Vec<u8> {
    let root = root.into();
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
    cmd.set_graphics_pipeline(pso);
    cmd.set_viewport(0.0, 0.0, size as f32, size as f32, 0.0, 1.0);
    cmd.set_scissor(0, 0, size, size);
    cmd.draw(root, root, vertex_count, instance_count, 0, 0);
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
/// transient per-draw arguments (root structs, configs). Caller releases it with
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
// Clip-space orientation: Kiln normalizes every backend to Y-up NDC, so a quad in
// the top-left NDC quadrant must land in the top-left of the read-back image. The
// graphics-pipeline counterpart to mesh.rs's `mesh_clip_space_is_y_up`; both share
// the viewport flip that does the normalization.
// ---------------------------------------------------------------------------

const ORIENT_BODY: &str = /*slang*/
    r#"
struct VOut { float4 pos : SV_Position; };

// Two triangles covering the top-left NDC quadrant (x in [-1,0], y in [0,1]).
static const float2 QUAD[6] = {
    float2(-1.0, 0.0), float2(0.0, 0.0), float2(-1.0, 1.0),
    float2(-1.0, 1.0), float2(0.0, 0.0), float2( 0.0, 1.0),
};

[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID)
{
    VOut o; o.pos = float4(QUAD[vid], 0.0, 1.0); return o;
}

[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return float4(1.0, 1.0, 1.0, 1.0); }
"#;

#[test]
fn graphics_clip_space_is_y_up() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };

    let Some(vs) =
        common::compile_shader_or_skip(&device, ORIENT_BODY, "vsMain", ShaderStage::Vertex)
    else {
        return;
    };
    let Some(fs) =
        common::compile_shader_or_skip(&device, ORIENT_BODY, "fsMain", ShaderStage::Pixel)
    else {
        return;
    };
    let pso = make_graphics_pso(&device, &vs, &fs, 16, "orient");

    const SIZE: u32 = 128;
    // This shader reads no root data, so the draw's root pointer is NULL — a draw
    // that carries nothing needs no allocation.
    let pixels = common::timed("graphics clip-space orientation · submit+wait", || {
        render_draw(&device, &pso, None, SIZE, 6, 1)
    });
    common::save_rgba_png("graphics_clip_space_is_y_up", SIZE, SIZE, &pixels);

    let half = SIZE / 2;
    for y in 0..SIZE {
        for x in 0..SIZE {
            let i = ((y * SIZE + x) * 4) as usize;
            let (r, g, b) = (pixels[i], pixels[i + 1], pixels[i + 2]);
            if x < half && y < half {
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
}

// ---------------------------------------------------------------------------
// Interpolated triangle: per-vertex RGB colours interpolated across the face.
// Verifies barycentric interpolation and rasterization coverage (corners stay at
// the clear colour). Orientation-independent: every check is symmetric about the
// image centre.
// ---------------------------------------------------------------------------

const TRI_BODY: &str = /*slang*/
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

#[test]
fn graphics_interpolated_triangle() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };

    let Some(vs) = common::compile_shader_or_skip(&device, TRI_BODY, "vsMain", ShaderStage::Vertex)
    else {
        return;
    };
    let Some(fs) = common::compile_shader_or_skip(&device, TRI_BODY, "fsMain", ShaderStage::Pixel)
    else {
        return;
    };
    let pso = make_graphics_pso(&device, &vs, &fs, 16, "tri");

    const SIZE: u32 = 128;
    // Per-vertex colours come from the shader; the draw reads no root data, so its
    // root pointer is NULL.
    let pixels = common::timed("graphics interpolated triangle · submit+wait", || {
        render_draw(&device, &pso, None, SIZE, 3, 1)
    });
    common::save_rgba_png("graphics_interpolated_triangle", SIZE, SIZE, &pixels);

    let at = |x: u32, y: u32| {
        let i = ((y * SIZE + x) * 4) as usize;
        (pixels[i], pixels[i + 1], pixels[i + 2])
    };

    // 1) Corners are outside the inset triangle → clear (black).
    for (x, y) in [(0, 0), (SIZE - 1, 0), (0, SIZE - 1), (SIZE - 1, SIZE - 1)] {
        let (r, g, b) = at(x, y);
        assert!(
            r < 5 && g < 5 && b < 5,
            "corner ({x},{y}) not clear: ({r},{g},{b})"
        );
    }

    // 2) Image centre (NDC origin) ≈ 0.5·red + 0.25·green + 0.25·blue ≈ (128,64,64).
    let (cr, cg, cb) = at(SIZE / 2, SIZE / 2);
    let near = |a: u8, b: i32| (a as i32 - b).abs() <= 6;
    assert!(
        near(cr, 128) && near(cg, 64) && near(cb, 64),
        "centre interpolation off: got ({cr},{cg},{cb}), expected ~(128,64,64)"
    );

    // 3) Each vertex colour is reached somewhere (interpolation endpoints).
    let (mut max_r, mut max_g, mut max_b) = ((0u8, 0u8, 0u8), (0u8, 0u8, 0u8), (0u8, 0u8, 0u8));
    for y in 0..SIZE {
        for x in 0..SIZE {
            let p = at(x, y);
            if p.0 > max_r.0 {
                max_r = p;
            }
            if p.1 > max_g.1 {
                max_g = p;
            }
            if p.2 > max_b.2 {
                max_b = p;
            }
        }
    }
    assert!(
        max_r.0 > 200 && max_r.1 < 80 && max_r.2 < 80,
        "no red-dominant pixel: {max_r:?}"
    );
    assert!(
        max_g.1 > 200 && max_g.0 < 80 && max_g.2 < 80,
        "no green-dominant pixel: {max_g:?}"
    );
    assert!(
        max_b.2 > 200 && max_b.0 < 80 && max_b.1 < 80,
        "no blue-dominant pixel: {max_b:?}"
    );
}

// ---------------------------------------------------------------------------
// Instanced grid: a G×G instanced draw where each instance (SV_InstanceID) emits a
// quad covering its own NDC cell, coloured by its instance coordinates. The
// vertex-pipeline analogue of mesh.rs's `mesh_meshlet_grid`; verifies instancing,
// SV_InstanceID, vertex-stage root reads, and per-instance placement.
// ---------------------------------------------------------------------------

const GRID: u32 = 4;
const GRID_SIZE: u32 = 256;

gpu_struct! {
    pub struct GridCfg {
        dim: u32 as "uint", // grid is dim × dim instances
    }
}

const GRID_BODY: &str = /*slang*/
    r#"
struct VOut { float4 pos : SV_Position; float4 color : COLOR; };

// NB: `cfg` is a GLOBAL uniform, not an entry-point parameter. As of slangc
// 2026.10, a `uniform T*` declared as a parameter of a *struct-returning vertex*
// shader is silently NOT bound on the Metal target (the generated entry has no
// [[buffer]] and reads an uninitialized pointer). A module-scope `uniform T*`
// still lowers to buffer(0) matching Kiln's root model and binds correctly.
uniform GridCfg* cfg;

static const float2 CORNER[6] = {
    float2(0,0), float2(1,0), float2(0,1),
    float2(0,1), float2(1,0), float2(1,1),
};

[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    uint g = cfg.dim;
    uint gx = iid % g;
    uint gy = iid / g;
    float cell = 2.0 / float(g);
    float2 base = float2(-1.0 + float(gx) * cell, -1.0 + float(gy) * cell);
    float2 p = base + CORNER[vid] * cell;
    VOut o;
    o.pos = float4(p, 0.0, 1.0);
    o.color = float4((float(gx) + 0.5) / float(g), (float(gy) + 0.5) / float(g), 0.0, 1.0);
    return o;
}

[shader("fragment")]
float4 fsMain(VOut i) : SV_Target { return i.color; }
"#;

#[test]
fn graphics_instanced_grid() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };

    let src = format!("{}{}", GridCfg::SLANG, GRID_BODY);
    let Some(vs) = common::compile_shader_or_skip(&device, &src, "vsMain", ShaderStage::Vertex)
    else {
        return;
    };
    let Some(fs) = common::compile_shader_or_skip(&device, &src, "fsMain", ShaderStage::Pixel)
    else {
        return;
    };
    let pso = make_graphics_pso(&device, &vs, &fs, 16, "grid");

    // Per-draw config from the bump allocator (the vertex shader reads grid dim).
    let mut bump = test_bump(&device);
    let cfg = bump
        .alloc(std::mem::size_of::<GridCfg>() as u64, 16)
        .expect("bump cfg");
    cfg.upload(&GridCfg { dim: GRID }).expect("upload cfg");

    let pixels = common::timed("instanced grid (4×4) · submit+wait", || {
        render_draw(&device, &pso, cfg.gpu, GRID_SIZE, 6, GRID * GRID)
    });
    common::save_rgba_png("graphics_instanced_grid", GRID_SIZE, GRID_SIZE, &pixels);

    // Each instance (gx, gy) fills one cell. Y-up: gid.y = 0 is the bottom NDC cell
    // → the bottom rows, so the pixel-row band for gy is (GRID-1-gy).
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
// Root data from the bump allocator: the doc's preferred path for per-draw GPU
// arguments (NoGraphicsApi.md appendix). Renders the full-screen-colour shader but
// sources its root struct from a transient `{ cpu, gpu }` bump allocation instead of
// a dedicated `malloc`, proving the GPU actually dereferences a bump-provided
// pointer. Uses blue to distinguish it from `graphics_fullscreen_color` (red).
// ---------------------------------------------------------------------------

#[test]
fn graphics_root_from_bump_allocator() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };

    let src = format!("{}{}", Root::SLANG, GFX_BODY);
    let Some(vs) = common::compile_shader_or_skip(&device, &src, "vsMain", ShaderStage::Vertex)
    else {
        return;
    };
    let Some(fs) = common::compile_shader_or_skip(&device, &src, "fsMain", ShaderStage::Pixel)
    else {
        return;
    };
    let pso = make_graphics_pso(&device, &vs, &fs, 16, "bump-root");

    // One CPU-mapped buffer behind a per-frame bump allocator; the root is a transient
    // sub-allocation from it (the appendix's `myBumpAllocator.allocate<Data>()`).
    let buffer = device
        .create_buffer(&BufferDesc {
            size: 4096,
            memory: MemoryType::Default,
            label: Some("bump-root".into()),
        })
        .expect("create_buffer");
    let mut bump = BumpAllocator::new(buffer);

    let root = bump
        .alloc(std::mem::size_of::<Root>() as u64, 16)
        .expect("bump alloc for root");
    root.upload(&Root {
        color: [0.0, 0.0, 1.0, 1.0],
    })
    .expect("upload root via bump cpu pointer");

    let pixels = common::timed("draw with bump-allocated root · submit+wait", || {
        render_draw(&device, &pso, root.gpu, SIZE, 3, 1)
    });
    common::save_rgba_png("graphics_root_from_bump_allocator", SIZE, SIZE, &pixels);

    // Every pixel opaque blue — the GPU read the colour from the bump allocation.
    for px in 0..(SIZE * SIZE) as usize {
        let (r, g, b, a) = (
            pixels[px * 4],
            pixels[px * 4 + 1],
            pixels[px * 4 + 2],
            pixels[px * 4 + 3],
        );
        assert_eq!(
            (r, g, b, a),
            (0, 0, 255, 255),
            "pixel {px} not blue: ({r},{g},{b},{a})"
        );
    }

    device.destroy_buffer(bump.into_buffer());
}
