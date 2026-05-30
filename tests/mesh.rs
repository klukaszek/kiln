//! Headless mesh-shader path test (timed), driven by a backend-agnostic Slang shader.
//!
//! A mesh shader emits a full-screen triangle; the pixel shader colours it from a
//! pointer-first root. Renders to an offscreen texture and verifies via readback.
//! Skips if the device doesn't support mesh shaders.

mod common;

use kiln_rhi::gpu_struct;
use kiln_rhi::{
    ColorAttachment, ColorTarget, Cull, Format, LoadOp, MemoryType, MeshletPsoDesc, RenderPassDesc,
    RenderTarget, SampleCount, ShaderStage, StageFlags, StoreOp, TextureDesc, TextureDimension,
    TextureUsage, Topology,
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
        .create_texture(&tex_desc, tex_mem.gpu_address())
        .expect("create_texture");

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
        cmd.draw_meshlets(root.gpu_address(), root.gpu_address(), 1, 1, 1);
        cmd.end_render_pass();

        cmd.barrier(StageFlags::RASTER_COLOR_OUT, StageFlags::TRANSFER);
        cmd.copy_from_texture(readback.gpu_address(), tex_mem.gpu_address(), &texture);
        cmd.barrier(StageFlags::TRANSFER, StageFlags::ALL_COMMANDS);
        cmd.end();
        let queue = device.queue();
        queue.submit(cmd).expect("submit");
        queue.wait_idle();
    });

    let pixels = readback.as_slice::<u8>().expect("read readback");
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
