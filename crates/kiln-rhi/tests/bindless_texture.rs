//! Headless bindless texture-sampling test.
//!
//! Samples a bindless texture through handles stored in the root data.

mod common;

use kiln_rhi::gpu_struct;
use kiln_rhi::{
    AddressMode, ColorAttachment, ColorTarget, Cull, FilterMode, Format, GraphicsPsoDesc,
    HazardFlags, LoadOp, MemoryType, RenderPassDesc, SampleCount, SamplerDesc, SamplerHandle,
    ShaderStage, StageFlags, StoreOp, TextureDesc, TextureDimension, TextureHandle, TextureUsage,
    Topology,
};

// `gpu_struct!` maps these fields to Slang descriptor handles.
gpu_struct! {
    pub struct Root {
        tex:  TextureHandle,
        samp: SamplerHandle,
    }
}

const BODY: &str = /*slang*/
    r#"
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID)
{
    // Full-screen triangle; uv spans 0..1 across the visible region.
    float2 p = float2(float((vid << 1) & 2), float(vid & 2));
    VOut o;
    o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = p;
    return o;
}

[shader("fragment")]
float4 fsMain(VOut i, uniform Root* r) : SV_Target
{
    Texture2D tex = r.tex;
    SamplerState samp = r.samp;
    return tex.Sample(samp, i.uv);
}
"#;

const SIZE: u32 = 64;
const TEX: u32 = 4;
// Distinct from the clear colour, and each channel round-trips cleanly through u8 unorm.
const TEXEL: [u8; 4] = [32, 64, 96, 255];

#[test]
fn bindless_texture_sample() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };

    let src = format!("{}{}", Root::SLANG, BODY);
    let Some(vs) =
        kiln_rhi::compiler::compile_or_skip(&device, &src, "vsMain", ShaderStage::Vertex, &[])
    else {
        return;
    };
    let Some(fs) =
        kiln_rhi::compiler::compile_or_skip(&device, &src, "fsMain", ShaderStage::Pixel, &[])
    else {
        return;
    };

    let pso = device
        .create_graphics_pso(
            &GraphicsPsoDesc {
                topology: Topology::TriangleList,
                color_targets: vec![ColorTarget::new(Format::R8G8B8A8Unorm)],
                depth_format: None,
                sample_count: SampleCount::S1,
                cull: Cull::None,
                label: Some("bindless-tex".into()),
                ..Default::default()
            },
            &vs,
            &fs,
        )
        .expect("create_graphics_pso");

    let tex_desc = TextureDesc {
        width: TEX,
        height: TEX,
        depth: 1,
        mip_levels: 1,
        array_layers: 1,
        format: Format::R8G8B8A8Unorm,
        dimension: TextureDimension::D2,
        sample_count: SampleCount::S1,
        usage: TextureUsage::SAMPLED | TextureUsage::TRANSFER_DST,
        label: Some("bindless-src-tex".into()),
    };
    let tex_sa = device
        .texture_size_align(&tex_desc)
        .expect("tex size_align");
    let tex_mem = device
        .allocate_aligned(tex_sa.size, tex_sa.align, MemoryType::GpuOnly)
        .expect("tex mem");
    let texture = device
        .create_texture(&tex_desc, tex_mem.gpu())
        .expect("create_texture");

    let staging = device
        .upload_slice(&TEXEL.repeat((TEX * TEX) as usize))
        .expect("staging upload");

    let sampler = device
        .create_sampler(&SamplerDesc {
            min_filter: FilterMode::Nearest,
            mag_filter: FilterMode::Nearest,
            mip_filter: FilterMode::Nearest,
            address_u: AddressMode::ClampToEdge,
            address_v: AddressMode::ClampToEdge,
            address_w: AddressMode::ClampToEdge,
            label: Some("bindless-sampler".into()),
            ..Default::default()
        })
        .expect("create_sampler");

    let mut root = device
        .allocate(std::mem::size_of::<Root>() as u64, MemoryType::Upload)
        .expect("root");
    root.upload(&Root {
        tex: texture.gpu(),
        samp: sampler.gpu(),
    })
    .expect("upload root");

    let rt_desc = TextureDesc {
        width: SIZE,
        height: SIZE,
        depth: 1,
        mip_levels: 1,
        array_layers: 1,
        format: Format::R8G8B8A8Unorm,
        dimension: TextureDimension::D2,
        sample_count: SampleCount::S1,
        usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::TRANSFER_SRC,
        label: Some("bindless-rt".into()),
    };
    let rt_sa = device.texture_size_align(&rt_desc).expect("rt size_align");
    let rt_mem = device
        .allocate_aligned(rt_sa.size, rt_sa.align, MemoryType::GpuOnly)
        .expect("rt mem");
    let rt = device
        .create_texture(&rt_desc, rt_mem.gpu())
        .expect("create rt");
    let readback = device
        .allocate((SIZE * SIZE * 4) as u64, MemoryType::Readback)
        .expect("readback");

    common::timed("sample bindless texture · submit+wait", || {
        let mut cmd = device.create_command_buffer().expect("cmd");
        // Make the upload and descriptor visible before sampling.
        cmd.copy_buffer_to_texture(staging.gpu(), &texture);
        cmd.barrier_with_hazard(
            StageFlags::TRANSFER,
            StageFlags::PIXEL_SHADER,
            HazardFlags::DESCRIPTORS,
        );

        cmd.begin_render_pass(&RenderPassDesc {
            color_attachments: vec![ColorAttachment {
                target: rt.target(),
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                clear_color: [0.0, 0.0, 0.0, 1.0],
            }],
            depth_attachment: None,
            render_area: [0, 0, SIZE, SIZE],
            label: Some("bindless texture test"),
        });
        cmd.set_pipeline(&pso);
        cmd.set_viewport(0.0, 0.0, SIZE as f32, SIZE as f32, 0.0, 1.0);
        cmd.set_scissor(0, 0, SIZE, SIZE);
        cmd.draw(root.gpu(), 3, 1, 0, 0);
        cmd.end_render_pass();

        cmd.barrier(StageFlags::RASTER_COLOR_OUT, StageFlags::TRANSFER);
        cmd.copy_texture_to_buffer(&rt, readback.gpu());
        cmd.barrier(StageFlags::TRANSFER, StageFlags::ALL_COMMANDS);
        cmd.end();
        let queue = device.queue();
        queue.submit(cmd).expect("submit");
        queue.wait_idle();
    });

    let pixels = readback.as_slice::<u8>().expect("read readback");
    common::save_rgba_png("bindless_texture_sample", SIZE, SIZE, pixels);

    let near = |a: u8, b: u8| (a as i32 - b as i32).abs() <= 1;
    for px in 0..(SIZE * SIZE) as usize {
        let got = [
            pixels[px * 4],
            pixels[px * 4 + 1],
            pixels[px * 4 + 2],
            pixels[px * 4 + 3],
        ];
        assert!(
            near(got[0], TEXEL[0])
                && near(got[1], TEXEL[1])
                && near(got[2], TEXEL[2])
                && near(got[3], TEXEL[3]),
            "pixel {px}: sampled {got:?}, expected {TEXEL:?}"
        );
    }

    device.destroy(root);
    device.destroy(staging);
    device.destroy(readback);
    device.destroy(texture);
    device.destroy(tex_mem);
    device.destroy(rt);
    device.destroy(rt_mem);
    device.destroy(sampler);
}
