//! Headless texture and sampler path tests.

mod common;

use kiln_rhi::gpu_struct;
use kiln_rhi::{
    ALL_LAYERS, ALL_MIPS, AddressMode, ColorAttachment, ColorTarget, Cull, FilterMode, Format,
    GraphicsPsoDesc, HazardFlags, LoadOp, MemoryType, RenderPassDesc, SampleCount, SamplerDesc,
    SamplerHandle, ShaderStage, StageFlags, StoreOp, TextureDesc, TextureDimension, TextureHandle,
    TextureUsage, TextureViewDesc, Topology, ViewKind,
};

const W: u32 = 64;
const H: u32 = 64;
const BPP: usize = 4; // R8G8B8A8

fn test_texture_desc() -> TextureDesc {
    TextureDesc {
        width: W,
        height: H,
        depth: 1,
        mip_levels: 1,
        array_layers: 1,
        format: Format::R8G8B8A8Unorm,
        dimension: TextureDimension::D2,
        sample_count: SampleCount::S1,
        usage: TextureUsage::SAMPLED
            | TextureUsage::STORAGE
            | TextureUsage::TRANSFER_SRC
            | TextureUsage::TRANSFER_DST,
        label: Some("rhi-test-tex".into()),
    }
}

/// Placement-allocate a texture, then register sampled + storage bindless views.
#[test]
fn texture_create_and_views() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };
    let desc = test_texture_desc();

    let size_align = common::timed("texture_size_align", || {
        device.texture_size_align(&desc).expect("size_align")
    });
    eprintln!(
        "    texture {}x{} RGBA8 → size={} align={}",
        W, H, size_align.size, size_align.align
    );

    let mem = device
        .allocate_aligned(size_align.size, size_align.align, MemoryType::GpuOnly)
        .expect("texture backing memory");
    let mut texture = common::timed("create_texture (placement)", || {
        device
            .create_texture(&desc, mem.gpu())
            .expect("create_texture")
    });

    let view = TextureViewDesc {
        format: None,
        base_mip: 0,
        mip_count: ALL_MIPS,
        base_layer: 0,
        layer_count: ALL_LAYERS,
    };
    let sampled = common::timed("Texture::view (sampled)", || {
        texture
            .view(ViewKind::Sampled, &view)
            .expect("sampled view")
    });
    let storage = common::timed("Texture::view (storage)", || {
        texture
            .view(ViewKind::Storage, &view)
            .expect("storage view")
    });
    assert!(!sampled.is_null());
    assert!(!storage.is_null());
    device.destroy(texture);
    device.destroy(mem);
}

/// Upload a pattern into a texture and read it straight back out — exercises both
/// Texture upload and readback with a GPU round-trip and CPU verification.
#[test]
fn texture_copy_roundtrip() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };
    let desc = test_texture_desc();
    let size_align = device.texture_size_align(&desc).expect("size_align");
    let mem = device
        .allocate_aligned(size_align.size, size_align.align, MemoryType::GpuOnly)
        .expect("texture backing");
    let texture = device
        .create_texture(&desc, mem.gpu())
        .expect("create_texture");

    let bytes = (W as usize) * (H as usize) * BPP;
    let mut src = device
        .allocate(bytes as u64, MemoryType::Upload)
        .expect("upload");
    let dst = device
        .allocate(bytes as u64, MemoryType::Readback)
        .expect("readback");

    for (i, b) in src
        .as_mut_slice::<u8>()
        .expect("src slice")
        .iter_mut()
        .enumerate()
    {
        *b = (i as u8).wrapping_mul(31).wrapping_add(5);
    }

    common::timed("upload→texture→readback · submit+wait", || {
        let mut cmd = device.create_command_buffer().expect("cmd");
        cmd.copy_buffer_to_texture(src.gpu(), &texture);
        cmd.barrier(StageFlags::TRANSFER, StageFlags::TRANSFER);
        cmd.copy_texture_to_buffer(&texture, dst.gpu());
        cmd.barrier(StageFlags::TRANSFER, StageFlags::ALL_COMMANDS);
        cmd.end();
        let queue = device.queue();
        queue.submit(cmd).expect("submit");
        queue.wait_idle();
    });

    for (i, &b) in dst.as_slice::<u8>().expect("dst slice").iter().enumerate() {
        let expected = (i as u8).wrapping_mul(31).wrapping_add(5);
        assert_eq!(b, expected, "texel byte {i} mismatch");
    }

    device.destroy(src);
    device.destroy(dst);
    device.destroy(texture);
    device.destroy(mem);
}

gpu_struct! {
    pub struct ViewRoot {
        unorm: TextureHandle,
        srgb:  TextureHandle,
        samp:  SamplerHandle,
    }
}

const VIEW_BODY: &str = /*slang*/
    r#"
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

[shader("vertex")]
VOut vsMain(uint vid : SV_VertexID)
{
    float2 p = float2(float((vid << 1) & 2), float(vid & 2));
    VOut o;
    o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = p;
    return o;
}

// Left half samples the texture through its native UNORM view, right half through an sRGB
// view of the very same texels. Both are sampled unconditionally so the branch cannot be
// blamed for a difference.
[shader("fragment")]
float4 fsMain(VOut i, uniform ViewRoot* r) : SV_Target
{
    SamplerState samp = r.samp;
    Texture2D asUnorm = r.unorm;
    Texture2D asSrgb  = r.srgb;
    float4 a = asUnorm.Sample(samp, i.uv);
    float4 b = asSrgb.Sample(samp, i.uv);
    return i.uv.x < 0.5 ? a : b;
}
"#;

fn srgb_to_linear_u8(value: u8) -> u8 {
    let c = value as f32 / 255.0;
    let linear = if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    };
    (linear * 255.0).round() as u8
}

/// A format-reinterpreting view must actually change how the GPU decodes the texels, not merely
/// pass validation. One RGBA8Unorm texture is sampled through its native view and through an
/// sRGB view of the same memory; the sRGB view has to apply the sRGB EOTF, so the same bytes come
/// back materially darker.
#[test]
fn srgb_texture_view_reinterprets_the_same_texels() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };

    let src = format!("{}{}", ViewRoot::SLANG, VIEW_BODY);
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

    const TEX: u32 = 4;
    const TEXEL: [u8; 4] = [32, 64, 96, 255];

    let pso = device
        .create_graphics_pso(
            &GraphicsPsoDesc {
                topology: Topology::TriangleList,
                color_targets: vec![ColorTarget::new(Format::R8G8B8A8Unorm)],
                depth_format: None,
                sample_count: SampleCount::S1,
                cull: Cull::None,
                label: Some("srgb-view".into()),
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
        usage: TextureUsage::SAMPLED | TextureUsage::TRANSFER_DST | TextureUsage::FORMAT_VIEW,
        label: Some("srgb-view-src".into()),
    };
    let tex_sa = device.texture_size_align(&tex_desc).expect("size_align");
    let tex_mem = device
        .allocate_aligned(tex_sa.size, tex_sa.align, MemoryType::GpuOnly)
        .expect("tex mem");
    let mut texture = device
        .create_texture(&tex_desc, tex_mem.gpu())
        .expect("create_texture");

    let srgb_handle = common::timed("sampled_view (format reinterpret)", || {
        texture
            .view(
                ViewKind::Sampled,
                &TextureViewDesc {
                    format: Some(Format::R8G8B8A8Srgb),
                    ..Default::default()
                },
            )
            .expect("sRGB view of an RGBA8Unorm texture")
    });

    let staging = device
        .upload_slice(&TEXEL.repeat((TEX * TEX) as usize))
        .expect("staging");
    let sampler = device
        .create_sampler(&SamplerDesc {
            min_filter: FilterMode::Nearest,
            mag_filter: FilterMode::Nearest,
            mip_filter: FilterMode::Nearest,
            address_u: AddressMode::ClampToEdge,
            address_v: AddressMode::ClampToEdge,
            address_w: AddressMode::ClampToEdge,
            ..Default::default()
        })
        .expect("create_sampler");

    let mut root = device
        .allocate(std::mem::size_of::<ViewRoot>() as u64, MemoryType::Upload)
        .expect("root");
    root.upload(&ViewRoot {
        unorm: texture.gpu(),
        srgb: srgb_handle,
        samp: sampler.gpu(),
    })
    .expect("upload root");

    let rt_desc = TextureDesc {
        width: W,
        height: H,
        depth: 1,
        mip_levels: 1,
        array_layers: 1,
        format: Format::R8G8B8A8Unorm,
        dimension: TextureDimension::D2,
        sample_count: SampleCount::S1,
        usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::TRANSFER_SRC,
        label: Some("srgb-view-rt".into()),
    };
    let rt_sa = device.texture_size_align(&rt_desc).expect("rt size_align");
    let rt_mem = device
        .allocate_aligned(rt_sa.size, rt_sa.align, MemoryType::GpuOnly)
        .expect("rt mem");
    let rt = device.create_texture(&rt_desc, rt_mem.gpu()).expect("rt");
    let readback = device
        .allocate((W * H * 4) as u64, MemoryType::Readback)
        .expect("readback");

    common::timed("sample through both views · submit+wait", || {
        let mut cmd = device.create_command_buffer().expect("cmd");
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
            render_area: [0, 0, W, H],
            label: Some("srgb view test"),
        });
        cmd.set_pipeline(&pso);
        cmd.set_viewport(0.0, 0.0, W as f32, H as f32, 0.0, 1.0);
        cmd.set_scissor(0, 0, W, H);
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

    let pixels = readback.as_slice::<u8>().expect("readback slice");
    common::save_rgba_png("srgb_texture_view", W, H, pixels);

    let texel_at = |x: u32, y: u32| {
        let i = ((y * W + x) * 4) as usize;
        [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
    };
    let expected_srgb = [
        srgb_to_linear_u8(TEXEL[0]),
        srgb_to_linear_u8(TEXEL[1]),
        srgb_to_linear_u8(TEXEL[2]),
        TEXEL[3], // the sRGB transfer function does not apply to alpha
    ];
    eprintln!("    UNORM view → {TEXEL:?}   sRGB view → {expected_srgb:?}");

    let near = |a: u8, b: u8, tol: i32| (a as i32 - b as i32).abs() <= tol;
    for y in (0..H).step_by(8) {
        let left = texel_at(W / 4, y);
        let right = texel_at(3 * W / 4, y);
        for c in 0..4 {
            assert!(
                near(left[c], TEXEL[c], 1),
                "row {y} UNORM view channel {c}: got {left:?}, expected {TEXEL:?}"
            );
            assert!(
                near(right[c], expected_srgb[c], 2),
                "row {y} sRGB view channel {c}: got {right:?}, expected {expected_srgb:?}"
            );
        }
        // The whole point: the two views must not agree on colour.
        assert!(
            right[0] < left[0] && right[1] < left[1] && right[2] < left[2],
            "row {y}: sRGB view {right:?} should decode darker than UNORM view {left:?}"
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
