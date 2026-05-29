//! Headless graphics path test (timed), driven by a backend-agnostic Slang shader.
//!
//! Renders a full-screen triangle into an offscreen RGBA8 texture, colouring every pixel
//! from a pointer-first root struct, then reads the texture back and verifies it. Exercises
//! the render-pass path (begin/end), graphics PSO creation, and the two-stage root binding.

mod common;

use spectradio_rhi::{
    ColorAttachment, ColorTarget, Cull, Format, GraphicsPsoDesc, LoadOp, MemoryType,
    RenderPassDesc, RenderTarget, SampleCount, ShaderStage, StageFlags, StoreOp, TextureDesc,
    TextureDimension, TextureUsage, Topology,
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
    let Some(_vs) = common::compile_shader_or_skip(&device, &src, "vsMain", ShaderStage::Vertex)
    else {
        return;
    };
    let Some(_fs) = common::compile_shader_or_skip(&device, &src, "fsMain", ShaderStage::Pixel)
    else {
        return;
    };

    let pso = common::timed("create_graphics_pso", || {
        device
            .create_graphics_pso(&GraphicsPsoDesc {
                vertex_shader: 0,
                pixel_shader: 1,
                topology: Topology::TriangleList,
                color_targets: vec![ColorTarget::new(Format::R8G8B8A8Unorm)],
                depth_format: None,
                sample_count: SampleCount::S1,
                root_constant_size: 16,
                cull: Cull::None,
                ..Default::default()
            })
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
        .create_texture(&tex_desc, tex_mem.gpu_address())
        .expect("create_texture");

    // Root color + readback buffer.
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
        cmd.draw(root.gpu_address(), root.gpu_address(), 3, 1, 0, 0);
        cmd.end_render_pass();

        cmd.barrier(StageFlags::RASTER_COLOR_OUT, StageFlags::TRANSFER);
        cmd.copy_from_texture(readback.gpu_address(), tex_mem.gpu_address(), &texture);
        cmd.barrier(StageFlags::TRANSFER, StageFlags::ALL_COMMANDS);
        cmd.end();
        let queue = device.queue();
        queue.submit(cmd).expect("submit");
        queue.wait_idle();
    });

    // Every pixel should be opaque red.
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
            (255, 0, 0, 255),
            "pixel {px} not red: ({r},{g},{b},{a})"
        );
    }

    device.free(root);
    device.free(readback);
}
