//! Headless texture + sampler path tests (timed).

mod common;

use kiln_rhi::{
    AddressMode, FilterMode, Format, GpuViewDesc, MemoryType, SampleCount, SamplerDesc, StageFlags,
    TextureDesc, TextureDimension, TextureUsage, ALL_LAYERS, ALL_MIPS,
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
fn texture_create_and_view_descriptors() {
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
        .malloc_aligned(size_align.size, size_align.align, MemoryType::GpuOnly)
        .expect("texture backing memory");
    let texture = common::timed("create_texture (placement)", || {
        device
            .create_texture(&desc, mem.gpu())
            .expect("create_texture")
    });

    let view = GpuViewDesc {
        format: None,
        base_mip: 0,
        mip_count: ALL_MIPS,
        base_layer: 0,
        layer_count: ALL_LAYERS,
    };
    let sampled = common::timed("texture_view_descriptor (sampled)", || {
        device
            .texture_view_descriptor(&texture, &view)
            .expect("sampled view")
    });
    let storage = common::timed("rw_texture_view_descriptor (storage)", || {
        device
            .rw_texture_view_descriptor(&texture, &view)
            .expect("storage view")
    });
    assert_ne!(
        sampled, storage,
        "distinct views should get distinct bindless ids"
    );

    // texture drops before `mem` (reverse declaration order), so the backing outlives it.
}

/// Sampler creation cost.
#[test]
fn sampler_creation() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };

    let desc = SamplerDesc {
        min_filter: FilterMode::Linear,
        mag_filter: FilterMode::Linear,
        mip_filter: FilterMode::Linear,
        address_u: AddressMode::ClampToEdge,
        address_v: AddressMode::ClampToEdge,
        address_w: AddressMode::ClampToEdge,
        mip_lod_bias: 0.0,
        max_anisotropy: None,
        compare: None,
        min_lod: 0.0,
        max_lod: 0.0,
        label: Some("rhi-test-sampler".into()),
    };

    let _sampler = common::timed("create_sampler", || {
        device.create_sampler(&desc).expect("create_sampler")
    });
}

/// Upload a pattern into a texture and read it straight back out — exercises both
/// `copy_to_texture` and `copy_from_texture` with a GPU round-trip and CPU verification.
#[test]
fn texture_copy_roundtrip() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };
    let desc = test_texture_desc();
    let size_align = device.texture_size_align(&desc).expect("size_align");
    let mem = device
        .malloc_aligned(size_align.size, size_align.align, MemoryType::GpuOnly)
        .expect("texture backing");
    let texture = device
        .create_texture(&desc, mem.gpu())
        .expect("create_texture");

    let bytes = (W as usize) * (H as usize) * BPP;
    let mut src = device
        .malloc(bytes as u64, MemoryType::Default)
        .expect("upload");
    let dst = device
        .malloc(bytes as u64, MemoryType::Readback)
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
        cmd.copy_to_texture(mem.gpu(), src.gpu(), &texture);
        cmd.barrier(StageFlags::TRANSFER, StageFlags::TRANSFER);
        cmd.copy_from_texture(dst.gpu(), mem.gpu(), &texture);
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

    device.free(src);
    device.free(dst);
}
