//! Compiles the `gpu_struct!` and bump-arena examples from the README, so the documentation
//! cannot drift from the API without a test failing.

use kiln_rhi::gpu_struct;
use kiln_rhi::{
    AllocationDesc, BumpAllocator, CommandBuffer, Device, GpuPtr, GraphicsPso, MemoryType,
    RhiResult,
};

gpu_struct! {
    pub struct Vertex {
        position: [f32; 3],
        pad: u32,
    }
}

gpu_struct! {
    pub struct DrawRoot {
        vertices: GpuPtr<Vertex>,
        count: u32,
        pad: u32,
    }
}

/// The direct-allocation snippet from the README.
#[allow(dead_code)]
fn readme_direct_example(device: &Device, vertex_count: u32) -> RhiResult<()> {
    let mut root = device.create_allocation(&AllocationDesc {
        size: size_of::<DrawRoot>() as u64,
        memory: MemoryType::Upload,
        label: Some("draw-root".into()),
        ..Default::default()
    })?;
    root.upload(&DrawRoot {
        vertices: GpuPtr::from_addr(0),
        count: vertex_count,
        pad: 0,
    })?;
    Ok(())
}

/// The bump-arena snippet from the README.
#[allow(dead_code)]
fn readme_arena_example(device: &Device, vertex_count: u32) -> RhiResult<()> {
    let allocation = device.create_allocation(&AllocationDesc {
        size: 64 * 1024,
        memory: MemoryType::Upload,
        label: Some("frame-roots".into()),
        ..Default::default()
    })?;
    let mut frame_arena = BumpAllocator::new(allocation);

    frame_arena.reset();
    let root = frame_arena
        .alloc(size_of::<DrawRoot>() as u64, 16)
        .expect("frame arena exhausted")
        .cast::<DrawRoot>();
    root.write(&DrawRoot {
        vertices: GpuPtr::from_addr(0),
        count: vertex_count,
        pad: 0,
    })?;
    Ok(())
}

/// The snippet at the top of the README.
#[allow(dead_code)]
fn readme_draw_example(
    cmd: &mut CommandBuffer,
    pipeline: &GraphicsPso,
    root: kiln_rhi::Allocation,
) {
    let vertex_count = 3;
    cmd.set_pipeline(pipeline);
    cmd.draw(root.gpu(), vertex_count, 1, 0, 0);
}

#[test]
fn readme_gpu_struct_emits_the_documented_slang() {
    assert_eq!(
        DrawRoot::SLANG,
        "struct DrawRoot {\n    Vertex* vertices;\n    uint count;\n    uint pad;\n};\n"
    );
}
