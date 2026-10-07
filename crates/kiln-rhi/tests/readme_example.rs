//! Compiles the `gpu_struct!` and bump-arena examples from the README, so the documentation
//! cannot drift from the API without a test failing.

use kiln_rhi::gpu_struct;
use kiln_rhi::{
    Allocation, BumpAllocator, CommandBuffer, Device, GraphicsPso, MemoryType, RhiResult,
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
fn readme_direct_example(device: &Device, vertices: &[Vertex], vertex_count: u32) -> RhiResult<()> {
    let vertex_buffer = device.upload_slice(vertices)?;

    let mut root = device
        .allocate::<DrawRoot>(MemoryType::Upload)?
        .labeled("draw-root");
    root.upload(&DrawRoot {
        vertices: vertex_buffer.gpu(),
        count: vertex_count,
        pad: 0,
    })?;
    Ok(())
}

/// The bump-arena snippet from the README.
#[allow(dead_code)]
fn readme_arena_example(
    device: &Device,
    vertex_buffer: &Allocation<Vertex>,
    vertex_count: u32,
) -> RhiResult<()> {
    let mut frame_arena = BumpAllocator::new(
        device
            .allocate_bytes(64 * 1024, MemoryType::Upload)?
            .labeled("frame-roots"),
    );

    frame_arena.reset();
    let _root = frame_arena
        .upload(&DrawRoot {
            vertices: vertex_buffer.gpu(),
            count: vertex_count,
            pad: 0,
        })
        .expect("frame arena exhausted");
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
