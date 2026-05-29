//! Headless compute path test (timed), driven by a backend-agnostic Slang shader.
//!
//! One Slang source is compiled to SPIR-V (Vulkan) or metallib (Metal) by the harness and
//! dispatched through the high-level RHI. The `Data` root struct is declared once (below)
//! and shared between host and shader via `gpu_struct!`, so there is no manual byte layout.

mod common;

use spectradio_rhi::{ComputePsoDesc, MemoryType, ShaderStage, StageFlags};

// Shared host/device data contract. `Data::SLANG` is the matching Slang declaration.
gpu_struct! {
    pub struct Data {
        input: u64 as "uint*",
        output: u64 as "uint*",
        count: u32 as "uint",
    }
}

const COMPUTE_BODY: &str = r#"
[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID, uniform Data* data)
{
    if (tid.x >= data.count)
        return;
    data.output[tid.x] = data.input[tid.x] * 2u;
}
"#;

#[test]
fn compute_doubles_buffer() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };

    let src = format!("{}{}", Data::SLANG, COMPUTE_BODY);
    let Some(_module) =
        common::compile_shader_or_skip(&device, &src, "computeMain", ShaderStage::Compute)
    else {
        return;
    };

    // First (and only) module compiled → index 0.
    let pso = common::timed("create_compute_pso", || {
        device
            .create_compute_pso(&ComputePsoDesc {
                compute_shader: 0,
                root_constant_size: 16,
                threads_per_threadgroup: [64, 1, 1],
                label: Some("double".into()),
            })
            .expect("create_compute_pso")
    });

    const N: u32 = 1024;
    let input = device
        .malloc((N * 4) as u64, MemoryType::Default)
        .expect("input");
    let output = device
        .malloc((N * 4) as u64, MemoryType::Readback)
        .expect("output");
    let data = device
        .malloc(std::mem::size_of::<Data>() as u64, MemoryType::Default)
        .expect("root data");

    unsafe {
        let ip = input.mapped_ptr().expect("input mapped") as *mut u32;
        for i in 0..N {
            *ip.add(i as usize) = i;
        }
        // Build the root struct type-safely — no hand-computed offsets.
        common::write_mapped(
            data.mapped_ptr().expect("data mapped"),
            Data {
                input: input.gpu_address().0,
                output: output.gpu_address().0,
                count: N,
            },
        );
    }

    common::timed("dispatch 1024 · submit+wait", || {
        let mut cmd = device.create_command_buffer().expect("cmd");
        cmd.set_compute_pipeline(&pso);
        cmd.dispatch(data.gpu_address(), N.div_ceil(64), 1, 1);
        cmd.barrier(StageFlags::COMPUTE, StageFlags::ALL_COMMANDS);
        cmd.end();
        let queue = device.queue();
        queue.submit(cmd).expect("submit");
        queue.wait_idle();
    });

    unsafe {
        let op = output.mapped_ptr().expect("output mapped") as *const u32;
        for i in 0..N {
            assert_eq!(*op.add(i as usize), i * 2, "element {i} not doubled");
        }
    }

    device.free(input);
    device.free(output);
    device.free(data);
}
