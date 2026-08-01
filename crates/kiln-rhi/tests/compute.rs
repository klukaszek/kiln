//! Headless compute path test driven by a backend-agnostic Slang shader.

mod common;

use kiln_rhi::{ComputePsoDesc, GpuAddress, MemoryType, ShaderStage, StageFlags, gpu_struct};

gpu_struct! {
    pub struct Data {
        input: GpuAddress as "uint*",
        output: GpuAddress as "uint*",
        count: u32,
        // Keep the host and Slang layouts identical.
        _pad: u32,
    }
}

const COMPUTE_BODY: &str = /*slang*/
    r#"
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
    let Some(module) =
        kiln_rhi::compiler::compile_or_skip(&device, &src, "computeMain", ShaderStage::Compute)
    else {
        return;
    };

    let pso = common::timed("create_compute_pso", || {
        device
            .create_compute_pso(
                &ComputePsoDesc {
                    threads_per_threadgroup: [64, 1, 1],
                    label: Some("double".into()),
                },
                &module,
            )
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

    input
        .upload_slice(&(0..N).collect::<Vec<u32>>())
        .expect("upload input");
    data.upload(&Data {
        input: input.gpu(),
        output: output.gpu(),
        count: N,
        _pad: 0,
    })
    .expect("upload root");

    common::timed("dispatch 1024 · submit+wait", || {
        let mut cmd = device.create_command_buffer().expect("cmd");
        cmd.set_compute_pipeline(&pso);
        cmd.dispatch(data.gpu(), N.div_ceil(64), 1, 1);
        cmd.barrier(StageFlags::COMPUTE, StageFlags::ALL_COMMANDS);
        // Submitting unrelated work must not retire a pipeline referenced by `cmd`.
        drop(pso);
        let queue = device.queue();
        let unrelated = device
            .create_command_buffer()
            .expect("unrelated command buffer");
        queue.submit(unrelated).expect("submit unrelated work");
        queue.wait_idle();
        queue.submit(cmd).expect("submit");
        queue.wait_idle();
    });

    let result = output.as_slice::<u32>().expect("read output");
    for (i, &value) in result.iter().enumerate() {
        assert_eq!(value, i as u32 * 2, "element {i} not doubled");
    }

    device.free(input);
    device.free(output);
    device.free(data);
}
