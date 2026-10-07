//! Headless compute path test driven by a backend-agnostic Slang shader.

mod common;

use kiln_rhi::{ComputePsoDesc, MemoryType, ShaderStage, StageFlags, gpu_struct};

gpu_struct! {
    pub struct Data {
        input: GpuPtr<u32>,
        output: GpuPtr<u32>,
        count: u32,
        // Keep the host and Slang layouts identical.
        pad: u32,
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
fn compute_barrier_across_pipeline_switches() {
    let (device, _gpu) = common::device();
    let src = format!("{}{}", Data::SLANG, COMPUTE_BODY);
    let module =
        kiln_rhi::compiler::compile(&device, &src, "computeMain", ShaderStage::Compute, &[])
            .expect("compile module");
    // No `threads_per_threadgroup`: it comes from the shader's `[numthreads]` via reflection.
    let desc = ComputePsoDesc {
        label: Some("pipeline-switch dependency"),
        ..Default::default()
    };
    let pipelines = [
        device
            .create_compute_pso(&desc, &module)
            .expect("first PSO"),
        device
            .create_compute_pso(&desc, &module)
            .expect("second PSO"),
    ];
    const N: u32 = 65536;
    let mut a = device
        .allocate_array::<u32>(N as usize, MemoryType::Readback)
        .expect("a");
    let b = device
        .allocate_array::<u32>(N as usize, MemoryType::Readback)
        .expect("b");
    a.as_mut_slice().expect("mapped a").fill(1);
    let mut roots = device
        .allocate_array::<Data>(2, MemoryType::Upload)
        .expect("roots");
    roots
        .upload_slice(&[
            Data {
                input: a.gpu(),
                output: b.gpu(),
                count: N,
                pad: 0,
            },
            Data {
                input: b.gpu(),
                output: a.gpu(),
                count: N,
                pad: 0,
            },
        ])
        .expect("upload roots");
    let mut cmd = device.create_command_buffer().expect("cmd");
    for pass in 0..16 {
        let index = pass % 2;
        cmd.set_pipeline(&pipelines[index]);
        let root = roots.gpu().offset(index as u64);
        cmd.dispatch(root, N / 64, 1, 1);
        cmd.barrier(StageFlags::COMPUTE, StageFlags::COMPUTE);
    }
    device.queue().submit(cmd).expect("submit");
    device.queue().wait_idle();
    assert!(a.as_slice().expect("read a").iter().all(|&v| v == 1 << 16));
    device.destroy(a);
    device.destroy(b);
    device.destroy(roots);
}

#[test]
fn compute_doubles_buffer() {
    let (device, _gpu) = common::device();

    let src = format!("{}{}", Data::SLANG, COMPUTE_BODY);
    let module =
        kiln_rhi::compiler::compile(&device, &src, "computeMain", ShaderStage::Compute, &[])
            .expect("compile module");

    let pso = common::timed("create_compute_pso", || {
        device
            .create_compute_pso(
                &ComputePsoDesc {
                    label: Some("double"),
                    ..Default::default()
                },
                &module,
            )
            .expect("create_compute_pso")
    });

    const N: u32 = 1024;
    let mut input = device
        .allocate_array::<u32>(N as usize, MemoryType::Upload)
        .expect("input");
    let output = device
        .allocate_array::<u32>(N as usize, MemoryType::Readback)
        .expect("output");
    let mut data = device
        .allocate::<Data>(MemoryType::Upload)
        .expect("root data");

    input
        .upload_slice(&(0..N).collect::<Vec<u32>>())
        .expect("upload input");
    data.upload(&Data {
        input: input.gpu(),
        output: output.gpu(),
        count: N,
        pad: 0,
    })
    .expect("upload root");

    common::timed("dispatch 1024 · submit+wait", || {
        let mut cmd = device.create_command_buffer().expect("cmd");
        cmd.set_pipeline(&pso);
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

    let result = output.as_slice().expect("read output");
    for (i, &value) in result.iter().enumerate() {
        assert_eq!(value, i as u32 * 2, "element {i} not doubled");
    }

    device.destroy(input);
    device.destroy(output);
    device.destroy(data);
}

/// The shader's `[numthreads]` and an explicit `ComputePsoDesc` size must agree. They used to
/// diverge silently: Metal dispatched the descriptor's shape, Vulkan the shader's.
#[test]
fn a_threadgroup_size_disagreeing_with_the_shader_is_rejected() {
    let (device, _gpu) = common::device();
    let src = format!("{}{}", Data::SLANG, COMPUTE_BODY);
    let module =
        kiln_rhi::compiler::compile(&device, &src, "computeMain", ShaderStage::Compute, &[])
            .expect("compile module");

    assert_eq!(
        module.threads_per_threadgroup(),
        Some([64, 1, 1]),
        "reflection should carry the shader's declared [numthreads]"
    );

    let mismatched = device.create_compute_pso(
        &ComputePsoDesc {
            threads_per_threadgroup: Some([32, 1, 1]),
            label: Some("mismatched"),
        },
        &module,
    );
    let Err(err) = mismatched else {
        panic!("a size that disagrees with the shader must not build");
    };
    let message = err.to_string();
    assert!(
        message.contains("32") && message.contains("64"),
        "the error should name both sizes, got: {message}"
    );

    // The same size stated explicitly is fine.
    device
        .create_compute_pso(
            &ComputePsoDesc {
                threads_per_threadgroup: Some([64, 1, 1]),
                label: Some("matching"),
            },
            &module,
        )
        .expect("an explicit size matching the shader should build");
}
