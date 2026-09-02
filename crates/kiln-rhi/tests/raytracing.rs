//! Headless inline ray-query test using a one-triangle BLAS and single-instance TLAS.

mod common;

use kiln_rhi::gpu_struct;
use kiln_rhi::{
    AccelHandle, BlasDesc, BlasMeshDesc, BuildAccelFlags, ComputePsoDesc, GeometryFlags,
    GeometryType, GpuPtr, MemoryType, ShaderStage, StageFlags, TlasDesc, TlasInstance,
};

gpu_struct! {
    pub struct Root {
        output: GpuPtr<u32> as "uint*",
        tlas: AccelHandle,
    }
}

// The TLAS is passed as a bindless handle in the root data on both backends.
const RQ_BODY: &str = /*slang*/
    r#"
[shader("compute")]
[numthreads(1, 1, 1)]
void rqMain(uint3 tid : SV_DispatchThreadID, uniform Root* data)
{
    RayDesc ray;
    ray.Origin = float3(0.0, 0.0, -1.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.0;
    ray.TMax = 1000.0;

    RaytracingAccelerationStructure tlas = data.tlas;
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);
    q.Proceed();
    data.output[0] = (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? 1u : 0u;
}
"#;

#[test]
fn ray_query_triangle_hit() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };

    let src = format!("{}{}", Root::SLANG, RQ_BODY);
    let Some(module) = kiln_rhi::compiler::compile_caps_or_skip(
        &device,
        &src,
        "rqMain",
        ShaderStage::Compute,
        &["spvRayQueryKHR"],
    ) else {
        return;
    };

    let pso = match device.create_compute_pso(
        &ComputePsoDesc {
            threads_per_threadgroup: [1, 1, 1],
            label: Some("ray-query".into()),
        },
        &module,
    ) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("skipping: compute PSO creation failed ({e})");
            return;
        }
    };

    // Triangle at z=0, with the ray starting at z=-1.
    let verts: [[f32; 3]; 3] = [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]];
    let mut vbuf = device
        .allocate(std::mem::size_of_val(&verts) as u64, MemoryType::Default)
        .expect("vertex buffer");
    vbuf.upload(&verts).expect("upload vertices");

    let blas_desc = BlasDesc {
        meshes: vec![BlasMeshDesc {
            geometry_type: GeometryType::Triangles,
            flags: GeometryFlags::OPAQUE,
            vertex_buffer: vbuf.ptr(),
            vertex_stride: 12,
            vertex_count: 3,
            index_buffer: GpuPtr::NULL,
            index_count: 0,
            aabb_buffer: GpuPtr::NULL,
            aabb_count: 0,
        }],
        flags: BuildAccelFlags::PREFER_FAST_TRACE,
    };
    let blas = match device.create_blas(&blas_desc) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping: ray tracing unsupported on this device ({e})");
            return;
        }
    };

    common::timed("build BLAS · submit+wait", || {
        let mut cmd = device.create_command_buffer().expect("cmd");
        cmd.build_blas(&blas, &blas_desc);
        cmd.end();
        let q = device.queue();
        q.submit(cmd).expect("submit");
        q.wait_idle();
    });

    // Identity instance referencing the BLAS.
    let stride = device.tlas_instance_stride();
    let instbuf = device
        .allocate(stride as u64, MemoryType::Default)
        .expect("instance buffer");
    let instance = TlasInstance {
        transform: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        instance_custom_index_and_mask: 0xFF << 24, // mask = 0xFF
        instance_sbt_offset_and_flags: 0,
        acceleration_structure_reference: blas.handle(),
    };
    device
        .write_tlas_instance(&instbuf, 0, &instance)
        .expect("write instance");

    let tlas_desc = TlasDesc {
        instance_buffer: instbuf.ptr(),
        instance_count: 1,
        flags: BuildAccelFlags::PREFER_FAST_TRACE,
    };
    let tlas = device.create_tlas(&tlas_desc).expect("create_tlas");

    common::timed("build TLAS · submit+wait", || {
        let mut cmd = device.create_command_buffer().expect("cmd");
        cmd.build_tlas(&tlas, &tlas_desc);
        cmd.end();
        let q = device.queue();
        q.submit(cmd).expect("submit");
        q.wait_idle();
    });

    let output = device.allocate(4, MemoryType::Readback).expect("output");
    let mut root = device
        .allocate(std::mem::size_of::<Root>() as u64, MemoryType::Default)
        .expect("root");
    root.upload(&Root {
        output: output.ptr(),
        tlas: tlas.handle(),
    })
    .expect("upload root");

    common::timed("ray query dispatch · submit+wait", || {
        let mut cmd = device.create_command_buffer().expect("cmd");
        cmd.set_compute_pipeline(&pso);
        cmd.dispatch(root.gpu(), 1, 1, 1);
        cmd.barrier(StageFlags::COMPUTE, StageFlags::ALL_COMMANDS);
        cmd.end();
        let q = device.queue();
        q.submit(cmd).expect("submit");
        q.wait_idle();
    });

    let hit = output.read::<u32>().expect("read hit result");
    assert_eq!(hit, 1, "ray query should report a triangle hit");

    // Resources borrowed while recording must remain alive until the submitted work retires.
    drop(tlas);
    drop(blas);
    device.free(vbuf);
    device.free(instbuf);
    device.free(output);
    device.free(root);
}
