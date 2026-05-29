//! Headless ray-tracing path test (timed): inline ray query (RayQuery) in a compute kernel.
//!
//! Builds a one-triangle BLAS + a single-instance TLAS, then dispatches a Slang compute
//! shader that traces one ray at the triangle and writes hit/miss to a buffer. This is the
//! cross-backend (Metal + Vulkan) ray-tracing model — no RT pipelines / SBT.

mod common;

use spectradio_rhi::gpu_struct;
use spectradio_rhi::{
    BlasDesc, BlasMeshDesc, BuildAccelFlags, ComputePsoDesc, GeometryFlags, GeometryType,
    GpuAddress, MemoryType, ShaderStage, StageFlags, TlasDesc, TlasInstance,
};

gpu_struct! {
    pub struct Root {
        output: u64 as "uint*",
    }
}

// The TLAS is a trailing entry-point parameter so the root stays at the RHI's normal slot
// and the acceleration structure binds to the next slot (Metal buffer(1) / Vulkan descriptor).
const RQ_BODY: &str = /*slang*/
    r#"
[shader("compute")]
[numthreads(1, 1, 1)]
void rqMain(uint3 tid : SV_DispatchThreadID,
            uniform Root* data,
            uniform RaytracingAccelerationStructure tlas)
{
    RayDesc ray;
    ray.Origin = float3(0.0, 0.0, -1.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.0;
    ray.TMax = 1000.0;

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
    let Some(_module) = common::compile_shader_caps_or_skip(
        &device,
        &src,
        "rqMain",
        ShaderStage::Compute,
        &["spvRayQueryKHR"],
    ) else {
        return;
    };

    let pso = match device.create_compute_pso(&ComputePsoDesc {
        compute_shader: 0,
        root_constant_size: 8,
        threads_per_threadgroup: [1, 1, 1],
        label: Some("ray-query".into()),
    }) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("skipping: compute PSO creation failed ({e})");
            return;
        }
    };

    // One triangle at z=0 straddling the ray origin (0,0,-1) travelling +Z.
    let verts: [[f32; 3]; 3] = [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]];
    let vbuf = device
        .malloc(std::mem::size_of_val(&verts) as u64, MemoryType::Default)
        .expect("vertex buffer");
    vbuf.upload(&verts).expect("upload vertices");

    let blas_desc = BlasDesc {
        meshes: vec![BlasMeshDesc {
            geometry_type: GeometryType::Triangles,
            flags: GeometryFlags::OPAQUE,
            vertex_buffer: vbuf.gpu_address(),
            vertex_stride: 12,
            vertex_count: 3,
            index_buffer: GpuAddress(0),
            index_count: 0,
            aabb_buffer: GpuAddress(0),
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

    // Single identity instance referencing the BLAS, encoded in the native layout.
    let stride = device.tlas_instance_stride();
    let instbuf = device
        .malloc(stride as u64, MemoryType::Default)
        .expect("instance buffer");
    let instance = TlasInstance {
        transform: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        instance_custom_index_and_mask: 0xFF << 24, // mask = 0xFF
        instance_sbt_offset_and_flags: 0,
        acceleration_structure_reference: device.accel_gpu_address(&blas),
    };
    device
        .write_tlas_instance(&instbuf, 0, &instance)
        .expect("write instance");

    let tlas_desc = TlasDesc {
        instance_buffer: instbuf.gpu_address(),
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

    // Ray-query dispatch.
    let output = device.malloc(4, MemoryType::Readback).expect("output");
    let root = device
        .malloc(std::mem::size_of::<Root>() as u64, MemoryType::Default)
        .expect("root");
    root.upload(&Root {
        output: output.gpu_address().0,
    })
    .expect("upload root");

    common::timed("ray query dispatch · submit+wait", || {
        let mut cmd = device.create_command_buffer().expect("cmd");
        cmd.set_compute_pipeline(&pso);
        cmd.bind_acceleration_structure(1, &tlas);
        cmd.dispatch(root.gpu_address(), 1, 1, 1);
        cmd.barrier(StageFlags::COMPUTE, StageFlags::ALL_COMMANDS);
        cmd.end();
        let q = device.queue();
        q.submit(cmd).expect("submit");
        q.wait_idle();
    });

    let hit = output.read::<u32>().expect("read hit result");
    assert_eq!(hit, 1, "ray query should report a triangle hit");

    device.free(vbuf);
    device.free(instbuf);
    device.free(output);
    device.free(root);
}
