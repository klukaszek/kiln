//! Headless inline ray-query test using a one-triangle BLAS and single-instance TLAS.

mod common;

use kiln_rhi::gpu_struct;
use kiln_rhi::{
    BlasDesc, BlasGeometry, BlasMeshDesc, BuildAccelFlags, ComputePsoDesc, GeometryFlags,
    MemoryType, ShaderStage, StageFlags, TlasDesc, TlasInstance,
};

gpu_struct! {
    pub struct Root {
        output: GpuPtr<u32>,
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

/// The one-triangle scene both tests trace against: a BLAS, a single-instance TLAS, the buffers
/// backing them, and the ray-query pipeline. Built once per test rather than inline, so the
/// destroy-ordering test can reuse it without repeating 60 lines of setup.
struct Scene {
    blas: kiln_rhi::AccelerationStructure,
    tlas: kiln_rhi::AccelerationStructure,
    vbuf: kiln_rhi::Allocation<[f32; 3]>,
    instbuf: kiln_rhi::Allocation,
    pso: kiln_rhi::ComputePso,
}

impl Scene {
    /// Release everything except the acceleration structures, which each test disposes of itself.
    fn destroy_buffers(
        self,
        device: &kiln_rhi::Device,
    ) -> (
        kiln_rhi::AccelerationStructure,
        kiln_rhi::AccelerationStructure,
    ) {
        device.destroy(self.vbuf);
        device.destroy(self.instbuf);
        drop(self.pso);
        (self.blas, self.tlas)
    }
}

fn build_scene(device: &kiln_rhi::Device) -> Scene {
    let src = format!("{}{}", Root::SLANG, RQ_BODY);
    let module = kiln_rhi::compiler::compile(
        device,
        &src,
        "rqMain",
        ShaderStage::Compute,
        &["spvRayQueryKHR"],
    )
    .expect("compile module");

    let pso = device
        .create_compute_pso(
            &ComputePsoDesc {
                label: Some("ray-query"),
                ..Default::default()
            },
            &module,
        )
        .expect("create_compute_pso");

    // Triangle at z=0, with the ray starting at z=-1.
    let verts: [[f32; 3]; 3] = [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]];
    let vbuf = device.upload_slice(&verts).expect("vertex buffer");

    let blas_desc = BlasDesc {
        meshes: &[BlasMeshDesc {
            flags: GeometryFlags::OPAQUE,
            geometry: BlasGeometry::Triangles {
                vertices: vbuf.gpu(),
                stride: 12,
                count: 3,
                indices: None,
            },
        }],
        flags: BuildAccelFlags::PREFER_FAST_TRACE,
    };
    let blas = device.create_blas(&blas_desc).expect("create_blas");

    common::timed("build BLAS · submit+wait", || {
        let mut cmd = device.create_command_buffer().expect("cmd");
        cmd.build_blas(&blas, &blas_desc);
        cmd.end().expect("end command buffer");
        let q = device.queue();
        q.submit(cmd).expect("submit");
        q.wait_idle();
    });

    // Identity instance referencing the BLAS.
    let stride = device.tlas_instance_stride();
    let mut instbuf = device
        .allocate_bytes(stride as u64, MemoryType::Upload)
        .expect("instance buffer");
    let instance = TlasInstance {
        transform: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        instance_custom_index_and_mask: 0xFF << 24, // mask = 0xFF
        instance_sbt_offset_and_flags: 0,
        acceleration_structure_reference: blas.gpu(),
    };
    device
        .write_tlas_instance(&mut instbuf, 0, &instance)
        .expect("write instance");

    let tlas_desc = TlasDesc {
        instance_buffer: instbuf.gpu().cast(),
        instance_count: 1,
        flags: BuildAccelFlags::PREFER_FAST_TRACE,
    };
    let tlas = device.create_tlas(&tlas_desc).expect("create_tlas");

    common::timed("build TLAS · submit+wait", || {
        let mut cmd = device.create_command_buffer().expect("cmd");
        cmd.build_tlas(&tlas, &tlas_desc);
        cmd.end().expect("end command buffer");
        let q = device.queue();
        q.submit(cmd).expect("submit");
        q.wait_idle();
    });

    Scene {
        blas,
        tlas,
        vbuf,
        instbuf,
        pso,
    }
}

/// Record one ray-query dispatch against `scene`, returning the root and output allocations so
/// the caller controls when they are released relative to the submit.
fn dispatch_ray_query(
    device: &kiln_rhi::Device,
    scene: &Scene,
) -> (kiln_rhi::Allocation<u32>, kiln_rhi::Allocation<Root>) {
    let output = device
        .allocate::<u32>(MemoryType::Readback)
        .expect("output");
    let mut root = device.allocate::<Root>(MemoryType::Upload).expect("root");
    root.upload(&Root {
        output: output.gpu(),
        tlas: scene.tlas.gpu(),
    })
    .expect("upload root");

    let mut cmd = device.create_command_buffer().expect("cmd");
    cmd.set_pipeline(&scene.pso);
    cmd.dispatch(root.gpu(), 1, 1, 1);
    cmd.barrier(StageFlags::COMPUTE, StageFlags::ALL_COMMANDS);
    cmd.end().expect("end command buffer");
    device.queue().submit(cmd).expect("submit");
    (output, root)
}

#[test]
fn ray_query_triangle_hit() {
    let (device, _gpu) = common::device();
    let scene = build_scene(&device);

    let (output, root) = common::timed("ray query dispatch · submit+wait", || {
        let handles = dispatch_ray_query(&device, &scene);
        device.queue().wait_idle();
        handles
    });

    let hit = output.read().expect("read hit result");
    assert_eq!(hit, 1, "ray query should report a triangle hit");

    // An acceleration structure is a `DeviceResource`: `destroy` hands it to the retirement
    // queue, which holds it until the submissions tracing against it have retired.
    let (blas, tlas) = scene.destroy_buffers(&device);
    device.destroy(tlas);
    device.destroy(blas);
    device.destroy(output);
    device.destroy(root);
}

/// Destroying a TLAS with a ray-query dispatch still in flight must not pull the structure out
/// from under the GPU. An acceleration structure is a `DeviceResource`, so its release is
/// deferred to the submission fence exactly like an `Allocation`'s.
#[test]
fn destroying_an_acceleration_structure_in_flight_defers_its_release() {
    let (device, _gpu) = common::device();
    let scene = build_scene(&device);

    let (output, root) = dispatch_ray_query(&device, &scene);

    // No fence of any kind between the submit and the destroy.
    let (blas, tlas) = scene.destroy_buffers(&device);
    device.destroy(tlas);
    device.destroy(blas);

    // Churn the pool so a prematurely released range would be handed straight back out.
    let squatter = device
        .allocate_bytes(1 << 20, MemoryType::Upload)
        .expect("squatter");

    device.queue().wait_idle();

    let hit = output.read().expect("read hit result");
    assert_eq!(
        hit, 1,
        "the ray query must still see the TLAS it was recorded against"
    );

    device.destroy(squatter);
    device.destroy(output);
    device.destroy(root);
}
