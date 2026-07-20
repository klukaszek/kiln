use kiln_rhi::{
    AccelerationStructure, BlasDesc, BlasMeshDesc, BuildAccelFlags, Device, GeometryFlags,
    GeometryType, GpuAddress, GpuAllocation, MemoryType, TlasDesc, TlasInstance,
};

use crate::scene::{Scene, Vertex};

/// Ray-tracing acceleration over the scene. The instance buffer and BLAS stay
/// alive because the TLAS references their GPU memory.
pub struct SceneAccel {
    instance_buffer: GpuAllocation,
    blas: AccelerationStructure,
    pub tlas: AccelerationStructure,
}

impl SceneAccel {
    pub(super) fn build(
        device: &Device,
        scene: &Scene,
        vertices: &GpuAllocation,
    ) -> anyhow::Result<Self> {
        let blas_desc = BlasDesc {
            meshes: vec![BlasMeshDesc {
                geometry_type: GeometryType::Triangles,
                flags: GeometryFlags::OPAQUE,
                vertex_buffer: vertices.gpu(),
                vertex_stride: std::mem::size_of::<Vertex>() as u64,
                vertex_count: u32::try_from(scene.vertices.len())?,
                index_buffer: GpuAddress(0),
                index_count: 0,
                aabb_buffer: GpuAddress(0),
                aabb_count: 0,
            }],
            flags: BuildAccelFlags::PREFER_FAST_TRACE,
        };
        let blas = device.create_blas(&blas_desc)?;
        build_blas(device, &blas, &blas_desc)?;

        let instance_buffer = device.malloc(
            u64::try_from(device.tlas_instance_stride())?,
            MemoryType::Default,
        )?;
        let tlas = match build_tlas(device, &blas, &instance_buffer) {
            Ok(tlas) => tlas,
            Err(error) => {
                device.free(instance_buffer);
                return Err(error);
            }
        };

        Ok(Self {
            instance_buffer,
            blas,
            tlas,
        })
    }

    pub(super) fn destroy(self, device: &Device) {
        let Self {
            instance_buffer,
            blas,
            tlas,
        } = self;
        drop(tlas);
        drop(blas);
        device.free(instance_buffer);
    }
}

fn build_blas(
    device: &Device,
    blas: &AccelerationStructure,
    desc: &BlasDesc,
) -> anyhow::Result<()> {
    let mut cmd = device.create_command_buffer()?;
    cmd.build_blas(blas, desc);
    cmd.end();
    let queue = device.queue();
    queue.submit(cmd)?;
    queue.wait_idle();
    Ok(())
}

fn build_tlas(
    device: &Device,
    blas: &AccelerationStructure,
    instance_buffer: &GpuAllocation,
) -> anyhow::Result<AccelerationStructure> {
    device.write_tlas_instance(
        instance_buffer,
        0,
        &TlasInstance {
            transform: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            instance_custom_index_and_mask: 0xFF << 24,
            instance_sbt_offset_and_flags: 0,
            acceleration_structure_reference: blas.gpu(),
        },
    )?;

    let desc = TlasDesc {
        instance_buffer: instance_buffer.gpu(),
        instance_count: 1,
        flags: BuildAccelFlags::PREFER_FAST_TRACE,
    };
    let tlas = device.create_tlas(&desc)?;
    let mut cmd = device.create_command_buffer()?;
    cmd.build_tlas(&tlas, &desc);
    cmd.end();
    let queue = device.queue();
    queue.submit(cmd)?;
    queue.wait_idle();
    Ok(tlas)
}
