use kiln_rhi::{
    AccelerationStructure, BlasDesc, BlasMeshDesc, BuildAccelFlags, Device, GeometryFlags,
    GeometryType, GpuAddress, GpuAllocation, MemoryType, TlasDesc, TlasInstance,
};

use crate::base::gpu::{GpuArray, GpuUploadBatch};
use crate::base::renderer as render;
use crate::base::scene::{CpuStorage, Scene};

/// Ray-tracing acceleration owned exclusively by the path-tracing backend.
pub(super) struct SceneAccel {
    instance_buffer: GpuAllocation,
    blas: AccelerationStructure,
    pub(super) tlas: AccelerationStructure,
    ray_vertices: GpuArray<[f32; 3]>,
    ray_indices: GpuArray<u32>,
}

impl SceneAccel {
    pub(super) fn build(device: &Device, scene: &Scene<CpuStorage>) -> render::Result<Self> {
        let (vertices, indices) = build_ray_geometry(scene);
        let mut uploads = GpuUploadBatch::new(device);
        uploads.upload(&vertices)?;
        uploads.upload(&indices)?;
        let [vertex_buffer, index_buffer] = uploads.finish()?;
        let ray_vertices = GpuArray::new(vertex_buffer, vertices.len());
        let ray_indices = GpuArray::new(index_buffer, indices.len());
        let desc = BlasDesc {
            meshes: vec![BlasMeshDesc {
                geometry_type: GeometryType::Triangles,
                flags: GeometryFlags::OPAQUE,
                vertex_buffer: ray_vertices.gpu(),
                vertex_stride: std::mem::size_of::<[f32; 3]>() as u64,
                vertex_count: ray_vertices.len(),
                index_buffer: ray_indices.gpu(),
                index_count: ray_indices.len(),
                aabb_buffer: GpuAddress(0),
                aabb_count: 0,
            }],
            flags: BuildAccelFlags::PREFER_FAST_TRACE,
        };
        let blas = match device.create_blas(&desc) {
            Ok(blas) => blas,
            Err(error) => {
                ray_vertices.destroy(device);
                ray_indices.destroy(device);
                return Err(error.into());
            }
        };
        if let Err(error) = build_blas(device, &blas, &desc) {
            drop(blas);
            ray_vertices.destroy(device);
            ray_indices.destroy(device);
            return Err(error);
        }
        let instance_buffer =
            match device.malloc(device.tlas_instance_stride() as u64, MemoryType::Default) {
                Ok(value) => value,
                Err(error) => {
                    drop(blas);
                    ray_vertices.destroy(device);
                    ray_indices.destroy(device);
                    return Err(error.into());
                }
            };
        let tlas = match build_tlas(device, &blas, &instance_buffer) {
            Ok(value) => value,
            Err(error) => {
                device.free(instance_buffer);
                drop(blas);
                ray_vertices.destroy(device);
                ray_indices.destroy(device);
                return Err(error);
            }
        };
        Ok(Self {
            instance_buffer,
            blas,
            tlas,
            ray_vertices,
            ray_indices,
        })
    }

    pub(super) fn destroy(self, device: &Device) {
        let Self {
            instance_buffer,
            blas,
            tlas,
            ray_vertices,
            ray_indices,
        } = self;
        drop(tlas);
        drop(blas);
        device.free(instance_buffer);
        ray_vertices.destroy(device);
        ray_indices.destroy(device);
    }
}

fn build_ray_geometry(scene: &Scene) -> (Vec<[f32; 3]>, Vec<u32>) {
    let triangles = scene.geometry.world_triangles();
    let vertex_count = triangles.len() * 3;
    let mut vertices = Vec::with_capacity(vertex_count);
    let mut indices = Vec::with_capacity(vertex_count);
    for triangle in &triangles {
        for vertex in &triangle.vertices {
            indices.push(vertices.len() as u32);
            vertices.push(vertex.position.to_array());
        }
    }
    (vertices, indices)
}

fn build_blas(
    device: &Device,
    blas: &AccelerationStructure,
    desc: &BlasDesc,
) -> render::Result<()> {
    let mut cmd = device.create_command_buffer()?;
    cmd.build_blas(blas, desc);
    cmd.end();
    device.queue().submit(cmd)?;
    device.queue().wait_idle();
    Ok(())
}

fn build_tlas(
    device: &Device,
    blas: &AccelerationStructure,
    instance_buffer: &GpuAllocation,
) -> render::Result<AccelerationStructure> {
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
    device.queue().submit(cmd)?;
    device.queue().wait_idle();
    Ok(tlas)
}
