use kiln_rhi::{
    AccelerationStructure, Allocation, BlasDesc, BlasMeshDesc, BuildAccelFlags, Device,
    GeometryFlags, GeometryType, GpuPtr, MAX_FRAMES_IN_FLIGHT, MemoryType, TlasDesc, TlasInstance,
};

use crate::base::gpu::{GpuArray, GpuUploadBatch};
use crate::base::renderer as render;
use crate::base::scene::Scene;

/// Ray-tracing acceleration owned exclusively by the path-tracing backend.
///
/// Mesh geometry is kept in local space and each scene occurrence gets a TLAS instance. This is
/// important for editor transforms: changing an instance transform must not rebuild the scene's
/// BLAS geometry.
pub(super) struct SceneAccel {
    instance_buffers: Vec<Allocation>,
    blases: Vec<AccelerationStructure>,
    pub(super) tlas: AccelerationStructure,
    ray_vertices: Vec<GpuArray<[f32; 3]>>,
    ray_indices: Vec<GpuArray<u32>>,
    retired_tlas: [Vec<AccelerationStructure>; MAX_FRAMES_IN_FLIGHT],
}

impl SceneAccel {
    pub(super) fn build(device: &Device, scene: &Scene) -> render::Result<Self> {
        if scene.geometry.instances.is_empty() {
            return Err(render::Error::Capacity(
                "ray-traced scenes require at least one geometry instance",
            ));
        }
        if scene.geometry.instances.len() > 0x00FF_FFFF {
            return Err(render::Error::Capacity(
                "ray-traced scene has too many geometry instances",
            ));
        }

        let geometries = build_ray_geometry(scene);
        let mut uploads = GpuUploadBatch::new(device);
        for (vertices, indices) in &geometries {
            if vertices.is_empty() || indices.is_empty() {
                return Err(render::Error::Capacity(
                    "ray-traced instances must contain triangle geometry",
                ));
            }
            uploads.upload(vertices)?;
            uploads.upload(indices)?;
        }
        let allocations = uploads.finish_vec()?;
        let mut pending = allocations
            .into_iter()
            .map(Some)
            .collect::<Vec<Option<Allocation>>>();
        let mut blases = Vec::with_capacity(geometries.len());
        let mut ray_vertices = Vec::with_capacity(geometries.len());
        let mut ray_indices = Vec::with_capacity(geometries.len());

        for (instance_index, (vertices, indices)) in geometries.iter().enumerate() {
            let vertex_allocation = pending[instance_index * 2]
                .take()
                .expect("vertex upload allocation missing");
            let index_allocation = pending[instance_index * 2 + 1]
                .take()
                .expect("index upload allocation missing");
            let vertices_gpu = GpuArray::new(vertex_allocation, vertices.len());
            let indices_gpu = GpuArray::new(index_allocation, indices.len());
            let desc = BlasDesc {
                meshes: vec![BlasMeshDesc {
                    geometry_type: GeometryType::Triangles,
                    flags: GeometryFlags::OPAQUE,
                    vertex_buffer: vertices_gpu.gpu(),
                    vertex_stride: std::mem::size_of::<[f32; 3]>() as u64,
                    vertex_count: vertices_gpu.len(),
                    index_buffer: indices_gpu.gpu(),
                    index_count: indices_gpu.len(),
                    aabb_buffer: GpuPtr::NULL,
                    aabb_count: 0,
                }],
                flags: BuildAccelFlags::PREFER_FAST_TRACE,
            };
            let blas = match device.create_blas(&desc) {
                Ok(blas) => blas,
                Err(error) => {
                    vertices_gpu.destroy(device);
                    indices_gpu.destroy(device);
                    cleanup_partial(device, blases, ray_vertices, ray_indices, pending);
                    return Err(error.into());
                }
            };
            if let Err(error) = build_blas(device, &blas, &desc) {
                drop(blas);
                vertices_gpu.destroy(device);
                indices_gpu.destroy(device);
                cleanup_partial(device, blases, ray_vertices, ray_indices, pending);
                return Err(error);
            }
            blases.push(blas);
            ray_vertices.push(vertices_gpu);
            ray_indices.push(indices_gpu);
        }

        let instance_buffer_size =
            device.tlas_instance_stride() as u64 * scene.geometry.instances.len() as u64;
        let mut instance_buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            match device.allocate(instance_buffer_size, MemoryType::Default) {
                Ok(buffer) => instance_buffers.push(buffer),
                Err(error) => {
                    for buffer in instance_buffers {
                        device.free(buffer);
                    }
                    cleanup_partial(device, blases, ray_vertices, ray_indices, pending);
                    return Err(error.into());
                }
            }
        }
        let tlas = match build_tlas(device, &blases, scene, &instance_buffers[0], None) {
            Ok(tlas) => tlas,
            Err(error) => {
                for buffer in instance_buffers {
                    device.free(buffer);
                }
                cleanup_partial(device, blases, ray_vertices, ray_indices, pending);
                return Err(error);
            }
        };
        Ok(Self {
            instance_buffers,
            blases,
            tlas,
            ray_vertices,
            ray_indices,
            retired_tlas: std::array::from_fn(|_| Vec::new()),
        })
    }

    pub(super) fn begin_frame(&mut self, frame_slot: usize) {
        if let Some(retired) = self.retired_tlas.get_mut(frame_slot) {
            retired.clear();
        }
    }

    pub(super) fn update_transforms(
        &mut self,
        device: &Device,
        scene: &Scene,
        dirty_instances: &[usize],
        frame_slot: usize,
    ) -> render::Result<()> {
        if scene.geometry.instances.len() != self.blases.len() {
            return Err(render::Error::Unsupported(
                "changing geometry instance topology requires renderer preparation",
            ));
        }
        for &index in dirty_instances {
            if index >= self.blases.len() {
                return Err(render::Error::Capacity(
                    "scene instance update index is out of bounds",
                ));
            }
        }
        let Some(retired) = self.retired_tlas.get_mut(frame_slot) else {
            return Err(render::Error::Capacity("frame slot is out of bounds"));
        };
        let Some(instance_buffer) = self.instance_buffers.get(frame_slot) else {
            return Err(render::Error::Capacity("frame slot is out of bounds"));
        };
        let tlas = build_tlas(
            device,
            &self.blases,
            scene,
            instance_buffer,
            // Each slot owns a distinct instance buffer, so it must be populated completely before
            // its TLAS is built. This avoids writing transforms into a buffer still read by the
            // other in-flight frame.
            None,
        )?;
        let old_tlas = std::mem::replace(&mut self.tlas, tlas);
        retired.push(old_tlas);
        Ok(())
    }

    pub(super) fn destroy(self, device: &Device) {
        let Self {
            instance_buffers,
            blases,
            tlas,
            ray_vertices,
            ray_indices,
            retired_tlas,
        } = self;
        drop(tlas);
        drop(retired_tlas);
        drop(blases);
        for instance_buffer in instance_buffers {
            device.free(instance_buffer);
        }
        for vertices in ray_vertices {
            vertices.destroy(device);
        }
        for indices in ray_indices {
            indices.destroy(device);
        }
    }
}

fn cleanup_partial(
    device: &Device,
    blases: Vec<AccelerationStructure>,
    ray_vertices: Vec<GpuArray<[f32; 3]>>,
    ray_indices: Vec<GpuArray<u32>>,
    pending: Vec<Option<Allocation>>,
) {
    drop(blases);
    for vertices in ray_vertices {
        vertices.destroy(device);
    }
    for indices in ray_indices {
        indices.destroy(device);
    }
    for allocation in pending.into_iter().flatten() {
        device.free(allocation);
    }
}

fn build_ray_geometry(scene: &Scene) -> Vec<(Vec<[f32; 3]>, Vec<u32>)> {
    scene
        .geometry
        .instances
        .iter()
        .map(|instance| {
            let mesh = &scene.geometry.meshes[instance.mesh.0];
            let mut vertices = Vec::with_capacity(mesh.indices.len());
            let mut indices = Vec::with_capacity(mesh.indices.len());
            for primitive in &mesh.primitives {
                let start = primitive.index_start;
                let end = start + primitive.index_count;
                for corners in mesh.indices[start..end].chunks_exact(3) {
                    let base = vertices.len() as u32;
                    vertices.extend(
                        corners
                            .iter()
                            .map(|&index| mesh.vertices[index as usize].position.to_array()),
                    );
                    indices.extend([base, base + 1, base + 2]);
                }
            }
            (vertices, indices)
        })
        .collect()
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
    blases: &[AccelerationStructure],
    scene: &Scene,
    instance_buffer: &Allocation,
    dirty_instances: Option<&[usize]>,
) -> render::Result<AccelerationStructure> {
    let write_instance = |index: usize| -> render::Result<()> {
        let instance = &scene.geometry.instances[index];
        device.write_tlas_instance(
            instance_buffer,
            index,
            &TlasInstance {
                transform: transform_rows(instance.transform),
                instance_custom_index_and_mask: (index as u32) | (0xFF << 24),
                instance_sbt_offset_and_flags: 0,
                acceleration_structure_reference: blases[index].gpu(),
            },
        )?;
        Ok(())
    };
    if let Some(dirty_instances) = dirty_instances {
        for &index in dirty_instances {
            write_instance(index)?;
        }
    } else {
        for index in 0..scene.geometry.instances.len() {
            write_instance(index)?;
        }
    }
    let desc = TlasDesc {
        instance_buffer: instance_buffer.ptr(),
        instance_count: scene.geometry.instances.len() as u32,
        flags: BuildAccelFlags::PREFER_FAST_TRACE,
    };
    let tlas = device.create_tlas(&desc)?;
    let mut cmd = device.create_command_buffer()?;
    cmd.build_tlas(&tlas, &desc);
    cmd.end();
    device.queue().submit(cmd)?;
    Ok(tlas)
}

fn transform_rows(transform: glam::DMat4) -> [[f32; 4]; 3] {
    [
        [
            transform.x_axis.x as f32,
            transform.y_axis.x as f32,
            transform.z_axis.x as f32,
            transform.w_axis.x as f32,
        ],
        [
            transform.x_axis.y as f32,
            transform.y_axis.y as f32,
            transform.z_axis.y as f32,
            transform.w_axis.y as f32,
        ],
        [
            transform.x_axis.z as f32,
            transform.y_axis.z as f32,
            transform.z_axis.z as f32,
            transform.w_axis.z as f32,
        ],
    ]
}
