use kiln_rhi::{
    AccelerationStructure, Allocation, BlasDesc, BlasMeshDesc, BuildAccelFlags, Device,
    GeometryFlags, GeometryType, GpuPtr, MemoryType, TlasDesc, TlasInstance,
};

use crate::render;
use crate::scene::Scene;
use crate::tracer::upload::{GpuArray, GpuUploadBatch};

/// Ray-tracing acceleration owned exclusively by the path-tracing backend.
///
/// Mesh geometry is kept in local space and each scene occurrence gets a TLAS instance. This is
/// important for editor transforms: changing an instance transform must not rebuild the scene's
/// BLAS geometry.
pub(crate) struct SceneAccel {
    instance_buffer: Allocation,
    blases: Vec<AccelerationStructure>,
    pub(crate) tlas: AccelerationStructure,
    ray_vertices: Vec<GpuArray<[f32; 3]>>,
    ray_indices: Vec<GpuArray<u32>>,
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
        if geometries
            .iter()
            .any(|(vertices, indices)| vertices.is_empty() || indices.is_empty())
        {
            return Err(render::Error::Capacity(
                "ray-traced instances must contain triangle geometry",
            ));
        }

        // Everything built so far lives in `partial`, so any failure unwinds through one path.
        let mut partial = Partial::default();
        match build_into(device, scene, &geometries, &mut partial) {
            Ok(built) => Ok(built),
            Err(error) => {
                partial.destroy(device);
                Err(error)
            }
        }
    }

    /// Rebuild the TLAS after instance transforms moved. BLAS geometry is in local space, so it
    /// is reused untouched. The caller has already drained the queue, so the old TLAS is freed
    /// immediately rather than retired against a frame slot.
    pub(super) fn rebuild_tlas(&mut self, device: &Device, scene: &Scene) -> render::Result<()> {
        if scene.geometry.instances.len() != self.blases.len() {
            return Err(render::Error::Unsupported(
                "changing geometry instance topology requires renderer preparation",
            ));
        }
        let tlas = build_tlas(device, &self.blases, scene, &self.instance_buffer)?;
        self.tlas = tlas;
        Ok(())
    }

    pub(super) fn destroy(self, device: &Device) {
        let Self {
            instance_buffer,
            blases,
            tlas,
            ray_vertices,
            ray_indices,
        } = self;
        drop(tlas);
        drop(blases);
        device.destroy(instance_buffer);
        for vertices in ray_vertices {
            vertices.destroy(device);
        }
        for indices in ray_indices {
            indices.destroy(device);
        }
    }
}

/// Resources owned during [`SceneAccel::build`], before the finished structure exists.
#[derive(Default)]
struct Partial {
    blases: Vec<AccelerationStructure>,
    ray_vertices: Vec<GpuArray<[f32; 3]>>,
    ray_indices: Vec<GpuArray<u32>>,
    instance_buffer: Option<Allocation>,
}

impl Partial {
    fn destroy(self, device: &Device) {
        drop(self.blases);
        for vertices in self.ray_vertices {
            vertices.destroy(device);
        }
        for indices in self.ray_indices {
            indices.destroy(device);
        }
        if let Some(instance_buffer) = self.instance_buffer {
            device.destroy(instance_buffer);
        }
    }
}

/// Build one BLAS per instance, then the TLAS over them. Each resource is moved into `partial` as
/// soon as it exists, so the caller's cleanup covers whatever was reached.
fn build_into(
    device: &Device,
    scene: &Scene,
    geometries: &[(Vec<[f32; 3]>, Vec<u32>)],
    partial: &mut Partial,
) -> render::Result<SceneAccel> {
    let mut uploads = GpuUploadBatch::new(device)?;
    for (vertices, indices) in geometries {
        partial.ray_vertices.push(uploads.upload(vertices)?);
        partial.ray_indices.push(uploads.upload(indices)?);
    }
    uploads.submit()?;

    let mut blas_descs = Vec::with_capacity(geometries.len());
    for (vertices, indices) in partial.ray_vertices.iter().zip(&partial.ray_indices) {
        let desc = BlasDesc {
            meshes: vec![BlasMeshDesc {
                geometry_type: GeometryType::Triangles,
                flags: GeometryFlags::OPAQUE,
                vertex_buffer: vertices.gpu(),
                vertex_stride: std::mem::size_of::<[f32; 3]>() as u64,
                vertex_count: vertices.len(),
                index_buffer: indices.gpu(),
                index_count: indices.len(),
                aabb_buffer: GpuPtr::NULL,
                aabb_count: 0,
            }],
            flags: BuildAccelFlags::PREFER_FAST_TRACE,
        };
        blas_descs.push(desc);
    }

    for desc in &blas_descs {
        let blas = device.create_blas(desc)?;
        partial.blases.push(blas);
        build_blas(
            device,
            partial.blases.last().expect("just pushed the blas"),
            desc,
        )?;
    }

    let instance_buffer_size =
        device.tlas_instance_stride() as u64 * scene.geometry.instances.len() as u64;
    partial.instance_buffer = Some(device.allocate(instance_buffer_size, MemoryType::Upload)?);
    let instance_buffer = partial
        .instance_buffer
        .as_ref()
        .expect("just set the instance buffer");
    let tlas = build_tlas(device, &partial.blases, scene, instance_buffer)?;

    Ok(SceneAccel {
        instance_buffer: partial
            .instance_buffer
            .take()
            .expect("just set the instance buffer"),
        blases: std::mem::take(&mut partial.blases),
        tlas,
        ray_vertices: std::mem::take(&mut partial.ray_vertices),
        ray_indices: std::mem::take(&mut partial.ray_indices),
    })
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
) -> render::Result<AccelerationStructure> {
    for (index, instance) in scene.geometry.instances.iter().enumerate() {
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
    }
    let desc = TlasDesc {
        instance_buffer: instance_buffer.gpu().cast(),
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
