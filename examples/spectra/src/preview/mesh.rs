use glam::Vec4;
use kiln_rhi::{Device, gpu_struct};

use crate::render::{self, Error, GpuArray, GpuUploadBatch};
use crate::scene::Scene;

gpu_struct! {
    pub(super) struct RasterVertex {
        pos: Vec4,
        normal: Vec4,
        color: Vec4,
    }
}

pub(super) struct PreviewMesh {
    pub(super) vertices: GpuArray<RasterVertex>,
    pub(super) triangle_count: u32,
}

impl PreviewMesh {
    pub(super) fn build(device: &Device, scene: &Scene) -> render::Result<Self> {
        if scene.triangle_count() > (u32::MAX as usize) / 3 {
            return Err(Error::Capacity("preview geometry exceeds u32 indexing"));
        }
        let mut vertices = Vec::with_capacity(scene.triangle_count() * 3);
        for triangle in &scene.geometry.triangles {
            let material = &scene.materials[triangle.material.0];
            let color = material.preview_color();
            for vertex in &triangle.vertices {
                vertices.push(RasterVertex {
                    pos: vertex.position.extend(1.0),
                    normal: vertex.normal.unwrap_or(glam::Vec3::Y).extend(0.0),
                    color: color.extend(1.0),
                });
            }
        }
        let triangle_count = scene.triangle_count() as u32;
        let mut uploads = GpuUploadBatch::new(device);
        uploads.upload(&vertices)?;
        let [vertices_gpu] = uploads.finish()?;
        Ok(Self {
            vertices: GpuArray::new(vertices_gpu, vertices.len()),
            triangle_count,
        })
    }

    pub(super) fn destroy(self, device: &Device) {
        self.vertices.destroy(device);
    }
}
