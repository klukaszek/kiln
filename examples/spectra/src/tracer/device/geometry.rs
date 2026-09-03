//! Scene triangles and instance transforms, as the trace kernel indexes them.
//!
//! Triangles are stored mesh-local and instanced through the TLAS, so moving an object rewrites
//! one transform rather than its geometry.

use glam::{DMat4, Vec4};

use crate::scene::{Mesh, Scene, Triangle};

use super::records::{GpuInstance, GpuTriangle};

#[derive(Clone, Copy, Debug)]
pub(super) struct InstanceLayout {
    triangle_start: usize,
}

pub(super) fn build_gpu_triangles(scene: &Scene, emission: &[f32]) -> Vec<GpuTriangle> {
    let mut triangles = Vec::with_capacity(scene.triangle_count());
    for instance in &scene.geometry.instances {
        let mesh = &scene.geometry.meshes[instance.mesh.0];
        for primitive in &mesh.primitives {
            let indices =
                &mesh.indices[primitive.index_start..primitive.index_start + primitive.index_count];
            for corners in indices.chunks_exact(3) {
                triangles.push(gpu_triangle(
                    Triangle {
                        vertices: [
                            mesh.vertices[corners[0] as usize],
                            mesh.vertices[corners[1] as usize],
                            mesh.vertices[corners[2] as usize],
                        ],
                        material: primitive.material,
                    },
                    emission,
                ));
            }
        }
    }
    triangles
}

fn gpu_triangle(triangle: Triangle, emission: &[f32]) -> GpuTriangle {
    let material = triangle.material;
    let (normal, area) = triangle.geometric_normal_and_area();
    let uv0 = triangle.vertices[0].uv.unwrap_or_default();
    let uv1 = triangle.vertices[1].uv.unwrap_or_default();
    let uv2 = triangle.vertices[2].uv.unwrap_or_default();
    GpuTriangle {
        normal_area: normal.extend(area),
        uv01: Vec4::new(uv0.x, uv0.y, uv1.x, uv1.y),
        uv2,
        material_id: material.0 as u32,
        emission: emission.get(material.0).copied().unwrap_or(0.0),
    }
}

pub(super) fn world_triangle_area(transform: DMat4, mesh: &Mesh, corners: [u32; 3]) -> f32 {
    let vertices = corners
        .map(|index| transform.transform_point3(mesh.vertices[index as usize].position.as_dvec3()));
    let cross = (vertices[1] - vertices[0]).cross(vertices[2] - vertices[0]);
    (0.5 * cross.length()) as f32
}

pub(super) fn build_instance_layout(scene: &Scene) -> Vec<InstanceLayout> {
    let mut triangle_base = 0;
    scene
        .geometry
        .instances
        .iter()
        .map(|instance| {
            let mesh = &scene.geometry.meshes[instance.mesh.0];
            let triangle_count = mesh
                .primitives
                .iter()
                .map(|primitive| primitive.index_count / 3)
                .sum::<usize>();
            let layout = InstanceLayout {
                triangle_start: triangle_base,
            };
            triangle_base += triangle_count;
            layout
        })
        .collect()
}

pub(super) fn build_gpu_instances(scene: &Scene, layout: &[InstanceLayout]) -> Vec<GpuInstance> {
    layout
        .iter()
        .zip(&scene.geometry.instances)
        .map(|(layout, instance)| build_gpu_instance(*layout, instance.transform))
        .collect()
}

fn build_gpu_instance(layout: InstanceLayout, transform: glam::DMat4) -> GpuInstance {
    let normal = transform.inverse().transpose();
    GpuInstance {
        triangle_base: layout.triangle_start as u32,
        _pad: 0,
        _pad2: 0,
        _pad3: 0,
        normal_x: normal.x_axis.truncate().as_vec3().extend(0.0),
        normal_y: normal.y_axis.truncate().as_vec3().extend(0.0),
        normal_z_determinant: normal
            .z_axis
            .truncate()
            .as_vec3()
            .extend(transform.determinant() as f32),
    }
}
