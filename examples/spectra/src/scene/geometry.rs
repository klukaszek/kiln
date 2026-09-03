//! Instanced geometry and derived world-space views.

use glam::{DMat4, DVec3, Vec3};

use super::{Instance, MaterialId, Mesh, MeshId, Primitive, Vertex};

/// A world-space triangle in the derived backend-friendly view.
#[derive(Clone, Copy, Debug)]
pub struct Triangle {
    pub vertices: [Vertex; 3],
    pub material: MaterialId,
}

impl Triangle {
    pub fn geometric_normal_and_area(&self) -> (Vec3, f32) {
        let p0 = self.vertices[0].position;
        let cross = (self.vertices[1].position - p0).cross(self.vertices[2].position - p0);
        let twice_area = cross.length();
        (cross.normalize(), 0.5 * twice_area)
    }
}

/// Indexed scene geometry. Backends can request a derived world-space triangle view when their
/// representation requires flattened geometry, while indexed meshes and instances remain the
/// canonical scene data.
#[derive(Clone)]
pub struct Geometry {
    pub meshes: Vec<Mesh>,
    pub instances: Vec<Instance>,
}

impl Geometry {
    pub fn new(meshes: Vec<Mesh>, instances: Vec<Instance>) -> Self {
        Self { meshes, instances }
    }

    pub fn triangle_count(&self) -> usize {
        self.instances
            .iter()
            .map(|instance| {
                self.meshes[instance.mesh.0]
                    .primitives
                    .iter()
                    .map(|primitive| primitive.index_count / 3)
                    .sum::<usize>()
            })
            .sum()
    }

    /// World-space bounds of every instance, or `None` when the scene has no vertices.
    pub fn world_bounds(&self) -> Option<(DVec3, DVec3)> {
        let mut min = DVec3::splat(f64::INFINITY);
        let mut max = DVec3::splat(f64::NEG_INFINITY);
        for instance in &self.instances {
            let Some(mesh) = self.meshes.get(instance.mesh.0) else {
                continue;
            };
            for vertex in &mesh.vertices {
                let world = instance
                    .transform
                    .transform_point3(vertex.position.as_dvec3());
                min = min.min(world);
                max = max.max(world);
            }
        }
        (min.cmple(max).all()).then_some((min, max))
    }

    pub fn world_triangles(&self) -> Vec<Triangle> {
        let mut triangles = Vec::new();
        for instance in &self.instances {
            triangles.extend(flatten_instance_triangles(&self.meshes, instance));
        }
        triangles
    }

    /// Return only the triangles belonging to one instance. Renderer update paths use this to
    /// patch an edited instance without flattening every occurrence in the scene.
    pub fn instance_triangles(&self, instance_index: usize) -> Vec<Triangle> {
        let instance = &self.instances[instance_index];
        flatten_instance_triangles(&self.meshes, instance)
    }

    pub fn from_triangles(triangles: Vec<Triangle>) -> Self {
        let capacity = triangles.len() * 3;
        let mut vertices = Vec::with_capacity(capacity);
        let mut indices = Vec::with_capacity(capacity);
        let mut primitives = Vec::with_capacity(triangles.len());
        for triangle in &triangles {
            let start = indices.len();
            let base = vertices.len() as u32;
            vertices.extend(triangle.vertices);
            indices.extend([base, base + 1, base + 2]);
            primitives.push(Primitive {
                index_start: start,
                index_count: 3,
                material: triangle.material,
            });
        }
        let mesh = Mesh {
            vertices,
            indices,
            primitives,
            emissive_components: Vec::new(),
        };
        Self::new(
            vec![mesh],
            vec![Instance {
                mesh: MeshId(0),
                path: "/Geometry".into(),
                transform: DMat4::IDENTITY,
            }],
        )
    }
}

fn flatten_instance_triangles(meshes: &[Mesh], instance: &Instance) -> Vec<Triangle> {
    let mesh = &meshes[instance.mesh.0];
    let transform = instance.transform;
    let normal_transform = transform.inverse().transpose();
    let mut triangles = Vec::new();
    for primitive in &mesh.primitives {
        let start = primitive.index_start;
        let indices = &mesh.indices[start..start + primitive.index_count];
        for corners in indices.chunks_exact(3) {
            let mut vertices = [Vertex::default(); 3];
            for (dst, index) in vertices.iter_mut().zip(corners) {
                let source = mesh.vertices[*index as usize];
                dst.position = transform
                    .transform_point3(source.position.as_dvec3())
                    .as_vec3();
                dst.source_index = source.source_index;
                dst.normal = source.normal.map(|normal| {
                    normal_transform
                        .transform_vector3(normal.as_dvec3())
                        .normalize()
                        .as_vec3()
                });
                dst.uv = source.uv;
            }
            let geometric = (vertices[1].position - vertices[0].position)
                .cross(vertices[2].position - vertices[0].position)
                .normalize();
            for vertex in &mut vertices {
                if vertex.normal.is_none() {
                    vertex.normal = Some(geometric);
                }
            }
            triangles.push(Triangle {
                vertices,
                material: primitive.material,
            });
        }
    }
    triangles
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_vertices() -> Vec<Vertex> {
        vec![
            Vertex {
                position: Vec3::ZERO,
                source_index: None,
                normal: None,
                uv: None,
            },
            Vertex {
                position: Vec3::X,
                source_index: None,
                normal: None,
                uv: None,
            },
            Vertex {
                position: Vec3::Y,
                source_index: None,
                normal: None,
                uv: None,
            },
        ]
    }

    #[test]
    fn instances_are_flattened_into_world_space() {
        let mesh = Mesh::with_material(triangle_vertices(), vec![0, 1, 2], MaterialId::DEFAULT);
        let geometry = Geometry::new(
            vec![mesh],
            vec![Instance {
                mesh: MeshId(0),
                path: "/Geometry".into(),
                transform: DMat4::from_translation(glam::DVec3::Z),
            }],
        );
        assert_eq!(geometry.world_triangles()[0].vertices[0].position, Vec3::Z);
    }
}
