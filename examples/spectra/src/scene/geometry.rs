//! Instanced geometry and its derived world-space triangle view.

use glam::{DMat4, Vec3};

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

/// Indexed scene geometry. `triangles` is a derived world-space view for backends that flatten
/// the IR; the indexed meshes and instances remain available to renderers that can consume them.
pub struct Geometry {
    pub meshes: Vec<Mesh>,
    pub instances: Vec<Instance>,
    pub triangles: Vec<Triangle>,
}

impl Geometry {
    pub fn new(meshes: Vec<Mesh>, instances: Vec<Instance>) -> Self {
        let triangles = flatten_triangles(&meshes, &instances);
        Self {
            meshes,
            instances,
            triangles,
        }
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
        };
        Self::new(
            vec![mesh],
            vec![Instance {
                mesh: MeshId(0),
                transform: DMat4::IDENTITY,
            }],
        )
    }
}

fn flatten_triangles(meshes: &[Mesh], instances: &[Instance]) -> Vec<Triangle> {
    let mut triangles = Vec::new();
    for instance in instances {
        let mesh = &meshes[instance.mesh.0];
        let transform = instance.transform;
        let normal_transform = transform.inverse().transpose();
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
                normal: None,
                uv: None,
            },
            Vertex {
                position: Vec3::X,
                normal: None,
                uv: None,
            },
            Vertex {
                position: Vec3::Y,
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
                transform: DMat4::from_translation(glam::DVec3::Z),
            }],
        );
        assert_eq!(geometry.triangles[0].vertices[0].position, Vec3::Z);
    }
}
