use std::collections::HashMap;

use glam::{DMat4, Vec2, Vec3};

use super::{Material, MaterialId};

/// Vertex attributes retained by the CPU scene IR.
#[derive(Clone, Copy, Debug, Default)]
pub struct Vertex {
    pub position: Vec3,
    /// Original indexed point identity when the importer can preserve it. This lets topology
    /// consumers join vertices split only by normals or UV seams.
    pub source_index: Option<u32>,
    pub normal: Option<Vec3>,
    pub uv: Option<Vec2>,
}

/// A contiguous indexed range with one material binding.
#[derive(Clone, Copy, Debug)]
pub struct Primitive {
    pub index_start: usize,
    pub index_count: usize,
    pub material: MaterialId,
}

/// A connected emissive region on a mesh. This is scene geometry metadata, not an analytic light.
#[derive(Clone, Debug)]
pub struct EmissiveComponent {
    pub material: MaterialId,
    pub triangles: Vec<[u32; 3]>,
}

/// An indexed mesh in asset/local space. Attributes are intentionally optional.
#[derive(Clone)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub primitives: Vec<Primitive>,
    pub emissive_components: Vec<EmissiveComponent>,
}

impl Mesh {
    pub fn with_material(
        vertices: Vec<Vertex>,
        indices: Vec<u32>,
        material_id: MaterialId,
    ) -> Self {
        let primitive = Primitive {
            index_start: 0,
            index_count: indices.len(),
            material: material_id,
        };
        Self {
            vertices,
            indices,
            primitives: vec![primitive],
            emissive_components: Vec::new(),
        }
    }

    pub fn refresh_emissive_components(&mut self, materials: &[Material]) {
        let mut triangles = Vec::new();
        for primitive in &self.primitives {
            if !materials
                .get(primitive.material.0)
                .is_some_and(Material::is_emissive)
            {
                continue;
            }
            let range = primitive.index_start..primitive.index_start + primitive.index_count;
            for corners in self.indices[range].chunks_exact(3) {
                let indices = [corners[0], corners[1], corners[2]];
                triangles.push((
                    primitive.material,
                    indices
                        .map(|index| self.vertices[index as usize].source_index.unwrap_or(index)),
                    indices,
                ));
            }
        }

        let mut sets = DisjointSet::new(triangles.len());
        let mut edges = HashMap::<(MaterialId, u32, u32), usize>::new();
        for (triangle_index, (material, topology, _)) in triangles.iter().enumerate() {
            for edge in [
                [topology[0], topology[1]],
                [topology[1], topology[2]],
                [topology[2], topology[0]],
            ] {
                let (a, b) = if edge[0] < edge[1] {
                    (edge[0], edge[1])
                } else {
                    (edge[1], edge[0])
                };
                if let Some(other) = edges.insert((*material, a, b), triangle_index) {
                    sets.union(triangle_index, other);
                }
            }
        }

        let mut grouped = HashMap::<(usize, MaterialId), Vec<[u32; 3]>>::new();
        for (triangle_index, (material, _, indices)) in triangles.into_iter().enumerate() {
            grouped
                .entry((sets.find(triangle_index), material))
                .or_default()
                .push(indices);
        }
        let mut components = grouped.into_iter().collect::<Vec<_>>();
        components.sort_by_key(|((root, _), _)| *root);
        self.emissive_components = components
            .into_iter()
            .map(|((_, material), triangles)| EmissiveComponent {
                material,
                triangles,
            })
            .collect();
    }
}

struct DisjointSet {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl DisjointSet {
    fn new(count: usize) -> Self {
        Self {
            parent: (0..count).collect(),
            size: vec![1; count],
        }
    }

    fn find(&mut self, value: usize) -> usize {
        if self.parent[value] != value {
            self.parent[value] = self.find(self.parent[value]);
        }
        self.parent[value]
    }

    fn union(&mut self, left: usize, right: usize) {
        let mut left = self.find(left);
        let mut right = self.find(right);
        if left == right {
            return;
        }
        if self.size[left] < self.size[right] {
            std::mem::swap(&mut left, &mut right);
        }
        self.parent[right] = left;
        self.size[left] += self.size[right];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_mesh() -> Mesh {
        Mesh::with_material(
            vec![
                Vertex {
                    position: Vec3::ZERO,
                    ..Vertex::default()
                },
                Vertex {
                    position: Vec3::X,
                    ..Vertex::default()
                },
                Vertex {
                    position: Vec3::Y,
                    ..Vertex::default()
                },
            ],
            vec![0, 1, 2],
            MaterialId::DEFAULT,
        )
    }

    #[test]
    fn emissive_component_cache_tracks_material_edits() {
        let mut mesh = triangle_mesh();
        let mut materials = vec![Material::default()];

        mesh.refresh_emissive_components(&materials);
        assert!(mesh.emissive_components.is_empty());

        materials[0].emission.color = Vec3::ONE;
        mesh.refresh_emissive_components(&materials);
        assert_eq!(mesh.emissive_components.len(), 1);
        assert_eq!(mesh.emissive_components[0].triangles.len(), 1);

        materials[0].emission.color = Vec3::ZERO;
        mesh.refresh_emissive_components(&materials);
        assert!(mesh.emissive_components.is_empty());
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MeshId(pub usize);

/// An occurrence of a mesh in world space.
#[derive(Clone, Debug)]
pub struct Instance {
    pub mesh: MeshId,
    pub path: String,
    pub transform: DMat4,
}
