use glam::{DMat4, Vec2, Vec3};

use super::MaterialId;

/// Vertex attributes retained by the CPU scene IR.
#[derive(Clone, Copy, Debug, Default)]
pub struct Vertex {
    pub position: Vec3,
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

/// An indexed mesh in asset/local space. Attributes are intentionally optional.
#[derive(Clone)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub primitives: Vec<Primitive>,
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
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MeshId(pub usize);

/// An occurrence of a mesh in world space.
#[derive(Clone, Copy, Debug)]
pub struct Instance {
    pub mesh: MeshId,
    pub transform: DMat4,
}
