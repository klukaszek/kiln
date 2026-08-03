//! Renderer-independent scene data.
//!
//! This module deliberately contains no GPU representation and no USD reader types. Importers
//! produce this model; renderers prepare the representation their own pipelines require.

mod geometry;
mod material;
mod mesh;

pub use geometry::{Geometry, Triangle};
pub use material::{Emission, Material, PrincipledBsdf, Surface};
pub use mesh::{Instance, Mesh, MeshId, Primitive, Vertex};

use glam::{DMat4, DVec3, Vec3};

/// Importer-independent perspective projection parameters.
#[derive(Clone, Copy, Debug)]
pub struct Projection {
    pub vertical_fov_rad: f32,
    pub clipping_range: [f32; 2],
}

/// Camera transform and projection in the scene IR.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub world: DMat4,
    pub projection: Projection,
}

impl Camera {
    pub fn position(&self) -> Vec3 {
        self.world.w_axis.truncate().as_vec3()
    }
}

/// Stable identifier for a material referenced by scene geometry.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaterialId(pub usize);

impl MaterialId {
    pub const DEFAULT: Self = Self(0);
}

/// CPU scene IR shared by all renderer backends.
pub struct Scene {
    pub geometry: Geometry,
    pub materials: Vec<Material>,
    pub camera: Camera,
    /// World up axis from stage metadata. Geometry and cameras are already in world space.
    pub up: DVec3,
}

impl Scene {
    pub fn triangle_count(&self) -> usize {
        self.geometry.triangles.len()
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn loads_bundled_cornell_box() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/cornell-box.usda");
        let scene = crate::usd::load(&path).unwrap();

        assert!(scene.triangle_count() > 0);
        assert!(!scene.materials.is_empty());
        assert!(!scene.geometry.meshes.is_empty());
        assert!(!scene.geometry.meshes[0].indices.is_empty());
    }

    #[test]
    fn scene_can_be_built_without_an_importer() {
        let material_id = MaterialId::DEFAULT;
        let vertices = vec![
            Vertex {
                position: glam::Vec3::new(0.0, 0.0, 0.0),
                normal: Some(glam::Vec3::Z),
                uv: Some(glam::Vec2::new(0.0, 0.0)),
            },
            Vertex {
                position: glam::Vec3::new(1.0, 0.0, 0.0),
                normal: Some(glam::Vec3::Z),
                uv: Some(glam::Vec2::new(1.0, 0.0)),
            },
            Vertex {
                position: glam::Vec3::new(0.0, 1.0, 0.0),
                normal: Some(glam::Vec3::Z),
                uv: Some(glam::Vec2::new(0.0, 1.0)),
            },
        ];
        let mesh = Mesh::with_material(vertices, vec![0, 1, 2], material_id);
        let geometry = Geometry::new(
            vec![mesh],
            vec![Instance {
                mesh: MeshId(0),
                transform: glam::DMat4::IDENTITY,
            }],
        );
        let scene = Scene {
            geometry,
            materials: vec![Material::default()],
            camera: Camera {
                world: glam::DMat4::IDENTITY,
                projection: Projection {
                    vertical_fov_rad: 1.0,
                    clipping_range: [0.1, 100.0],
                },
            },
            up: glam::DVec3::Y,
        };
        assert_eq!(scene.triangle_count(), 1);
    }
}
