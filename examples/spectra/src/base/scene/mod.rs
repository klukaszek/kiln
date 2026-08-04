//! Renderer-independent scene data and generic backend storage.
//!
//! Importers produce [`Scene<CpuStorage>`]. Renderers prepare the same scene into
//! [`Scene<S>`], where `S` owns that renderer's device representation.

use std::convert::Infallible;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use kiln_rhi::Device;

mod geometry;
mod material;
mod mesh;

pub use geometry::{Geometry, Triangle};
pub use material::{
    ColorSpace, Emission, Image, ImageId, Material, PrincipledBsdf, Surface, Texture, TextureId,
    WrapMode,
};
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

/// The canonical scene payload shared by all renderer backends.
#[derive(Clone)]
pub struct SceneData {
    pub geometry: Geometry,
    pub materials: Vec<Material>,
    pub images: Vec<Image>,
    pub textures: Vec<Texture>,
    pub camera: Camera,
    /// World up axis from stage metadata. Geometry and cameras are already in world space.
    pub up: DVec3,
}

/// Marker storage for an imported, CPU-only scene.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuStorage;

/// Renderer-owned scene storage.
pub trait SceneStorage: Sized {
    type Config;
    type Error;

    fn build(
        device: &Device,
        source: &Scene<CpuStorage>,
        config: &Self::Config,
    ) -> Result<Self, Self::Error>;

    fn destroy(self, device: &Device);
}

impl SceneStorage for CpuStorage {
    type Config = ();
    type Error = Infallible;

    fn build(
        _device: &Device,
        _source: &Scene<CpuStorage>,
        _config: &Self::Config,
    ) -> Result<Self, Self::Error> {
        Ok(Self)
    }

    fn destroy(self, _device: &Device) {}
}

/// One scene type whose storage is specialized for the renderer consuming it.
pub struct Scene<S: SceneStorage = CpuStorage> {
    data: Arc<SceneData>,
    pub storage: S,
}

impl Scene<CpuStorage> {
    pub fn new(data: SceneData) -> Self {
        Self {
            data: Arc::new(data),
            storage: CpuStorage,
        }
    }

    pub fn prepare<S: SceneStorage>(
        &self,
        device: &Device,
        config: &S::Config,
    ) -> Result<Scene<S>, S::Error> {
        Ok(Scene {
            data: Arc::clone(&self.data),
            storage: S::build(device, self, config)?,
        })
    }
}

impl<S: SceneStorage> Scene<S> {
    pub fn destroy(self, device: &Device) {
        self.storage.destroy(device);
    }
}

impl<S: SceneStorage> Deref for Scene<S> {
    type Target = SceneData;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<S: SceneStorage> DerefMut for Scene<S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.data)
    }
}

impl SceneData {
    pub fn triangle_count(&self) -> usize {
        self.geometry.triangle_count()
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn loads_bundled_cornell_box() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/cornell-box.usda");
        let scene = crate::importers::usd::load(&path).unwrap();

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
        let scene = Scene::new(SceneData {
            geometry,
            materials: vec![Material::default()],
            images: Vec::new(),
            textures: Vec::new(),
            camera: Camera {
                world: glam::DMat4::IDENTITY,
                projection: Projection {
                    vertical_fov_rad: 1.0,
                    clipping_range: [0.1, 100.0],
                },
            },
            up: glam::DVec3::Y,
        });
        assert_eq!(scene.triangle_count(), 1);
    }
}
