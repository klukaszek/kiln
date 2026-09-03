//! Renderer-independent scene data.
//!
//! Importers produce [`Scene`]. The path tracer builds its own GPU storage from it.

use std::ops::{Deref, DerefMut};

mod geometry;
mod illuminant;
mod light;
mod material;
mod mesh;
mod node;

pub use geometry::{Geometry, Triangle};
pub use illuminant::{EmitterExtent, Illuminant, IntensityUnit, SpectrumSource};
pub use light::{Light, LightKind};
pub use material::{
    Channel, Channels, ColorSpace, Image, ImageId, Material, PrincipledBsdf, ScalarMap, Surface,
    Texture, TextureId, WrapMode, luminance,
};
pub use mesh::{EmissiveComponent, Instance, Mesh, MeshId, Primitive, Vertex};
pub use node::{NodeId, NodeObject, SceneNode, build_scene_nodes};

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
    pub lights: Vec<Light>,
    pub nodes: Vec<SceneNode>,
    pub materials: Vec<Material>,
    pub images: Vec<Image>,
    pub textures: Vec<Texture>,
    pub camera: Camera,
    /// World up axis from stage metadata. Geometry and cameras are already in world space.
    pub up: DVec3,
}

/// The editable CPU scene. Renderer-owned GPU resources are deliberately not stored here: a
/// renderer receives a snapshot during construction and owns its prepared storage separately.
pub struct Scene {
    data: SceneData,
}

impl Scene {
    pub fn new(data: SceneData) -> Self {
        Self { data }
    }

    pub fn refresh_emissive_components(&mut self) {
        let materials = self.materials.clone();
        for mesh in &mut self.geometry.meshes {
            mesh.refresh_emissive_components(&materials);
        }
    }

    pub fn material_usage(&self, material: MaterialId) -> usize {
        self.geometry
            .meshes
            .iter()
            .flat_map(|mesh| mesh.primitives.iter())
            .filter(|primitive| primitive.material == material)
            .map(|primitive| primitive.index_count / 3)
            .sum()
    }

    pub fn add_material(&mut self) -> MaterialId {
        let mut index = self.materials.len();
        let name = loop {
            let candidate = format!("Material {index:02}");
            if self
                .materials
                .iter()
                .all(|material| material.name != candidate)
            {
                break candidate;
            }
            index += 1;
        };
        let id = MaterialId(self.materials.len());
        self.materials.push(Material::named(name));
        id
    }

    pub fn rename_material(&mut self, material: MaterialId, name: impl Into<String>) -> bool {
        let name = name.into().trim().to_owned();
        if name.is_empty() {
            return false;
        }
        let Some(target) = self.materials.get_mut(material.0) else {
            return false;
        };
        if target.name == name {
            return false;
        }
        target.name = name;
        true
    }

    /// Delete a material and remap every primitive that referenced it to the default material.
    /// Material zero is intentionally permanent so deletion never leaves an invalid binding.
    pub fn remove_material(&mut self, material: MaterialId) -> Option<Material> {
        if material == MaterialId::DEFAULT || material.0 >= self.materials.len() {
            return None;
        }
        let removed = self.materials.remove(material.0);
        for mesh in &mut self.geometry.meshes {
            for primitive in &mut mesh.primitives {
                primitive.material = if primitive.material == material {
                    MaterialId::DEFAULT
                } else if primitive.material.0 > material.0 {
                    MaterialId(primitive.material.0 - 1)
                } else {
                    primitive.material
                };
            }
        }
        self.refresh_emissive_components();
        Some(removed)
    }
}

impl Deref for Scene {
    type Target = SceneData;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for Scene {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
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
        assert!(scene.lights.is_empty());
        assert!(
            scene
                .geometry
                .meshes
                .iter()
                .any(|mesh| !mesh.emissive_components.is_empty())
        );
        assert!(!scene.nodes.is_empty());
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
                source_index: None,
                normal: Some(glam::Vec3::Z),
                uv: Some(glam::Vec2::new(0.0, 0.0)),
            },
            Vertex {
                position: glam::Vec3::new(1.0, 0.0, 0.0),
                source_index: None,
                normal: Some(glam::Vec3::Z),
                uv: Some(glam::Vec2::new(1.0, 0.0)),
            },
            Vertex {
                position: glam::Vec3::new(0.0, 1.0, 0.0),
                source_index: None,
                normal: Some(glam::Vec3::Z),
                uv: Some(glam::Vec2::new(0.0, 1.0)),
            },
        ];
        let mesh = Mesh::with_material(vertices, vec![0, 1, 2], material_id);
        let geometry = Geometry::new(
            vec![mesh],
            vec![Instance {
                mesh: MeshId(0),
                path: "/Geometry".into(),
                transform: glam::DMat4::IDENTITY,
            }],
        );
        let scene = Scene::new(SceneData {
            geometry,
            lights: Vec::new(),
            nodes: Vec::new(),
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

    #[test]
    fn material_lifecycle_renames_and_remaps_geometry_bindings() {
        let mesh = Mesh {
            vertices: vec![Vertex::default(); 3],
            indices: vec![0, 1, 2],
            primitives: vec![
                Primitive {
                    index_start: 0,
                    index_count: 3,
                    material: MaterialId(2),
                },
                Primitive {
                    index_start: 0,
                    index_count: 3,
                    material: MaterialId(3),
                },
            ],
            emissive_components: Vec::new(),
        };
        let mut scene = Scene::new(SceneData {
            geometry: Geometry::new(vec![mesh], Vec::new()),
            lights: Vec::new(),
            nodes: Vec::new(),
            materials: vec![
                Material::named("Default"),
                Material::named("Keep"),
                Material::named("Remove"),
                Material::named("Shift"),
            ],
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

        assert_eq!(scene.material_usage(MaterialId(2)), 1);
        assert!(scene.rename_material(MaterialId(2), "Renamed"));
        assert_eq!(scene.materials[2].name, "Renamed");
        assert!(!scene.rename_material(MaterialId(2), "   "));

        let removed = scene.remove_material(MaterialId(2)).unwrap();
        assert_eq!(removed.name, "Renamed");
        assert_eq!(scene.materials.len(), 3);
        assert_eq!(scene.materials[2].name, "Shift");
        assert_eq!(
            scene.geometry.meshes[0].primitives[0].material,
            MaterialId::DEFAULT
        );
        assert_eq!(
            scene.geometry.meshes[0].primitives[1].material,
            MaterialId(2)
        );
        assert_eq!(scene.material_usage(MaterialId(2)), 1);
        assert!(scene.remove_material(MaterialId::DEFAULT).is_none());
    }
}
