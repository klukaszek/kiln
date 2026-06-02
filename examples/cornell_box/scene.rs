//! USD scene loading for the Cornell box example.
//!
//! This module is the render-facing facade. It keeps the current raster path simple
//! (`Scene::vertices` plus camera helpers), while the implementation is split into the
//! scene parts that the path tracer will need next: camera, geometry, materials,
//! transforms, and small math helpers.
//!
//! All matrix math follows openusd's convention: `[f64; 16]` row-major, **row-vector**
//! (`p' = p · M`, translation in indices 12..14).

mod camera;
mod geometry;
mod material;
mod math;
mod transform;

pub use camera::Camera;
pub use material::Material;

use openusd::schemas::geom::find_geom_prims;
use openusd::usd::Stage;

use crate::Vertex;

/// Everything the renderer currently needs from the scene.
pub struct Scene {
    pub vertices: Vec<Vertex>,
    pub triangle_materials: Vec<u32>,
    pub materials: Vec<Material>,
    pub camera: Camera,
}

impl Scene {
    /// Number of triangles in the soup (three vertices each).
    pub fn triangle_count(&self) -> u32 {
        (self.vertices.len() / 3) as u32
    }

    /// Camera world-space position (translation row of the world transform).
    pub fn camera_pos(&self) -> [f32; 3] {
        self.camera.position()
    }

    /// Build the row-vector `view_proj = view · proj` for the given viewport aspect.
    pub fn view_proj_rows(&self, aspect: f32) -> [[f32; 4]; 4] {
        self.camera.view_proj_rows(aspect)
    }
}

/// Load `path` (a `.usda`/`.usdc`/`.usdz`) into a [`Scene`].
pub fn load(path: &str) -> anyhow::Result<Scene> {
    let stage = Stage::open(path)?;
    let prims = find_geom_prims(&stage)?;
    let geometry = geometry::load_geometry(&stage, &prims.meshes)?;

    Ok(Scene {
        vertices: geometry.vertices,
        triangle_materials: geometry.triangle_materials,
        materials: geometry.materials,
        camera: camera::load_first(&stage, &prims.cameras)?,
    })
}
