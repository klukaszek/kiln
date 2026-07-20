//! USD scene loading.
//!
//! This module is the render-facing facade. It owns the GPU vertex layout the loaders
//! produce ([`Vertex`]), while the implementation is split into the parts of a stage the
//! renderer consumes: camera, geometry, materials, transforms, and small math helpers.
//!
//! All matrix math follows openusd's convention: `[f64; 16]` row-major, **row-vector**
//! (`p' = p · M`, translation in indices 12..14).

mod camera;
mod geometry;
mod material;
mod transform;

pub mod gpu;
pub mod spectral;

pub use camera::Camera;
pub use material::Material;

use glam::{DVec3, Vec3, Vec4};
use kiln_rhi::gpu_struct;
use openusd::schemas::geom::find_geom_prims;
use openusd::sdf::{self, Value};
use openusd::usd::Stage;

gpu_struct! {
    /// Per-vertex render data. Declared ahead of any root struct whose Slang source
    /// dereferences a `Vertex*`.
    pub struct Vertex {
        pos: Vec4,    // world position, w = 1
        normal: Vec4, // world normal,   w = 0
        color: Vec4,  // linear RGB,      w = 1
    }
}

/// Everything the renderer currently needs from the scene.
pub struct Scene {
    pub vertices: Vec<Vertex>,
    pub triangle_materials: Vec<u32>,
    pub materials: Vec<Material>,
    pub camera: Camera,
    /// World up axis from the stage's `upAxis` metadata (Y when unauthored).
    /// Geometry and cameras arrive in world space, so a Z-up stage stays Z-up;
    /// interactive camera controls must level against this axis, not Y.
    pub up: DVec3,
}

impl Scene {
    /// Number of triangles in the soup (three vertices each).
    pub fn triangle_count(&self) -> usize {
        self.vertices.len() / 3
    }

    /// Camera world-space position (translation row of the world transform).
    pub fn camera_pos(&self) -> Vec3 {
        self.camera.position()
    }

    /// Build the row-vector `view_proj = view · proj` for the given viewport aspect.
    pub fn view_proj_rows(&self, aspect: f32) -> [Vec4; 4] {
        self.camera.view_proj_rows(aspect)
    }
}

/// Load `path` (a `.usda`/`.usdc`/`.usdz`) into a [`Scene`].
pub fn load(path: &std::path::Path) -> anyhow::Result<Scene> {
    let path = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("non-UTF-8 scene path {}", path.display()))?;
    let stage = Stage::open(path)?;
    let prims = find_geom_prims(&stage)?;
    let geometry = geometry::load_geometry(&stage, &prims.meshes)?;

    Ok(Scene {
        vertices: geometry.vertices,
        triangle_materials: geometry.triangle_materials,
        materials: geometry.materials,
        camera: camera::load_first(&stage, &prims.cameras)?,
        up: stage_up_axis(&stage)?,
    })
}

/// The stage's `upAxis` layer metadata, read off the pseudo-root.
fn stage_up_axis(stage: &Stage) -> anyhow::Result<DVec3> {
    let axis = stage.field::<Value>(sdf::path("/")?, "upAxis")?;
    Ok(match axis {
        Some(Value::Token(axis) | Value::String(axis)) if axis == "Z" => DVec3::Z,
        _ => DVec3::Y,
    })
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn loads_bundled_cornell_box() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/cornell-box.usda");
        let scene = load(&path).unwrap();

        assert!(!scene.vertices.is_empty());
        assert_eq!(scene.vertices.len() % 3, 0);
        assert_eq!(scene.triangle_materials.len(), scene.triangle_count());
        assert!(!scene.materials.is_empty());
    }
}
