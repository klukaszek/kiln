//! USD importer. All `openusd` types stop at this module boundary.

mod material;
mod mesh;
mod transform;

use std::path::Path;

use openusd::schemas::geom::{find_geom_prims, read_camera};
use openusd::sdf;
use openusd::usd::Stage;

use crate::scene::{Camera, Projection, Scene};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("openusd: {0}")]
    OpenUsd(String),
    #[error("USD index exceeds supported capacity")]
    IndexOverflow(#[from] std::num::TryFromIntError),
    #[error("invalid USD data: {0}")]
    Invalid(String),
    #[error("scene has no camera")]
    MissingCamera,
    #[error("scene path is not valid UTF-8: {0}")]
    NonUtf8Path(String),
    #[error("connected Preview Surface input {input} is unsupported (source {connection})")]
    ConnectedInput { input: String, connection: String },
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<anyhow::Error> for Error {
    fn from(error: anyhow::Error) -> Self {
        Self::OpenUsd(error.to_string())
    }
}

pub fn load(path: &Path) -> Result<Scene> {
    let path = path
        .to_str()
        .ok_or_else(|| Error::NonUtf8Path(path.display().to_string()))?;
    let stage = Stage::open(path)?;
    let prims = find_geom_prims(&stage)?;

    // Material discovery is importer orchestration. Mesh decoding only receives material IDs and
    // can therefore be reused by another importer without constructing a material cache.
    let mut materials = material::MaterialLibrary::new();
    let geometry = mesh::load(&stage, &prims.meshes, &mut materials)?;
    let camera = load_camera(&stage, &prims.cameras)?;
    let up = transform::stage_up_axis(&stage)?;

    Ok(Scene {
        geometry,
        materials: materials.into_materials(),
        camera,
        up,
    })
}

fn load_camera(stage: &Stage, camera_paths: &[String]) -> Result<Camera> {
    let camera_path = camera_paths.first().ok_or(Error::MissingCamera)?;
    let path = sdf::path(camera_path)?;
    let camera = read_camera(stage, &path)?
        .ok_or_else(|| Error::Invalid(format!("{camera_path} is not a readable camera")))?;
    Ok(Camera {
        world: transform::world_xform(stage, &path)?,
        projection: Projection {
            vertical_fov_rad: camera.vertical_fov_rad(),
            clipping_range: camera.clipping_range,
        },
    })
}
