//! USD importer. All `openusd` types stop at this module boundary.

mod material;
mod mesh;
mod texture;
mod transform;

use std::path::{Path, PathBuf};

use openusd::schemas::geom::{find_geom_prims, read_camera};
use openusd::schemas::lux::{
    find_lux_prims, read_cylinder_light, read_disk_light, read_distant_light, read_dome_light,
    read_rect_light, read_sphere_light,
};
use openusd::sdf;
use openusd::usd::Stage;

use glam::Vec3;

use crate::base::scene::{
    Camera, Illuminant, Light, LightKind, Projection, Scene, SceneData, build_scene_nodes,
};

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
    #[error("unsupported texture source for {input}: {connection} ({reason})")]
    UnsupportedTexture {
        input: String,
        connection: String,
        reason: String,
    },
    #[error("texture asset not found: {0}")]
    MissingTexture(String),
    #[error("failed to decode texture {path}: {source}")]
    Texture {
        path: PathBuf,
        source: image::ImageError,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<anyhow::Error> for Error {
    fn from(error: anyhow::Error) -> Self {
        Self::OpenUsd(error.to_string())
    }
}

pub fn load(path: &Path) -> Result<Scene> {
    let path_str = path
        .to_str()
        .ok_or_else(|| Error::NonUtf8Path(path.display().to_string()))?;
    let stage = Stage::open(path_str)?;
    let prims = find_geom_prims(&stage)?;
    let lux_prims = find_lux_prims(&stage)?;

    // Material discovery is importer orchestration. Mesh decoding only receives material IDs and
    // can therefore be reused by another importer without constructing a material cache.
    let textures = texture::TextureLibrary::new(path, &stage);
    let mut materials = material::MaterialLibrary::new(textures);
    let geometry = mesh::load(&stage, &prims.meshes, &mut materials)?;
    let camera = load_camera(&stage, &prims.cameras)?;
    let up = transform::stage_up_axis(&stage)?;
    let lights = load_lights(&stage, &lux_prims)?;
    let nodes = build_scene_nodes(&geometry.instances, &lights);

    let (materials, images, textures) = materials.into_parts();
    Ok(Scene::new(SceneData {
        geometry,
        lights,
        nodes,
        materials,
        images,
        textures,
        camera,
        up,
    }))
}

fn load_lights(stage: &Stage, prims: &openusd::schemas::lux::LuxPrims) -> Result<Vec<Light>> {
    let mut lights = Vec::new();
    for path in &prims.distant {
        if let Some(value) = read_distant_light(stage, &sdf::path(path)?)? {
            lights.push(imported_light(
                stage,
                path,
                &value.common,
                LightKind::Directional {
                    angle_deg: value.angle_deg,
                },
            )?);
        }
    }
    for path in &prims.sphere {
        if let Some(value) = read_sphere_light(stage, &sdf::path(path)?)? {
            let kind = if value.treat_as_point {
                LightKind::Point
            } else {
                LightKind::Sphere {
                    radius: value.radius,
                }
            };
            lights.push(imported_light(stage, path, &value.common, kind)?);
        }
    }
    for path in &prims.rect {
        if let Some(value) = read_rect_light(stage, &sdf::path(path)?)? {
            lights.push(imported_light(
                stage,
                path,
                &value.common,
                LightKind::Rect {
                    width: value.width,
                    height: value.height,
                },
            )?);
        }
    }
    for path in &prims.disk {
        if let Some(value) = read_disk_light(stage, &sdf::path(path)?)? {
            lights.push(imported_light(
                stage,
                path,
                &value.common,
                LightKind::Disk {
                    radius: value.radius,
                },
            )?);
        }
    }
    for path in &prims.cylinder {
        if let Some(value) = read_cylinder_light(stage, &sdf::path(path)?)? {
            lights.push(imported_light(
                stage,
                path,
                &value.common,
                LightKind::Rect {
                    width: value.radius * 2.0,
                    height: value.length,
                },
            )?);
        }
    }
    for path in &prims.dome {
        if let Some(value) = read_dome_light(stage, &sdf::path(path)?)? {
            lights.push(imported_light(stage, path, &value.common, LightKind::Dome)?);
        }
    }

    Ok(lights)
}

fn imported_light(
    stage: &Stage,
    path: &str,
    common: &openusd::schemas::lux::ReadLight,
    kind: LightKind,
) -> Result<Light> {
    let name = path
        .rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
        .unwrap_or("Light");
    let transform = transform::world_xform(stage, &sdf::path(path)?)?;
    // UsdLux intensity is unitless, so it maps onto the renderer's native quantity; exposure stays
    // a separate control so the inspector shows the authored stops rather than a folded product.
    Ok(Light {
        name: name.to_owned(),
        path: path.to_owned(),
        kind,
        transform,
        illuminant: Illuminant {
            exposure: common.exposure,
            ..Illuminant::luminance(Vec3::from_array(common.color), common.intensity.max(0.0))
        },
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
