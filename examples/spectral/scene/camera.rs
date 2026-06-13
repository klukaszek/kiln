use glam::{DMat4, Vec3, Vec4};
use openusd::schemas::geom::{ReadCamera, read_camera};
use openusd::sdf;
use openusd::usd::Stage;

use super::transform::world_xform;

/// Authored camera data plus its local-to-world transform (glam column-vector
/// convention — see [`world_xform`] for how openusd's data transposes in).
pub struct Camera {
    pub world: DMat4,
    pub usd: ReadCamera,
}

impl Camera {
    pub fn position(&self) -> Vec3 {
        self.world.w_axis.truncate().as_vec3()
    }

    /// Right-handed (camera looks down -Z), Y-up, depth range [0, 1] — matching
    /// Kiln's normalized clip space. Returned as the four rows of the row-vector
    /// `view·proj` (= the columns of the column-vector matrix) for layout-free
    /// shader use.
    pub fn view_proj_rows(&self, aspect: f32) -> [Vec4; 4] {
        let view = self.world.inverse();
        let proj = DMat4::perspective_rh(
            self.usd.vertical_fov_rad() as f64,
            aspect as f64,
            self.usd.clipping_range[0].max(1e-3) as f64,
            self.usd.clipping_range[1] as f64,
        );
        let vp = proj * view;
        [
            vp.x_axis.as_vec4(),
            vp.y_axis.as_vec4(),
            vp.z_axis.as_vec4(),
            vp.w_axis.as_vec4(),
        ]
    }
}

pub fn load_first(stage: &Stage, camera_paths: &[String]) -> anyhow::Result<Camera> {
    let cam_path = camera_paths
        .first()
        .ok_or_else(|| anyhow::anyhow!("scene has no Camera prim"))?;
    let cam_sdf = sdf::path(cam_path)?;
    let usd = read_camera(stage, &cam_sdf)?
        .ok_or_else(|| anyhow::anyhow!("camera prim {cam_path} did not read as a Camera"))?;

    Ok(Camera {
        world: world_xform(stage, &cam_sdf)?,
        usd,
    })
}
