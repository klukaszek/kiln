use openusd::math::{IDENTITY_MAT4, mat4_inverse, mat4_mul};
use openusd::schemas::geom::{ReadCamera, read_camera};
use openusd::sdf;
use openusd::usd::Stage;

use super::math::perspective_rh_zo;
use super::transform::world_xform;

/// Authored camera data plus its local-to-world transform.
pub struct Camera {
    pub world: [f64; 16],
    pub usd: ReadCamera,
}

impl Camera {
    pub fn position(&self) -> [f32; 3] {
        let m = &self.world;
        [m[12] as f32, m[13] as f32, m[14] as f32]
    }

    /// Right-handed (camera looks down -Z), Y-up, depth range [0, 1] - matching Kiln's
    /// normalized clip space. Returned as four `float4` rows for layout-free shader use.
    pub fn view_proj_rows(&self, aspect: f32) -> [[f32; 4]; 4] {
        let view = mat4_inverse(&self.world).unwrap_or(IDENTITY_MAT4);
        let proj = perspective_rh_zo(
            self.usd.vertical_fov_rad(),
            aspect,
            self.usd.clipping_range[0].max(1e-3),
            self.usd.clipping_range[1],
        );
        let vp = mat4_mul(&view, &proj);
        [
            [vp[0] as f32, vp[1] as f32, vp[2] as f32, vp[3] as f32],
            [vp[4] as f32, vp[5] as f32, vp[6] as f32, vp[7] as f32],
            [vp[8] as f32, vp[9] as f32, vp[10] as f32, vp[11] as f32],
            [vp[12] as f32, vp[13] as f32, vp[14] as f32, vp[15] as f32],
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
