use glam::{DMat4, DVec3};
use openusd::sdf;
use openusd::usd::Stage;

use super::Result;

pub(super) fn stage_up_axis(stage: &Stage) -> Result<DVec3> {
    let axis = stage.field::<openusd::sdf::Value>(sdf::path("/")?, "upAxis")?;
    Ok(match axis {
        Some(openusd::sdf::Value::Token(axis) | openusd::sdf::Value::String(axis))
            if axis == "Z" =>
        {
            DVec3::Z
        }
        _ => DVec3::Y,
    })
}

/// Compose an OpenUSD prim's local-to-world transform. OpenUSD stores row-vector matrices;
/// reading them as glam column-major values and multiplying from the left gives the same result.
pub(super) fn world_xform(stage: &Stage, prim: &sdf::Path) -> Result<DMat4> {
    use openusd::schemas::geom::compute_local_to_parent_transform;

    let mut world = DMat4::IDENTITY;
    let mut current = Some(prim.clone());
    while let Some(path) = current {
        if path.name().is_none() {
            break;
        }
        let local = DMat4::from_cols_array(&compute_local_to_parent_transform(stage, &path, 0.0)?);
        world = local * world;
        current = path.parent();
    }
    Ok(world)
}
