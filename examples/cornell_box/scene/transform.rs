use openusd::math::{IDENTITY_MAT4, mat4_mul};
use openusd::sdf;
use openusd::usd::Stage;

/// Compose a prim's local-to-world transform by walking ancestors to the root. Row-vector
/// convention: `world = local · parent_local · ... · root_local`, accumulated child-first.
pub fn world_xform(stage: &Stage, prim: &sdf::Path) -> anyhow::Result<[f64; 16]> {
    use openusd::schemas::geom::compute_local_to_parent_transform;

    let mut world = IDENTITY_MAT4;
    let mut cur = Some(prim.clone());
    while let Some(path) = cur {
        if path.name().is_none() {
            break;
        }
        let local = compute_local_to_parent_transform(stage, &path, 0.0)?;
        world = mat4_mul(&world, &local);
        cur = path.parent();
    }
    Ok(world)
}
