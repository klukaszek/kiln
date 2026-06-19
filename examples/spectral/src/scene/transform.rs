use glam::DMat4;
use openusd::sdf;
use openusd::usd::Stage;

/// Compose a prim's local-to-world transform by walking ancestors to the root.
///
/// openusd authors row-major, **row-vector** matrices (`p' = p · M`); read as
/// column-major they are exactly glam's column-vector convention, so
/// `from_cols_array` is a free transpose and the child-first accumulation
/// `p · local_child · local_parent · …` becomes left-multiplication.
pub fn world_xform(stage: &Stage, prim: &sdf::Path) -> anyhow::Result<DMat4> {
    use openusd::schemas::geom::compute_local_to_parent_transform;

    let mut world = DMat4::IDENTITY;
    let mut cur = Some(prim.clone());
    while let Some(path) = cur {
        if path.name().is_none() {
            break;
        }
        let local =
            DMat4::from_cols_array(&compute_local_to_parent_transform(stage, &path, 0.0)?);
        world = local * world;
        cur = path.parent();
    }
    Ok(world)
}
