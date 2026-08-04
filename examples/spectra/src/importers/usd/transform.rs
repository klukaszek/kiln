use glam::{DMat4, DQuat, DVec3};
use openusd::schemas::geom::read_xform_op_order;
use openusd::sdf::{self, Value};
use openusd::usd::Stage;

use super::{Error, Result};

const INVERT: &str = "!invert!";
const RESET: &str = "!resetXformStack!";

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

pub(super) fn world_xform(stage: &Stage, prim: &sdf::Path) -> Result<DMat4> {
    let mut world = DMat4::IDENTITY;
    let mut current = Some(prim.clone());
    while let Some(path) = current {
        if path.name().is_none() {
            break;
        }
        let (local, reset) = local_xform(stage, &path)?;
        world = local * world;
        if reset {
            break;
        }
        current = path.parent();
    }
    Ok(world)
}

fn local_xform(stage: &Stage, prim: &sdf::Path) -> Result<(DMat4, bool)> {
    let Some(order) = read_xform_op_order(stage, prim)? else {
        return Ok((DMat4::IDENTITY, false));
    };
    let reset = order.first().is_some_and(|op| op == RESET);

    let mut local = DMat4::IDENTITY;
    for op in &order[usize::from(reset)..] {
        local *= xform_op(stage, prim, op)?;
    }
    Ok((local, reset))
}

fn xform_op(stage: &Stage, prim: &sdf::Path, op: &str) -> Result<DMat4> {
    let (op, invert) = op.strip_prefix(INVERT).map_or((op, false), |op| (op, true));
    let Some(value) = stage.value_at(prim.append_property(op)?, 0.0)? else {
        return Ok(DMat4::IDENTITY);
    };
    let kind = op
        .strip_prefix("xformOp:")
        .and_then(|op| op.split(':').next())
        .ok_or_else(|| Error::Invalid(format!("invalid xform op {op}")))?;
    let matrix = match kind {
        "translate" => DMat4::from_translation(vec3(&value, op)?),
        "translateX" => DMat4::from_translation(DVec3::X * scalar(&value, op)?),
        "translateY" => DMat4::from_translation(DVec3::Y * scalar(&value, op)?),
        "translateZ" => DMat4::from_translation(DVec3::Z * scalar(&value, op)?),
        "scale" => DMat4::from_scale(vec3(&value, op)?),
        "scaleX" => DMat4::from_scale(DVec3::new(scalar(&value, op)?, 1.0, 1.0)),
        "scaleY" => DMat4::from_scale(DVec3::new(1.0, scalar(&value, op)?, 1.0)),
        "scaleZ" => DMat4::from_scale(DVec3::new(1.0, 1.0, scalar(&value, op)?)),
        "rotateX" => DMat4::from_rotation_x(scalar(&value, op)?.to_radians()),
        "rotateY" => DMat4::from_rotation_y(scalar(&value, op)?.to_radians()),
        "rotateZ" => DMat4::from_rotation_z(scalar(&value, op)?.to_radians()),
        "rotateXYZ" | "rotateXZY" | "rotateYXZ" | "rotateYZX" | "rotateZXY" | "rotateZYX" => {
            euler(&kind[6..], vec3(&value, op)?)
        }
        "orient" => DMat4::from_quat(quat(&value, op)?),
        "transform" => match value {
            Value::Matrix4d(matrix) => DMat4::from_cols_array(&matrix),
            _ => return Err(invalid_value(op)),
        },
        _ => return Err(Error::Invalid(format!("unsupported xform op {op}"))),
    };
    Ok(if invert { matrix.inverse() } else { matrix })
}

fn euler(order: &str, degrees: DVec3) -> DMat4 {
    order.bytes().fold(DMat4::IDENTITY, |matrix, axis| {
        let rotation = match axis {
            b'X' => DMat4::from_rotation_x(degrees.x.to_radians()),
            b'Y' => DMat4::from_rotation_y(degrees.y.to_radians()),
            b'Z' => DMat4::from_rotation_z(degrees.z.to_radians()),
            _ => unreachable!(),
        };
        rotation * matrix
    })
}

fn scalar(value: &Value, op: &str) -> Result<f64> {
    match value {
        Value::Half(value) => Ok(value.to_f32() as f64),
        Value::Float(value) => Ok(*value as f64),
        Value::Double(value) => Ok(*value),
        _ => Err(invalid_value(op)),
    }
}

fn vec3(value: &Value, op: &str) -> Result<DVec3> {
    match value {
        Value::Vec3h(value) => Ok(DVec3::from_array(value.map(|x| x.to_f32() as f64))),
        Value::Vec3f(value) => Ok(DVec3::from_array(value.map(f64::from))),
        Value::Vec3d(value) => Ok(DVec3::from_array(*value)),
        _ => Err(invalid_value(op)),
    }
}

fn quat(value: &Value, op: &str) -> Result<DQuat> {
    let [w, x, y, z] = match value {
        Value::Quath(value) => value.map(|x| x.to_f32() as f64),
        Value::Quatf(value) => value.map(f64::from),
        Value::Quatd(value) => *value,
        _ => return Err(invalid_value(op)),
    };
    Ok(DQuat::from_xyzw(x, y, z, w))
}

fn invalid_value(op: &str) -> Error {
    Error::Invalid(format!("invalid value for xform op {op}"))
}

#[cfg(test)]
mod tests {
    use openusd::schemas::geom::{set_rotate_z, set_scale, set_translate, set_xform_op_order};

    use super::*;

    #[test]
    fn xform_order_is_least_to_most_local() -> anyhow::Result<()> {
        let stage = Stage::builder().in_memory("xform-order.usda")?;
        stage.define_prim("/Parent")?.set_type_name("Xform")?;
        stage
            .define_prim("/Parent/Camera")?
            .set_type_name("Camera")?;
        let parent = sdf::path("/Parent")?;
        let camera = sdf::path("/Parent/Camera")?;

        set_translate(&stage, &parent, [10.0, 0.0, 0.0])?;
        set_translate(&stage, &camera, [1.0, 2.0, 3.0])?;
        set_rotate_z(&stage, &camera, 90.0)?;
        set_scale(&stage, &camera, [2.0, 2.0, 2.0])?;

        let world = world_xform(&stage, &camera)?;
        assert!(
            world
                .transform_point3(DVec3::ZERO)
                .abs_diff_eq(DVec3::new(11.0, 2.0, 3.0), 1e-12)
        );
        assert!(
            world
                .transform_point3(DVec3::X)
                .abs_diff_eq(DVec3::new(11.0, 4.0, 3.0), 1e-12)
        );
        Ok(())
    }

    #[test]
    fn reset_xform_stack_stops_parent_inheritance() -> anyhow::Result<()> {
        let stage = Stage::builder().in_memory("xform-reset.usda")?;
        stage.define_prim("/Parent")?.set_type_name("Xform")?;
        stage.define_prim("/Parent/Child")?.set_type_name("Xform")?;
        let parent = sdf::path("/Parent")?;
        let child = sdf::path("/Parent/Child")?;

        set_translate(&stage, &parent, [10.0, 0.0, 0.0])?;
        set_translate(&stage, &child, [1.0, 2.0, 3.0])?;
        set_xform_op_order(&stage, &child, [RESET, "xformOp:translate"])?;

        let world = world_xform(&stage, &child)?;
        assert!(
            world
                .transform_point3(DVec3::ZERO)
                .abs_diff_eq(DVec3::new(1.0, 2.0, 3.0), 1e-12)
        );
        Ok(())
    }
}
