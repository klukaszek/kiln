use std::collections::HashMap;

use glam::Vec3;
use openusd::sdf::{self, Value};
use openusd::usd::Stage;

use crate::scene::{Material, MaterialId, PrincipledBsdf, Surface};

use super::{Error, Result};

pub(super) struct MaterialLibrary {
    by_path: HashMap<String, MaterialId>,
    materials: Vec<Material>,
}

impl MaterialLibrary {
    pub(super) fn new() -> Self {
        Self {
            by_path: HashMap::new(),
            materials: vec![Material::default()],
        }
    }

    pub(super) fn id_for_prim(&mut self, stage: &Stage, prim_path: &str) -> Result<MaterialId> {
        let Some(material_path) = material_binding(stage, prim_path)? else {
            return Ok(self.default_id());
        };
        if let Some(&id) = self.by_path.get(&material_path) {
            return Ok(id);
        }
        let material = read_material(stage, &material_path)?;
        let id = MaterialId(self.materials.len());
        self.materials.push(material);
        self.by_path.insert(material_path, id);
        Ok(id)
    }

    pub(super) fn into_materials(self) -> Vec<Material> {
        self.materials
    }

    fn default_id(&self) -> MaterialId {
        MaterialId::DEFAULT
    }
}

fn material_binding(stage: &Stage, prim_path: &str) -> Result<Option<String>> {
    let rel = sdf::path(prim_path)?.append_property("material:binding")?;
    let paths = match stage.field::<Value>(rel, "targetPaths")? {
        Some(Value::PathListOp(op)) => op.flatten(),
        Some(Value::PathVec(paths)) => paths,
        _ => Vec::new(),
    };
    Ok(paths.first().map(|path| path.as_str().to_owned()))
}

fn read_material(stage: &Stage, material_path: &str) -> Result<Material> {
    let material = sdf::path(material_path)?;
    let mut rejected_outputs = Vec::new();
    let mut shader_path = None;
    for output_name in ["outputs:mtlx:surface", "outputs:surface"] {
        let output_property = material.append_property(output_name)?;
        let Some(output) = connected_path(stage, output_property)? else {
            continue;
        };
        let candidate = sdf::path(&output)?.prim_path();
        match shader_id(stage, &candidate) {
            Ok(Some(id)) if is_preview_surface(&id) => {
                shader_path = Some(candidate);
                break;
            }
            Ok(Some(id)) => rejected_outputs.push(format!("{output}: unsupported shader {id}")),
            Ok(None) => rejected_outputs.push(format!("{output}: missing info:id")),
            Err(error) => rejected_outputs.push(format!("{output}: {error}")),
        }
    }
    let shader_path = shader_path.ok_or_else(|| {
        if rejected_outputs.is_empty() {
            Error::Invalid(format!(
                "material {material_path} has no connected surface output"
            ))
        } else {
            Error::Invalid(format!(
                "material {material_path} has no supported surface output ({})",
                rejected_outputs.join(", ")
            ))
        }
    })?;

    let mut result = Material::default();
    let mut surface = PrincipledBsdf::default();
    if let Some(color) = read_vec3(stage, &shader_path, "inputs:diffuseColor")? {
        surface.base_color = color;
    }
    if let Some(color) = read_vec3(stage, &shader_path, "inputs:baseColor")? {
        surface.base_color = color;
    }
    if let Some(color) = read_vec3(stage, &shader_path, "inputs:emissiveColor")? {
        result.emission.color = color;
    }
    if let Some(color) = read_vec3(stage, &shader_path, "inputs:emissionColor")? {
        result.emission.color = color;
    }
    if let Some(value) = read_f32(stage, &shader_path, "inputs:roughness")? {
        surface.roughness = value;
    }
    if let Some(value) = read_f32(stage, &shader_path, "inputs:metallic")? {
        surface.metallic = value;
    }
    if let Some(value) = read_f32(stage, &shader_path, "inputs:ior")? {
        surface.ior = value;
    }
    result.surface = Surface::Principled(surface);
    for input in [
        "inputs:diffuseColor",
        "inputs:baseColor",
        "inputs:emissiveColor",
        "inputs:emissionColor",
        "inputs:roughness",
        "inputs:metallic",
        "inputs:ior",
    ] {
        reject_connected_input(stage, &shader_path, input)?;
    }
    Ok(result)
}

fn shader_id(stage: &Stage, shader_path: &sdf::Path) -> Result<Option<String>> {
    let shader_id_path = shader_path.append_property("info:id")?;
    Ok(match stage.field::<Value>(shader_id_path, "default")? {
        Some(Value::Token(id) | Value::String(id)) => Some(id),
        _ => None,
    })
}

fn is_preview_surface(shader_id: &str) -> bool {
    shader_id.contains("UsdPreviewSurface") || shader_id.contains("PreviewSurface")
}

fn reject_connected_input(stage: &Stage, shader: &sdf::Path, input: &str) -> Result<()> {
    let property = shader.append_property(input)?;
    if let Some(source) = connected_path(stage, property)? {
        return Err(Error::ConnectedInput {
            input: input.to_owned(),
            connection: source,
        });
    }
    Ok(())
}

fn connected_path(stage: &Stage, property: sdf::Path) -> Result<Option<String>> {
    let paths = match stage.field::<Value>(property, "connectionPaths")? {
        Some(Value::PathListOp(op)) => op.flatten(),
        Some(Value::PathVec(paths)) => paths,
        _ => Vec::new(),
    };
    Ok(paths.first().map(|path| path.as_str().to_owned()))
}

fn read_vec3(stage: &Stage, shader: &sdf::Path, name: &str) -> Result<Option<Vec3>> {
    let property = shader.append_property(name)?;
    Ok(match stage.field::<Value>(property, "default")? {
        Some(Value::Vec3f(value)) => Some(Vec3::from_array(value)),
        Some(Value::Vec3d(value)) => Some(glam::DVec3::from_array(value).as_vec3()),
        _ => None,
    })
}

fn read_f32(stage: &Stage, shader: &sdf::Path, name: &str) -> Result<Option<f32>> {
    let property = shader.append_property(name)?;
    Ok(match stage.field::<Value>(property, "default")? {
        Some(Value::Float(value)) => Some(value),
        Some(Value::Double(value)) => Some(value as f32),
        Some(Value::Half(value)) => Some(value.to_f32()),
        Some(Value::Int(value)) => Some(value as f32),
        Some(Value::Int64(value)) => Some(value as f32),
        _ => None,
    })
}
