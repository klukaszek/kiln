use std::collections::HashMap;

use glam::Vec3;
use openusd::sdf::{self, Value};
use openusd::usd::Stage;

const DEFAULT_BASE_COLOR: Vec3 = Vec3::splat(0.8);
const DEFAULT_EMISSION: Vec3 = Vec3::ZERO;
const DEFAULT_ROUGHNESS: f32 = 0.5;
const DEFAULT_METALLIC: f32 = 0.0;
const DEFAULT_IOR: f32 = 1.5;

/// The subset of UsdPreviewSurface implemented by the spectral tracer.
#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub base_color: Vec3,
    pub emission: Vec3,
    pub roughness: f32,
    pub metallic: f32,
    pub ior: f32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: DEFAULT_BASE_COLOR,
            emission: DEFAULT_EMISSION,
            roughness: DEFAULT_ROUGHNESS,
            metallic: DEFAULT_METALLIC,
            ior: DEFAULT_IOR,
        }
    }
}

impl Material {
    pub fn raster_color(&self) -> Vec3 {
        if self.is_emissive() {
            self.emission
        } else {
            self.base_color
        }
    }

    pub fn is_emissive(&self) -> bool {
        self.emission.max_element() > 0.0
    }
}

#[derive(Default)]
pub(super) struct MaterialCache {
    by_path: HashMap<String, u32>,
    materials: Vec<Material>,
}

impl MaterialCache {
    pub(super) fn material_index_for_prim(
        &mut self,
        stage: &Stage,
        prim_path: &str,
    ) -> anyhow::Result<u32> {
        let Some(material_path) = material_binding(stage, prim_path)? else {
            return self.default_material_index();
        };

        if let Some(&index) = self.by_path.get(&material_path) {
            return Ok(index);
        }

        let material = read_material(stage, &material_path)?;
        let index = u32::try_from(self.materials.len())?;
        self.materials.push(material);
        self.by_path.insert(material_path, index);
        Ok(index)
    }

    pub(super) fn material(&self, index: u32) -> &Material {
        &self.materials[index as usize]
    }

    pub(super) fn into_materials(mut self) -> Vec<Material> {
        if self.materials.is_empty() {
            self.materials.push(Material::default());
        }
        self.materials
    }

    fn default_material_index(&mut self) -> anyhow::Result<u32> {
        const DEFAULT_KEY: &str = "__default__";
        if let Some(&index) = self.by_path.get(DEFAULT_KEY) {
            return Ok(index);
        }

        let index = u32::try_from(self.materials.len())?;
        self.materials.push(Material::default());
        self.by_path.insert(DEFAULT_KEY.to_string(), index);
        Ok(index)
    }
}

/// The path of the material bound to `prim` via `rel material:binding`, if any. Composed
/// relationship targets arrive as a `PathListOp` (list-edited), so flatten them rather than
/// expecting a plain `PathVec`.
fn material_binding(stage: &Stage, prim_path: &str) -> anyhow::Result<Option<String>> {
    let rel = sdf::path(prim_path)?.append_property("material:binding")?;
    let paths = match stage.field::<Value>(rel, "targetPaths")? {
        Some(Value::PathListOp(op)) => op.flatten(),
        Some(Value::PathVec(v)) => v,
        _ => Vec::new(),
    };
    Ok(paths.first().map(|p| p.as_str().to_string()))
}

fn read_material(stage: &Stage, material_path: &str) -> anyhow::Result<Material> {
    let mat = sdf::path(material_path)?;
    let mut material = Material::default();

    for child in stage.prim_children(mat.clone())? {
        let shader = mat.append_path(child.as_str())?;
        apply_shader_inputs(stage, &shader, &mut material)?;
    }

    Ok(material)
}

fn apply_shader_inputs(
    stage: &Stage,
    shader: &sdf::Path,
    material: &mut Material,
) -> anyhow::Result<()> {
    if let Some(color) = read_vec3(stage, shader, "inputs:diffuseColor")? {
        material.base_color = color;
    }
    if let Some(color) = read_vec3(stage, shader, "inputs:baseColor")? {
        material.base_color = color;
    }
    if let Some(color) = read_vec3(stage, shader, "inputs:emissiveColor")? {
        material.emission = color;
    }
    if let Some(color) = read_vec3(stage, shader, "inputs:emissionColor")? {
        material.emission = color;
    }
    if let Some(roughness) = read_f32(stage, shader, "inputs:roughness")? {
        material.roughness = saturate(roughness);
    }
    if let Some(metallic) = read_f32(stage, shader, "inputs:metallic")? {
        material.metallic = saturate(metallic);
    }
    if let Some(ior) = read_f32(stage, shader, "inputs:ior")? {
        material.ior = ior.max(1.0);
    }
    Ok(())
}

fn read_vec3(stage: &Stage, shader: &sdf::Path, attr: &str) -> anyhow::Result<Option<Vec3>> {
    let prop = shader.append_property(attr)?;
    Ok(match stage.field::<Value>(prop, "default")? {
        Some(Value::Vec3f(c)) => Some(Vec3::from_array(c)),
        Some(Value::Vec3d(c)) => Some(glam::DVec3::from_array(c).as_vec3()),
        _ => None,
    })
}

fn read_f32(stage: &Stage, shader: &sdf::Path, attr: &str) -> anyhow::Result<Option<f32>> {
    let prop = shader.append_property(attr)?;
    Ok(match stage.field::<Value>(prop, "default")? {
        Some(Value::Float(v)) => Some(v),
        Some(Value::Double(v)) => Some(v as f32),
        Some(Value::Half(v)) => Some(v.to_f32()),
        Some(Value::Int(v)) => Some(v as f32),
        Some(Value::Int64(v)) => Some(v as f32),
        _ => None,
    })
}

fn saturate(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}
