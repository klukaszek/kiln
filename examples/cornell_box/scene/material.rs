use std::collections::HashMap;

use openusd::sdf::{self, Value};
use openusd::usd::Stage;

const DEFAULT_BASE_COLOR: [f32; 3] = [0.8, 0.8, 0.8];
const DEFAULT_SPECULAR_COLOR: [f32; 3] = [0.0, 0.0, 0.0];
const DEFAULT_EMISSION: [f32; 3] = [0.0, 0.0, 0.0];
const DEFAULT_OPACITY: f32 = 1.0;
const DEFAULT_ROUGHNESS: f32 = 0.5;
const DEFAULT_METALLIC: f32 = 0.0;
const DEFAULT_IOR: f32 = 1.5;
const DEFAULT_CLEARCOAT: f32 = 0.0;
const DEFAULT_CLEARCOAT_ROUGHNESS: f32 = 0.01;
const DEFAULT_OPACITY_THRESHOLD: f32 = 0.0;

/// Render material description pulled from UsdPreviewSurface-style shader inputs.
///
/// The current mesh preview collapses this to a single RGB value via [`Self::raster_color`],
/// but the fields are kept close to the authored surface parameters so the ray-query path can
/// build a real BSDF/emitter representation without reparsing USD.
#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub base_color: [f32; 3],
    pub specular_color: [f32; 3],
    pub emission: [f32; 3],
    pub opacity: f32,
    pub roughness: f32,
    pub metallic: f32,
    pub ior: f32,
    pub clearcoat: f32,
    pub clearcoat_roughness: f32,
    pub opacity_threshold: f32,
    pub use_specular_workflow: bool,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: DEFAULT_BASE_COLOR,
            specular_color: DEFAULT_SPECULAR_COLOR,
            emission: DEFAULT_EMISSION,
            opacity: DEFAULT_OPACITY,
            roughness: DEFAULT_ROUGHNESS,
            metallic: DEFAULT_METALLIC,
            ior: DEFAULT_IOR,
            clearcoat: DEFAULT_CLEARCOAT,
            clearcoat_roughness: DEFAULT_CLEARCOAT_ROUGHNESS,
            opacity_threshold: DEFAULT_OPACITY_THRESHOLD,
            use_specular_workflow: false,
        }
    }
}

impl Material {
    pub fn raster_color(&self) -> [f32; 3] {
        if self.is_emissive() {
            self.emission
        } else {
            self.base_color
        }
    }

    pub fn is_emissive(&self) -> bool {
        self.emission.iter().any(|&v| v > 0.0)
    }
}

#[derive(Default)]
pub struct MaterialCache {
    by_path: HashMap<String, u32>,
    materials: Vec<Material>,
}

impl MaterialCache {
    pub fn material_index_for_prim(
        &mut self,
        stage: &Stage,
        prim_path: &str,
    ) -> anyhow::Result<u32> {
        let Some(material_path) = material_binding(stage, prim_path)? else {
            return Ok(self.default_material_index());
        };

        if let Some(&index) = self.by_path.get(&material_path) {
            return Ok(index);
        }

        let material = read_material(stage, &material_path).unwrap_or_default();
        let index = self.materials.len() as u32;
        self.materials.push(material);
        self.by_path.insert(material_path, index);
        Ok(index)
    }

    pub fn material(&self, index: u32) -> Material {
        self.materials
            .get(index as usize)
            .copied()
            .unwrap_or_default()
    }

    pub fn into_materials(mut self) -> Vec<Material> {
        if self.materials.is_empty() {
            self.materials.push(Material::default());
        }
        self.materials
    }

    fn default_material_index(&mut self) -> u32 {
        const DEFAULT_KEY: &str = "__default__";
        if let Some(&index) = self.by_path.get(DEFAULT_KEY) {
            return index;
        }

        let index = self.materials.len() as u32;
        self.materials.push(Material::default());
        self.by_path.insert(DEFAULT_KEY.to_string(), index);
        index
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
    if let Some(color) = read_vec3(stage, shader, "inputs:specularColor")? {
        material.specular_color = color;
    }
    if let Some(color) = read_vec3(stage, shader, "inputs:emissiveColor")? {
        material.emission = color;
    }
    if let Some(color) = read_vec3(stage, shader, "inputs:emissionColor")? {
        material.emission = color;
    }
    if let Some(opacity) = read_f32(stage, shader, "inputs:opacity")? {
        material.opacity = saturate(opacity);
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
    if let Some(clearcoat) = read_f32(stage, shader, "inputs:clearcoat")? {
        material.clearcoat = saturate(clearcoat);
    }
    if let Some(roughness) = read_f32(stage, shader, "inputs:clearcoatRoughness")? {
        material.clearcoat_roughness = saturate(roughness);
    }
    if let Some(threshold) = read_f32(stage, shader, "inputs:opacityThreshold")? {
        material.opacity_threshold = saturate(threshold);
    }
    if let Some(use_specular_workflow) = read_bool(stage, shader, "inputs:useSpecularWorkflow")? {
        material.use_specular_workflow = use_specular_workflow;
    }

    Ok(())
}

fn read_vec3(stage: &Stage, shader: &sdf::Path, attr: &str) -> anyhow::Result<Option<[f32; 3]>> {
    let prop = shader.append_property(attr)?;
    Ok(match stage.field::<Value>(prop, "default")? {
        Some(Value::Vec3f(c)) => Some(c),
        Some(Value::Vec3d(c)) => Some([c[0] as f32, c[1] as f32, c[2] as f32]),
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

fn read_bool(stage: &Stage, shader: &sdf::Path, attr: &str) -> anyhow::Result<Option<bool>> {
    let prop = shader.append_property(attr)?;
    Ok(match stage.field::<Value>(prop, "default")? {
        Some(Value::Bool(v)) => Some(v),
        Some(Value::Int(v)) => Some(v != 0),
        Some(Value::Int64(v)) => Some(v != 0),
        _ => None,
    })
}

fn saturate(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}
