use glam::Vec3;
use openusd::sdf::{self, Value};
use openusd::usd::Stage;
use std::collections::HashMap;

use crate::scene::{
    Illuminant, Image, Material, MaterialId, PrincipledBsdf, ScalarMap, Surface, Texture,
};

use super::texture::TextureLibrary;
use super::{Error, Result};

pub(super) struct MaterialLibrary {
    by_path: HashMap<String, MaterialId>,
    materials: Vec<Material>,
    textures: TextureLibrary,
}

impl MaterialLibrary {
    pub(super) fn new(textures: TextureLibrary) -> Self {
        Self {
            by_path: HashMap::new(),
            materials: vec![Material::named("Default Material")],
            textures,
        }
    }

    pub(super) fn id_for_prim(&mut self, stage: &Stage, prim_path: &str) -> Result<MaterialId> {
        let Some(material_path) = material_binding(stage, prim_path)? else {
            return Ok(MaterialId::DEFAULT);
        };
        if let Some(&id) = self.by_path.get(&material_path) {
            return Ok(id);
        }
        let mut material = read_material(stage, &material_path, &mut self.textures)?;
        material.name = material_path
            .rsplit('/')
            .next()
            .filter(|name| !name.is_empty())
            .unwrap_or("Material")
            .to_owned();
        let id = MaterialId(self.materials.len());
        self.materials.push(material);
        self.by_path.insert(material_path, id);
        Ok(id)
    }

    pub(super) fn into_parts(self) -> (Vec<Material>, Vec<Image>, Vec<Texture>) {
        let (images, textures) = self.textures.into_parts();
        (self.materials, images, textures)
    }

    pub(super) fn materials(&self) -> &[Material] {
        &self.materials
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

fn read_material(
    stage: &Stage,
    material_path: &str,
    textures: &mut TextureLibrary,
) -> Result<Material> {
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
    read_base_color(
        stage,
        &shader_path,
        "inputs:diffuseColor",
        &mut surface,
        textures,
    )?;
    read_base_color(
        stage,
        &shader_path,
        "inputs:baseColor",
        &mut surface,
        textures,
    )?;
    // UsdPreviewSurface states emission as a bare colour with no unit, which lands on the
    // renderer's native quantity at unit intensity. The inspector can restate it in lumens or
    // watts afterwards; that only changes how the same emitter is authored, not what it emits.
    for input in ["inputs:emissiveColor", "inputs:emissionColor"] {
        if let Some(color) = read_vec3(stage, &shader_path, input)? {
            result.emission = Illuminant::luminance(color, 1.0);
        }
    }
    read_scalar(
        stage,
        &shader_path,
        "inputs:roughness",
        textures,
        &mut surface.roughness,
        &mut surface.roughness_map,
    )?;
    read_scalar(
        stage,
        &shader_path,
        "inputs:metallic",
        textures,
        &mut surface.metallic,
        &mut surface.metallic_map,
    )?;
    if let Some(value) = read_f32(stage, &shader_path, "inputs:ior")? {
        surface.ior = value;
    }
    result.surface = Surface::Principled(surface);
    for input in ["inputs:emissiveColor", "inputs:emissionColor", "inputs:ior"] {
        reject_connected_input(stage, &shader_path, input)?;
    }
    Ok(result)
}

fn read_base_color(
    stage: &Stage,
    shader: &sdf::Path,
    input: &str,
    surface: &mut PrincipledBsdf,
    textures: &mut TextureLibrary,
) -> Result<()> {
    let property = shader.append_property(input)?;
    if let Some(source) = connected_path(stage, property)? {
        surface.base_color = Vec3::ONE;
        surface.base_color_map = Some(textures.read(stage, input, &source)?.0);
    } else if let Some(color) = read_vec3(stage, shader, input)? {
        surface.base_color = color;
        surface.base_color_map = None;
    }
    Ok(())
}

/// A `UsdPreviewSurface` scalar is either authored inline or read from one channel of a texture.
fn read_scalar(
    stage: &Stage,
    shader: &sdf::Path,
    input: &str,
    textures: &mut TextureLibrary,
    value: &mut f32,
    map: &mut Option<ScalarMap>,
) -> Result<()> {
    let property = shader.append_property(input)?;
    if let Some(source) = connected_path(stage, property)? {
        let (texture, channel) = textures.read(stage, input, &source)?;
        *map = Some(ScalarMap { texture, channel });
    } else if let Some(scalar) = read_f32(stage, shader, input)? {
        *value = scalar;
        *map = None;
    }
    Ok(())
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

pub(super) fn connected_path(stage: &Stage, property: sdf::Path) -> Result<Option<String>> {
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

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::scene::TextureId;

    #[test]
    fn loads_preview_surface_uv_texture() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!("spectra-texture-{nonce}"));
        std::fs::create_dir(&directory).unwrap();
        let texture_path = directory.join("wood.png");
        image::RgbaImage::from_pixel(2, 2, image::Rgba([128, 64, 32, 255]))
            .save(&texture_path)
            .unwrap();
        let stage_path = directory.join("material.usda");
        std::fs::write(
            &stage_path,
            r#"#usda 1.0
def Xform "Mesh" {
    rel material:binding = </Mat>
}
def Material "Mat" {
    token outputs:surface.connect = </Mat/Preview.outputs:surface>
    def Shader "Preview" {
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor.connect = </Mat/Image.outputs:rgb>
        token outputs:surface
    }
    def Shader "Image" {
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @wood.png@
        token inputs:sourceColorSpace = "sRGB"
        float3 outputs:rgb
    }
}
"#,
        )
        .unwrap();

        let stage = Stage::open(stage_path.to_str().unwrap()).unwrap();
        let textures = TextureLibrary::new(&stage_path, &stage);
        let mut library = MaterialLibrary::new(textures);
        let material_id = library.id_for_prim(&stage, "/Mesh").unwrap();
        let (materials, images, textures) = library.into_parts();
        let Surface::Principled(surface) = materials[material_id.0].surface else {
            panic!("expected principled material");
        };
        assert_eq!(surface.base_color_map, Some(TextureId(0)));
        assert_eq!(images.len(), 1);
        assert_eq!(textures.len(), 1);
        assert_eq!((images[0].width, images[0].height), (2, 2));

        std::fs::remove_dir_all(directory).unwrap();
    }
}
