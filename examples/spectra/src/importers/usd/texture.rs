use std::collections::HashMap;
use std::path::{Path, PathBuf};

use openusd::sdf::{self, Value};
use openusd::usd::Stage;

use crate::scene::{Channel, Channels, ColorSpace, Image, ImageId, Texture, TextureId, WrapMode};

use super::{Error, Result};

pub(super) struct TextureLibrary {
    image_ids: HashMap<(PathBuf, ColorSpace), ImageId>,
    texture_ids: HashMap<TextureKey, TextureId>,
    images: Vec<Image>,
    textures: Vec<Texture>,
    search_dirs: Vec<PathBuf>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct TextureKey {
    image: ImageId,
    wrap_u: WrapMode,
    wrap_v: WrapMode,
}

impl TextureLibrary {
    pub(super) fn new(scene_path: &Path, stage: &Stage) -> Self {
        let mut search_dirs = Vec::new();
        if let Some(parent) = scene_path.parent() {
            search_dirs.push(parent.to_owned());
        }
        for identifier in stage.layer_identifiers() {
            let Some(parent) = Path::new(&identifier).parent() else {
                continue;
            };
            if !parent.as_os_str().is_empty() && !search_dirs.iter().any(|dir| dir == parent) {
                search_dirs.push(parent.to_owned());
            }
        }
        Self {
            image_ids: HashMap::new(),
            texture_ids: HashMap::new(),
            images: Vec::new(),
            textures: Vec::new(),
            search_dirs,
        }
    }

    /// Resolve a material connection to a texture and the channel it reads.
    pub(super) fn read(
        &mut self,
        stage: &Stage,
        input: &str,
        source: &str,
    ) -> Result<(TextureId, Channel)> {
        let channel = match source.rsplit_once('.').map(|(_, output)| output) {
            Some("outputs:r" | "outputs:rgb" | "outputs:rgba") => Channel::R,
            Some("outputs:g") => Channel::G,
            Some("outputs:b") => Channel::B,
            Some("outputs:a") => Channel::A,
            Some(other) => {
                return Err(self.unsupported(input, source, format!("unsupported output {other}")));
            }
            None => return Err(self.unsupported(input, source, "no output name".into())),
        };
        let shader = self.resolve_shader(stage, input, source)?;

        let asset = read_string(stage, &shader, "inputs:file")?
            .ok_or_else(|| Error::Invalid(format!("texture shader {shader} has no inputs:file")))?;
        let path = self.resolve(&asset)?;
        let color_space = match read_string(stage, &shader, "inputs:sourceColorSpace")?.as_deref() {
            Some("raw" | "Raw") => ColorSpace::Linear,
            _ => ColorSpace::Srgb,
        };
        let image = self.image(path, color_space)?;
        let key = TextureKey {
            image,
            wrap_u: read_wrap(stage, &shader, "inputs:wrapS")?,
            wrap_v: read_wrap(stage, &shader, "inputs:wrapT")?,
        };
        if let Some(&id) = self.texture_ids.get(&key) {
            return Ok((id, channel));
        }
        let id = TextureId(self.textures.len());
        self.textures.push(Texture {
            image: key.image,
            wrap_u: key.wrap_u,
            wrap_v: key.wrap_v,
        });
        self.texture_ids.insert(key, id);
        Ok((id, channel))
    }

    /// Follow a material input's connection to the `UsdUVTexture` that drives it.
    ///
    /// The connection may land on the texture shader directly, or on a `NodeGraph` that forwards
    /// one of its outputs to an inner shader. Exporters that emit a graph per texture produce the
    /// latter, so walk the chain rather than assuming a single hop.
    fn resolve_shader(&self, stage: &Stage, input: &str, source: &str) -> Result<sdf::Path> {
        /// Long enough for graph nesting, short enough that a cycle terminates.
        const MAX_HOPS: usize = 8;

        let mut current = source.to_owned();
        for _ in 0..MAX_HOPS {
            let output = current
                .rsplit_once('.')
                .map(|(_, output)| output)
                .unwrap_or_default();
            if !output.starts_with("outputs:") {
                return Err(self.unsupported(
                    input,
                    source,
                    format!("unsupported output {output}"),
                ));
            }
            let prim = sdf::path(&current)?.prim_path();
            let shader_id = read_string(stage, &prim, "info:id")?.unwrap_or_default();
            if matches!(shader_id.as_str(), "UsdUVTexture" | "ND_UsdUVTexture") {
                return Ok(prim);
            }
            let property = prim.append_property(output)?;
            let Some(next) = super::material::connected_path(stage, property)? else {
                return Err(self.unsupported(input, source, format!("shader {shader_id}")));
            };
            current = next;
        }
        Err(self.unsupported(
            input,
            source,
            format!("connection chain exceeds {MAX_HOPS} hops"),
        ))
    }

    pub(super) fn into_parts(self) -> (Vec<Image>, Vec<Texture>) {
        (self.images, self.textures)
    }

    fn image(&mut self, path: PathBuf, color_space: ColorSpace) -> Result<ImageId> {
        let key = (path, color_space);
        if let Some(&id) = self.image_ids.get(&key) {
            return Ok(id);
        }
        let decoded = image::open(&key.0).map_err(|source| Error::Texture {
            path: key.0.clone(),
            source,
        })?;
        let (width, height) = (decoded.width(), decoded.height());
        // Keep greyscale maps single-channel; widening them to RGBA quadruples a 4K map for
        // nothing, and scene texture sets run to gigabytes.
        let (texels, channels, color_space) = match decoded.color() {
            image::ColorType::L8 | image::ColorType::L16 => (
                decoded.into_luma8().into_raw(),
                Channels::One,
                // Scalar data carries no transfer function whatever the stage claims.
                ColorSpace::Linear,
            ),
            _ => (decoded.into_rgba8().into_raw(), Channels::Four, color_space),
        };
        let id = ImageId(self.images.len());
        self.images.push(Image {
            name: key.0.display().to_string(),
            width,
            height,
            texels,
            channels,
            color_space,
        });
        self.image_ids.insert(key, id);
        Ok(id)
    }

    fn resolve(&self, asset: &str) -> Result<PathBuf> {
        let path = Path::new(asset);
        if path.is_absolute() && path.is_file() {
            return Ok(path.to_owned());
        }
        self.search_dirs
            .iter()
            .map(|directory| directory.join(path))
            .find(|candidate| candidate.is_file())
            .ok_or_else(|| Error::MissingTexture(asset.to_owned()))
    }

    fn unsupported(&self, input: &str, source: &str, reason: String) -> Error {
        Error::UnsupportedTexture {
            input: input.to_owned(),
            connection: source.to_owned(),
            reason,
        }
    }
}

fn read_string(stage: &Stage, shader: &sdf::Path, name: &str) -> Result<Option<String>> {
    let property = shader.append_property(name)?;
    Ok(match stage.field::<Value>(property, "default")? {
        Some(Value::AssetPath(value) | Value::Token(value) | Value::String(value)) => Some(value),
        _ => None,
    })
}

fn read_wrap(stage: &Stage, shader: &sdf::Path, name: &str) -> Result<WrapMode> {
    Ok(match read_string(stage, shader, name)?.as_deref() {
        Some("black") => WrapMode::Black,
        Some("clamp") => WrapMode::Clamp,
        Some("mirror") => WrapMode::Mirror,
        _ => WrapMode::Repeat,
    })
}
