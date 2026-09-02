//! Renderer-neutral surface and emission descriptions.

use glam::Vec3;

use super::Illuminant;

/// Renderer-neutral principled surface parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrincipledBsdf {
    pub base_color: Vec3,
    pub base_color_texture: Option<TextureId>,
    pub roughness: f32,
    pub metallic: f32,
    pub ior: f32,
}

impl Default for PrincipledBsdf {
    fn default() -> Self {
        Self {
            base_color: Vec3::splat(0.8),
            base_color_texture: None,
            roughness: 0.5,
            metallic: 0.0,
            ior: 1.5,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TextureId(pub usize);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ImageId(pub usize);

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum ColorSpace {
    Linear,
    #[default]
    Srgb,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum WrapMode {
    Black,
    Clamp,
    Mirror,
    #[default]
    Repeat,
}

/// Decoded renderer-neutral image data.
#[derive(Clone)]
pub struct Image {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
    pub color_space: ColorSpace,
}

/// An image combined with renderer-neutral sampling state.
#[derive(Clone, Copy, Debug)]
pub struct Texture {
    pub image: ImageId,
    pub wrap_u: WrapMode,
    pub wrap_v: WrapMode,
}

impl Image {
    pub fn average_color(&self) -> Vec3 {
        let mut sum = Vec3::ZERO;
        for pixel in self.rgba8.chunks_exact(4) {
            let rgb = Vec3::new(pixel[0] as f32, pixel[1] as f32, pixel[2] as f32) / 255.0;
            sum += match self.color_space {
                ColorSpace::Linear => rgb,
                ColorSpace::Srgb => rgb.map(srgb_to_linear),
            };
        }
        sum / (self.rgba8.len() / 4) as f32
    }
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// Surface models understood by the renderer-independent scene IR.
///
/// Backends may lower these descriptions to their own parameter blocks. Keeping the model as an
/// enum leaves room for materials that cannot be represented by a single principled parameter
/// set without making the whole scene generic over a Rust type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Surface {
    Principled(PrincipledBsdf),
    Diffuse { albedo: Vec3 },
    Dielectric { ior: f32, roughness: f32 },
    Conductor { eta: Vec3, k: Vec3, roughness: f32 },
}

impl Default for Surface {
    fn default() -> Self {
        Self::Principled(PrincipledBsdf::default())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Material {
    pub name: String,
    pub surface: Surface,
    /// The emitter bound to every surface using this material. Emissive materials are how most
    /// exported USD scenes light themselves, so this is the same [`Illuminant`] an analytic light
    /// carries rather than a bare emissive colour.
    pub emission: Illuminant,
}

impl Material {
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    pub fn is_emissive(&self) -> bool {
        self.emission.emits()
    }
}

impl Default for Material {
    fn default() -> Self {
        Self {
            name: "Material".into(),
            surface: Surface::default(),
            emission: Illuminant::dark(),
        }
    }
}
