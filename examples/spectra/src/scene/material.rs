//! Renderer-neutral surface and emission descriptions.

use std::sync::LazyLock;

use glam::Vec3;

use super::Illuminant;

/// Renderer-neutral principled surface parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrincipledBsdf {
    /// Constant tint, multiplied by `base_color_map` when one is bound.
    pub base_color: Vec3,
    pub base_color_map: Option<TextureId>,
    /// Fallback used wherever `roughness_map` is absent.
    pub roughness: f32,
    pub roughness_map: Option<ScalarMap>,
    pub metallic: f32,
    pub metallic_map: Option<ScalarMap>,
    pub ior: f32,
}

/// A scalar material input read from one channel of a texture.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScalarMap {
    pub texture: TextureId,
    pub channel: Channel,
}

/// Which channel a [`ScalarMap`] reads.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Channel {
    #[default]
    R,
    G,
    B,
    A,
}

impl Channel {
    pub fn index(self) -> u32 {
        match self {
            Self::R => 0,
            Self::G => 1,
            Self::B => 2,
            Self::A => 3,
        }
    }
}

impl Default for PrincipledBsdf {
    fn default() -> Self {
        Self {
            base_color: Vec3::splat(0.8),
            base_color_map: None,
            roughness_map: None,
            metallic_map: None,
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
///
/// Scalar maps stay single-channel rather than being widened to RGBA. A 4K roughness map is 16 MB
/// as one channel and 64 MB as four, and a scene can carry dozens.
#[derive(Clone)]
pub struct Image {
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// Tightly packed, `channels` bytes per texel.
    pub texels: Vec<u8>,
    pub channels: Channels,
    pub color_space: ColorSpace,
}

/// How many channels an [`Image`] stores per texel.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Channels {
    /// Scalar data: roughness, metalness, occlusion. Always linear.
    One,
    Four,
}

impl Channels {
    pub fn count(self) -> usize {
        match self {
            Self::One => 1,
            Self::Four => 4,
        }
    }
}

/// An image combined with renderer-neutral sampling state.
#[derive(Clone, Copy, Debug)]
pub struct Texture {
    pub image: ImageId,
    pub wrap_u: WrapMode,
    pub wrap_v: WrapMode,
}

impl Image {
    /// Mean linear-sRGB colour, used as a material's representative reflectance when its base
    /// colour comes from a texture. A single-channel image is read as grey.
    ///
    /// Sampled on a stride rather than exhaustively. This produces one colour for one spectral fit
    /// per material, and a 4K map holds sixteen million texels; the difference between a few
    /// thousand well-spread samples and all of them is far below the fit error we already tolerate.
    pub fn average_color(&self) -> Vec3 {
        const TARGET_SAMPLES: usize = 8192;

        let stride = self.channels.count();
        let texels = self.texels.len() / stride;
        if texels == 0 {
            return Vec3::ZERO;
        }
        let step = texels.div_ceil(TARGET_SAMPLES).max(1);
        let to_linear: &[f32; 256] = match self.color_space {
            ColorSpace::Linear => &LINEAR_FROM_U8,
            ColorSpace::Srgb => &LINEAR_FROM_SRGB_U8,
        };

        let mut sum = Vec3::ZERO;
        let mut count = 0u32;
        for index in (0..texels).step_by(step) {
            let texel = &self.texels[index * stride..];
            sum += match self.channels {
                Channels::One => Vec3::splat(to_linear[texel[0] as usize]),
                Channels::Four => Vec3::new(
                    to_linear[texel[0] as usize],
                    to_linear[texel[1] as usize],
                    to_linear[texel[2] as usize],
                ),
            };
            count += 1;
        }
        sum / count.max(1) as f32
    }
}

/// Rec. 709 luminance of a linear-sRGB triple.
pub fn luminance(rgb: Vec3) -> f32 {
    rgb.dot(Vec3::new(0.2126, 0.7152, 0.0722))
}

/// Byte-to-linear tables. A texel channel has 256 possible values, so the sRGB transfer function
/// becomes a lookup instead of a `powf` per channel per texel.
static LINEAR_FROM_SRGB_U8: LazyLock<[f32; 256]> =
    LazyLock::new(|| std::array::from_fn(|i| srgb_to_linear(i as f32 / 255.0)));
static LINEAR_FROM_U8: LazyLock<[f32; 256]> =
    LazyLock::new(|| std::array::from_fn(|i| i as f32 / 255.0));

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
