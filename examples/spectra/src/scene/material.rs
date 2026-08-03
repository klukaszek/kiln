//! Renderer-neutral surface and emission descriptions.

use glam::Vec3;

/// Renderer-neutral principled surface parameters.
#[derive(Clone, Copy, Debug)]
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
#[derive(Clone, Copy, Debug)]
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

impl Surface {
    /// A deliberately approximate display color for the unlit raster preview.
    ///
    /// Transport backends must match on [`Surface`] directly instead of using this preview-only
    /// representation.
    pub fn preview_color(self) -> Vec3 {
        match self {
            Self::Principled(surface) => surface.base_color,
            Self::Diffuse { albedo } => albedo,
            Self::Dielectric { .. } => Vec3::ONE,
            Self::Conductor { eta, k, .. } => {
                let eta_minus_one = eta - Vec3::ONE;
                let eta_plus_one = eta + Vec3::ONE;
                let k_squared = k * k;
                ((eta_minus_one * eta_minus_one + k_squared)
                    / (eta_plus_one * eta_plus_one + k_squared))
                    .clamp(Vec3::ZERO, Vec3::ONE)
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Emission {
    pub color: Vec3,
}

impl Default for Emission {
    fn default() -> Self {
        Self { color: Vec3::ZERO }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Material {
    pub surface: Surface,
    pub emission: Emission,
}

impl Material {
    pub fn is_emissive(&self) -> bool {
        self.emission.color.max_element() > 0.0
    }

    pub fn preview_color(&self) -> Vec3 {
        if self.is_emissive() {
            self.emission.color
        } else {
            self.surface.preview_color()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conductor_preview_uses_normal_incidence_fresnel() {
        let surface = Surface::Conductor {
            eta: Vec3::splat(0.2),
            k: Vec3::splat(3.0),
            roughness: 0.2,
        };
        assert!(surface.preview_color().min_element() > 0.9);
    }
}
