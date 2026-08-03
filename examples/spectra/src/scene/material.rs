//! Renderer-neutral surface and emission descriptions.

use glam::Vec3;

/// Renderer-neutral principled surface parameters.
#[derive(Clone, Copy, Debug)]
pub struct PrincipledBsdf {
    pub base_color: Vec3,
    pub roughness: f32,
    pub metallic: f32,
    pub ior: f32,
}

impl Default for PrincipledBsdf {
    fn default() -> Self {
        Self {
            base_color: Vec3::splat(0.8),
            roughness: 0.5,
            metallic: 0.0,
            ior: 1.5,
        }
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
