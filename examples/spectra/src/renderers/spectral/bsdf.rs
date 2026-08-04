//! Lowering from scene materials to the principled GGX model implemented by the spectral shader.

use glam::Vec3;

use crate::base::renderer::{self, Error};
use crate::base::scene::{Material, Surface, TextureId};

const MIN_GGX_ROUGHNESS: f32 = 0.045;

#[derive(Clone, Copy, Debug)]
pub(super) struct PrincipledGgxParams {
    pub(super) base_color: Vec3,
    pub(super) texture_factor: Vec3,
    pub(super) base_color_texture: Option<TextureId>,
    pub(super) alpha: f32,
    pub(super) alpha_squared: f32,
    pub(super) metallic: f32,
    pub(super) dielectric_f0: f32,
    pub(super) specular_probability: f32,
}

pub(super) fn lower(
    material: &Material,
    representative_base_color: Vec3,
) -> renderer::Result<PrincipledGgxParams> {
    let Surface::Principled(surface) = material.surface else {
        return Err(Error::Unsupported(
            "path tracer requires principled materials",
        ));
    };

    // Avoid a singular GGX distribution while preserving the authored value everywhere else.
    let base_color = representative_base_color;
    let roughness = surface.roughness.max(MIN_GGX_ROUGHNESS);
    let alpha = roughness * roughness;
    let ior = surface.ior;
    let f0_root = (ior - 1.0) / (ior + 1.0);
    let dielectric_f0 = f0_root * f0_root;
    let diffuse_luminance = base_color.dot(Vec3::new(0.2126, 0.7152, 0.0722));
    let f0_luminance =
        dielectric_f0 * (1.0 - surface.metallic) + diffuse_luminance * surface.metallic;
    let diffuse_weight = (1.0 - surface.metallic) * diffuse_luminance;

    Ok(PrincipledGgxParams {
        base_color,
        texture_factor: surface.base_color,
        base_color_texture: surface.base_color_texture,
        alpha,
        alpha_squared: alpha * alpha,
        metallic: surface.metallic,
        dielectric_f0,
        specular_probability: (f0_luminance / (f0_luminance + diffuse_weight).max(1e-4))
            .clamp(0.05, 0.95),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unimplemented_surface_models() {
        let material = Material {
            surface: Surface::Diffuse {
                albedo: Vec3::splat(0.5),
            },
            ..Material::default()
        };
        assert!(lower(&material, Vec3::splat(0.5)).is_err());
    }
}
