//! Material lowering: surface parameters to GGX BSDF records, and base colours to the smooth
//! reflectance spectra the spectral integrator needs.

use glam::Vec3;

use crate::render;
use crate::scene::{Scene, Surface};

use crate::render::Error;
use crate::scene::{Material, ScalarMap};

use super::super::spectrum;
use super::records::{GpuMaterial, NO_MAP};

pub(super) struct MaterialArrays {
    pub(super) materials: Vec<GpuMaterial>,
    pub(super) reflectance: Vec<f32>,
    pub(super) texture_basis: Vec<f32>,
}

pub(super) fn build_material_arrays(
    scene: &Scene,
    image_averages: &[Vec3],
    spectrum: &spectrum::EmissionSpectrum,
) -> render::Result<MaterialArrays> {
    let mut fits = Vec::with_capacity(scene.materials.len());
    let mut materials = Vec::with_capacity(scene.materials.len());
    let mut any_textured = false;
    for material in &scene.materials {
        // The spectral reflectance fit needs one colour per material, so a textured base colour is
        // represented by its mean; the shader applies the per-texel detail on top.
        let representative = representative_base_color(material.surface, scene, image_averages);
        let fit = spectrum::fit_reflectance(representative);
        if fit.fit_error > 0.01 {
            eprintln!(
                "spectral fit for albedo {:?} off by {:.3} (moments {:?})",
                representative_surface_color(material.surface),
                fit.fit_error,
                fit.trig_moments
            );
        }
        fits.push(fit);
        let packed = pack_material(material)?;
        any_textured |= packed.base_color_map != NO_MAP
            || packed.roughness_map != NO_MAP
            || packed.metallic_map != NO_MAP;
        materials.push(packed);
    }

    let texture_basis = if any_textured {
        build_texture_basis(spectrum)
    } else {
        Vec::new()
    };
    Ok(MaterialArrays {
        materials,
        reflectance: build_reflectance_lut(&fits, spectrum),
        texture_basis,
    })
}

/// Pack a scene material into its device record. Nothing is derived here: the GGX coefficients
/// depend on inputs that may vary per texel, so the shader computes them at each hit.
fn pack_material(material: &Material) -> render::Result<GpuMaterial> {
    let Surface::Principled(surface) = material.surface else {
        return Err(Error::Unsupported(
            "path tracer requires principled materials",
        ));
    };
    Ok(GpuMaterial {
        base_color: surface.base_color.extend(1.0),
        roughness: surface.roughness,
        metallic: surface.metallic,
        ior: surface.ior,
        _pad: 0.0,
        base_color_map: surface
            .base_color_map
            .map_or(NO_MAP, |texture| texture.0 as u32),
        roughness_map: pack_scalar_map(surface.roughness_map),
        metallic_map: pack_scalar_map(surface.metallic_map),
        _pad2: 0,
    })
}

/// Binding index in the low 24 bits, channel in the high byte.
fn pack_scalar_map(map: Option<ScalarMap>) -> u32 {
    map.map_or(NO_MAP, |map| {
        (map.texture.0 as u32 & 0x00ff_ffff) | (map.channel.index() << 24)
    })
}

fn representative_surface_color(surface: Surface) -> Vec3 {
    match surface {
        Surface::Principled(surface) => surface.base_color,
        Surface::Diffuse { albedo } => albedo,
        Surface::Dielectric { .. } => Vec3::ONE,
        Surface::Conductor { eta, k, .. } => {
            let eta_minus_one = eta - Vec3::ONE;
            let eta_plus_one = eta + Vec3::ONE;
            let k_squared = k * k;
            ((eta_minus_one * eta_minus_one + k_squared)
                / (eta_plus_one * eta_plus_one + k_squared))
                .clamp(Vec3::ZERO, Vec3::ONE)
        }
    }
}

fn representative_base_color(surface: Surface, scene: &Scene, image_averages: &[Vec3]) -> Vec3 {
    let Surface::Principled(surface) = surface else {
        return representative_surface_color(surface);
    };
    match surface.base_color_map {
        Some(texture) => {
            let image = scene.textures[texture.0].image;
            surface.base_color * image_averages[image.0]
        }
        None => surface.base_color,
    }
}

fn build_texture_basis(light: &spectrum::EmissionSpectrum) -> Vec<f32> {
    let fits = texture_basis_colors().map(spectrum::fit_reflectance);
    build_reflectance_lut(&fits, light)
}

pub(super) fn texture_basis_colors() -> [Vec3; 7] {
    [
        Vec3::ONE,
        Vec3::new(0.0, 1.0, 1.0),
        Vec3::new(1.0, 0.0, 1.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
    ]
}

fn build_reflectance_lut(
    fits: &[spectrum::ReflectanceSpectrum],
    light: &spectrum::EmissionSpectrum,
) -> Vec<f32> {
    let table_len = light.texels.len();
    let mut lut = Vec::with_capacity(fits.len() * table_len);
    for fit in fits {
        let lagranges = fit.lagranges.map(f64::from);
        lut.extend(
            light
                .texels
                .iter()
                .map(|texel| spectrum::eval_reflectance(f64::from(texel.x), lagranges) as f32),
        );
    }
    lut
}
