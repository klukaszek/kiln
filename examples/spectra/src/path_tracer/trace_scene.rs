//! Device representation of a scene used by spectral tracing.
//!
//! Nothing in this module is part of the shared scene model: the ray geometry, acceleration
//! structure, BSDF parameters, light list, and spectral tables exist only for this backend.

use glam::{Vec2, Vec3, Vec4};
use kiln_rhi::{Device, gpu_struct};

use crate::render::{self, Error, GpuArray, GpuTextureBinding, GpuUploadBatch, TextureResources};
use crate::scene::{Scene, Surface};
use crate::spectrum::{self, Spd};

use super::acceleration::SceneAccel;
use super::bsdf::{self, PrincipledGgxParams};

gpu_struct! {
    pub(super) struct GpuBsdf {
        alpha: f32,
        alpha2: f32,
        metallic: f32,
        f0_dielectric: f32,
        spec_prob: f32,
        texture_id: u32,
        _pad: [f32; 2],
    }
}

gpu_struct! {
    pub(super) struct GpuMaterialTexture {
        factor: Vec4,
        binding_id: u32,
        _pad: [f32; 3],
    }
}

gpu_struct! {
    pub(super) struct GpuLight {
        p0_emission: Vec4,
        edge1_area: Vec4,
        edge2: Vec4,
        normal: Vec4,
    }
}

gpu_struct! {
    pub(super) struct GpuTriangle {
        normal_area: Vec4,
        uv01: Vec4,
        uv2: Vec2,
        material_id: u32,
        emission: f32,
    }
}

pub(super) struct TraceScene {
    pub(super) accel: SceneAccel,
    pub(super) triangles: GpuArray<GpuTriangle>,
    pub(super) bsdfs: GpuArray<GpuBsdf>,
    pub(super) lights: GpuArray<GpuLight>,
    pub(super) spectrum: GpuArray<Vec4>,
    pub(super) lambda: GpuArray<Vec4>,
    pub(super) reflectance: GpuArray<f32>,
    pub(super) material_textures: GpuArray<GpuMaterialTexture>,
    pub(super) texture_bindings: GpuArray<GpuTextureBinding>,
    pub(super) texture_basis: GpuArray<f32>,
    textures: TextureResources,
}

impl TraceScene {
    pub(super) fn build(
        device: &Device,
        scene: &Scene,
        light_spectrum: &Spd,
    ) -> render::Result<Self> {
        if scene.triangle_count() > (u32::MAX as usize) / 3 {
            return Err(Error::Capacity("trace geometry exceeds u32 indexing"));
        }
        let spectrum = light_spectrum.bake(spectrum::DEFAULT_RESOLUTION);
        let image_averages = scene
            .images
            .iter()
            .map(|image| image.average_color())
            .collect::<Vec<_>>();
        let mut fits = Vec::with_capacity(scene.materials.len());
        let mut lowered_bsdfs = Vec::with_capacity(scene.materials.len());
        let mut emission = Vec::with_capacity(scene.materials.len());
        for material in &scene.materials {
            let representative =
                representative_base_color(material.surface, scene, &image_averages);
            let lowered = bsdf::lower(material, representative)?;
            let fit = spectrum::fit_reflectance(lowered.base_color);
            if fit.fit_error > 0.01 {
                eprintln!(
                    "spectral fit for albedo {:?} off by {:.3} (moments {:?})",
                    material.surface.preview_color(),
                    fit.fit_error,
                    fit.trig_moments
                );
            }
            fits.push(fit);
            lowered_bsdfs.push(lowered);
            emission.push(if material.is_emissive() {
                spectrum.emission_scale(material.emission.color)
            } else {
                0.0
            });
        }
        let reflectance = build_reflectance_lut(&fits, &spectrum);
        let mut triangles = Vec::with_capacity(scene.triangle_count());
        let mut lights = Vec::new();
        for triangle in &scene.geometry.triangles {
            let material = triangle.material;
            let (normal, area) = triangle.geometric_normal_and_area();
            let emission = emission[material.0];
            let uv0 = triangle.vertices[0].uv.unwrap_or_default();
            let uv1 = triangle.vertices[1].uv.unwrap_or_default();
            let uv2 = triangle.vertices[2].uv.unwrap_or_default();
            triangles.push(GpuTriangle {
                normal_area: normal.extend(area),
                uv01: Vec4::new(uv0.x, uv0.y, uv1.x, uv1.y),
                uv2,
                material_id: material.0 as u32,
                emission,
            });
            if scene.materials[material.0].is_emissive() {
                let vertices = &triangle.vertices;
                let p0 = vertices[0].position;
                let edge1 = vertices[1].position - p0;
                let edge2 = vertices[2].position - p0;
                lights.push(GpuLight {
                    p0_emission: p0.extend(emission),
                    edge1_area: edge1.extend(area),
                    edge2: edge2.extend(0.0),
                    normal: normal.extend(0.0),
                });
            }
        }
        let accel = SceneAccel::build(device, scene)?;
        let textures = match TextureResources::upload(device, &scene.images, &scene.textures) {
            Ok(textures) => textures,
            Err(error) => {
                accel.destroy(device);
                return Err(error);
            }
        };
        let texture_bindings = textures.bindings();
        let mut material_textures = Vec::new();
        let mut bsdfs = Vec::with_capacity(lowered_bsdfs.len());
        for params in lowered_bsdfs {
            let texture_id = match params.base_color_texture {
                Some(binding) => {
                    let id = material_textures.len() as u32;
                    material_textures.push(GpuMaterialTexture {
                        factor: params.texture_factor.extend(1.0),
                        binding_id: binding.0 as u32,
                        _pad: [0.0; 3],
                    });
                    id
                }
                None => u32::MAX,
            };
            bsdfs.push(bsdf_to_gpu(params, texture_id));
        }
        let texture_basis = if material_textures.is_empty() {
            Vec::new()
        } else {
            build_texture_basis(&spectrum)
        };

        let uploaded = (|| {
            let mut uploads = GpuUploadBatch::new(device);
            uploads.upload(&spectrum.texels)?;
            uploads.upload(&spectrum.lambda_texels)?;
            uploads.upload(&bsdfs)?;
            uploads.upload(&reflectance)?;
            uploads.upload(&triangles)?;
            uploads.upload(&lights)?;
            uploads.upload(&material_textures)?;
            uploads.upload(texture_bindings)?;
            uploads.upload(&texture_basis)?;
            uploads.finish()
        })();
        let [
            spectrum_gpu,
            lambda_gpu,
            bsdf_gpu,
            reflectance_gpu,
            triangle_gpu,
            light_gpu,
            material_texture_gpu,
            texture_binding_gpu,
            texture_basis_gpu,
        ] = match uploaded {
            Ok(value) => value,
            Err(error) => {
                accel.destroy(device);
                textures.destroy(device);
                return Err(error);
            }
        };

        eprintln!(
            "spectral GPU scene: {} triangles, {} materials, {} emissive triangles, light spectrum {} ({} texels)",
            triangles.len(),
            bsdfs.len(),
            lights.len(),
            spectrum.name,
            spectrum.texels.len()
        );
        Ok(Self {
            accel,
            triangles: GpuArray::new(triangle_gpu, triangles.len()),
            bsdfs: GpuArray::new(bsdf_gpu, bsdfs.len()),
            lights: GpuArray::new(light_gpu, lights.len()),
            spectrum: GpuArray::new(spectrum_gpu, spectrum.texels.len()),
            lambda: GpuArray::new(lambda_gpu, spectrum.lambda_texels.len()),
            reflectance: GpuArray::new(reflectance_gpu, reflectance.len()),
            material_textures: GpuArray::new(material_texture_gpu, material_textures.len()),
            texture_bindings: GpuArray::new(texture_binding_gpu, texture_bindings.len()),
            texture_basis: GpuArray::new(texture_basis_gpu, texture_basis.len()),
            textures,
        })
    }

    pub(super) fn destroy(self, device: &Device) {
        self.accel.destroy(device);
        self.triangles.destroy(device);
        self.bsdfs.destroy(device);
        self.lights.destroy(device);
        self.spectrum.destroy(device);
        self.lambda.destroy(device);
        self.reflectance.destroy(device);
        self.material_textures.destroy(device);
        self.texture_bindings.destroy(device);
        self.texture_basis.destroy(device);
        self.textures.destroy(device);
    }
}

fn bsdf_to_gpu(params: PrincipledGgxParams, texture_id: u32) -> GpuBsdf {
    GpuBsdf {
        alpha: params.alpha,
        alpha2: params.alpha_squared,
        metallic: params.metallic,
        f0_dielectric: params.dielectric_f0,
        spec_prob: params.specular_probability,
        texture_id,
        _pad: [0.0; 2],
    }
}

fn representative_base_color(surface: Surface, scene: &Scene, image_averages: &[Vec3]) -> Vec3 {
    let Surface::Principled(surface) = surface else {
        return surface.preview_color();
    };
    match surface.base_color_texture {
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

fn texture_basis_colors() -> [Vec3; 7] {
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
    let mut lut = Vec::with_capacity(fits.len() * table_len * 2);
    for fit in fits {
        let lagranges = fit.lagranges.map(f64::from);
        for table in [&light.texels, &light.lambda_texels] {
            lut.extend(
                table
                    .iter()
                    .map(|texel| spectrum::eval_reflectance(f64::from(texel.x), lagranges) as f32),
            );
        }
    }
    lut
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hot_path_records_stay_compact() {
        assert_eq!(size_of::<GpuBsdf>(), 32);
        assert_eq!(size_of::<GpuTriangle>(), 48);
        assert_eq!(size_of::<GpuMaterialTexture>(), 32);
    }

    #[test]
    fn texture_basis_fits_rgb_cube_corners() {
        for color in texture_basis_colors() {
            let fit = spectrum::fit_reflectance(color);
            assert!(fit.fit_error < 0.035, "{color:?}: {}", fit.fit_error);
        }
    }
}
