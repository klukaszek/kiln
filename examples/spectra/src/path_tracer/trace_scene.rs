//! Device representation of a scene used by spectral tracing.
//!
//! Nothing in this module is part of the shared scene model: the ray geometry, acceleration
//! structure, BSDF parameters, light list, and spectral tables exist only for this backend.

use glam::Vec4;
use kiln_rhi::{Device, gpu_struct};

use crate::render::{self, Error, GpuArray, GpuUploadBatch};
use crate::scene::Scene;
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
        material_id: u32,
        emission: f32,
        _pad: [f32; 2],
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
        let mut fits = Vec::with_capacity(scene.materials.len());
        let mut bsdfs = Vec::with_capacity(scene.materials.len());
        let mut emission = Vec::with_capacity(scene.materials.len());
        for material in &scene.materials {
            let lowered = bsdf::lower(material)?;
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
            bsdfs.push(bsdf_to_gpu(lowered));
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
            triangles.push(GpuTriangle {
                normal_area: normal.extend(area),
                material_id: material.0 as u32,
                emission,
                _pad: [0.0; 2],
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

        let uploaded = (|| {
            let mut uploads = GpuUploadBatch::new(device);
            uploads.upload(&spectrum.texels)?;
            uploads.upload(&spectrum.lambda_texels)?;
            uploads.upload(&bsdfs)?;
            uploads.upload(&reflectance)?;
            uploads.upload(&triangles)?;
            uploads.upload(&lights)?;
            uploads.finish()
        })();
        let [
            spectrum_gpu,
            lambda_gpu,
            bsdf_gpu,
            reflectance_gpu,
            triangle_gpu,
            light_gpu,
        ] = match uploaded {
            Ok(value) => value,
            Err(error) => {
                accel.destroy(device);
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
    }
}

fn bsdf_to_gpu(params: PrincipledGgxParams) -> GpuBsdf {
    GpuBsdf {
        alpha: params.alpha,
        alpha2: params.alpha_squared,
        metallic: params.metallic,
        f0_dielectric: params.dielectric_f0,
        spec_prob: params.specular_probability,
        _pad: [0.0; 3],
    }
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
