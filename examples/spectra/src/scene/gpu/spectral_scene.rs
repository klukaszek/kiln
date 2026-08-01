use std::sync::atomic::Ordering;

use glam::Vec3;
use kiln_rhi::Device;

use super::{GpuBsdf, GpuLight, GpuTriangle, GpuUploadBatch, NEXT_GPU_REVISION, SpectralGpuScene};
use crate::scene::spectral::{self, Spd};
use crate::scene::{Material, Scene, Vertex};

pub(super) fn build(
    device: &Device,
    scene: &Scene,
    light_spectrum: &Spd,
) -> anyhow::Result<SpectralGpuScene> {
    validate_scene(scene)?;

    let spectrum = light_spectrum.bake(spectral::DEFAULT_RESOLUTION);
    let fits: Vec<_> = scene
        .materials
        .iter()
        .map(|material| spectral::fit_reflectance(material.base_color))
        .collect();
    report_inaccurate_fits(scene, &fits);

    let bsdfs: Vec<_> = scene.materials.iter().map(bsdf_to_gpu).collect();
    let emission: Vec<_> = scene
        .materials
        .iter()
        .map(|material| {
            if material.is_emissive() {
                spectrum.emission_scale(material.emission)
            } else {
                0.0
            }
        })
        .collect();
    let reflectance = build_reflectance_lut(&fits, &spectrum);
    let triangles: Vec<_> = scene
        .vertices
        .chunks_exact(3)
        .zip(&scene.triangle_materials)
        .map(|(vertices, &material_id)| {
            let (normal, area) = triangle_normal_area(vertices);
            GpuTriangle {
                normal_area: normal.extend(area),
                material_id,
                emission: emission[material_id as usize],
                _pad: [0.0; 2],
            }
        })
        .collect();
    let lights: Vec<_> = scene
        .triangle_materials
        .iter()
        .enumerate()
        .filter(|&(_, &id)| scene.materials[id as usize].is_emissive())
        .map(|(triangle, &id)| {
            make_light(
                &scene.vertices[triangle * 3..triangle * 3 + 3],
                emission[id as usize],
            )
        })
        .collect();

    let spectrum_len = u32::try_from(spectrum.texels.len())?;
    let light_count = u32::try_from(lights.len())?;
    let material_count = u32::try_from(bsdfs.len())?;
    let mut uploads = GpuUploadBatch::new(device);
    uploads.upload(&spectrum.texels)?;
    uploads.upload(&spectrum.lambda_texels)?;
    uploads.upload(&bsdfs)?;
    uploads.upload(&reflectance)?;
    uploads.upload(&triangles)?;
    uploads.upload(&lights)?;
    let [
        spectrum_buffer,
        lambda_buffer,
        bsdf_buffer,
        reflectance_buffer,
        triangle_buffer,
        light_buffer,
    ] = uploads.finish()?;

    eprintln!(
        "spectral gpu scene: {} triangles, {} materials, {} emissive triangles, light spectrum {} ({} texels)",
        triangles.len(),
        bsdfs.len(),
        lights.len(),
        spectrum.name,
        spectrum.texels.len(),
    );

    Ok(SpectralGpuScene {
        triangle_buffer,
        bsdf_buffer,
        light_buffer,
        spectrum_buffer,
        lambda_buffer,
        reflectance_buffer,
        spectrum_len,
        light_count,
        material_count,
        revision: NEXT_GPU_REVISION.fetch_add(1, Ordering::Relaxed),
    })
}

fn validate_scene(scene: &Scene) -> anyhow::Result<()> {
    let triangle_count = scene.triangle_count();
    anyhow::ensure!(triangle_count > 0, "scene has no triangles");
    anyhow::ensure!(
        scene.vertices.len().is_multiple_of(3) && scene.triangle_materials.len() == triangle_count,
        "triangle material count does not match geometry"
    );
    anyhow::ensure!(
        scene
            .triangle_materials
            .iter()
            .all(|&material| (material as usize) < scene.materials.len()),
        "triangle references an invalid material"
    );
    Ok(())
}

fn report_inaccurate_fits(scene: &Scene, fits: &[spectral::ReflectanceSpectrum]) {
    for (material, fit) in scene.materials.iter().zip(fits) {
        if fit.fit_error > 0.01 {
            eprintln!(
                "spectral fit for albedo {:?} off by {:.3} (moments {:?})",
                material.base_color, fit.fit_error, fit.trig_moments
            );
        }
    }
}

fn make_light(vertices: &[Vertex], emission: f32) -> GpuLight {
    let p0 = vertices[0].pos.truncate();
    let edge1 = vertices[1].pos.truncate() - p0;
    let edge2 = vertices[2].pos.truncate() - p0;
    let (normal, area) = triangle_normal_area(vertices);
    GpuLight {
        p0_emission: p0.extend(emission),
        edge1_area: edge1.extend(area),
        edge2: edge2.extend(0.0),
        normal: normal.extend(0.0),
    }
}

fn triangle_normal_area(vertices: &[Vertex]) -> (Vec3, f32) {
    let p0 = vertices[0].pos.truncate();
    let cross = (vertices[1].pos.truncate() - p0).cross(vertices[2].pos.truncate() - p0);
    let twice_area = cross.length();
    (cross / twice_area.max(f32::MIN_POSITIVE), 0.5 * twice_area)
}

fn bsdf_to_gpu(material: &Material) -> GpuBsdf {
    let roughness = material.roughness.clamp(0.045, 1.0);
    let alpha = roughness * roughness;
    let ior = material.ior.max(1.0001);
    let f0_root = (ior - 1.0) / (ior + 1.0);
    let f0_dielectric = f0_root * f0_root;
    let diffuse_lum = material.base_color.dot(Vec3::new(0.2126, 0.7152, 0.0722));
    let f0_lum = f0_dielectric * (1.0 - material.metallic) + diffuse_lum * material.metallic;
    let diff_weight = (1.0 - material.metallic) * diffuse_lum;
    GpuBsdf {
        alpha,
        alpha2: alpha * alpha,
        metallic: material.metallic,
        f0_dielectric,
        spec_prob: (f0_lum / (f0_lum + diff_weight).max(1e-4)).clamp(0.05, 0.95),
        _pad: [0.0; 3],
    }
}

fn build_reflectance_lut(
    fits: &[spectral::ReflectanceSpectrum],
    light: &spectral::EmissionSpectrum,
) -> Vec<f32> {
    let table_len = light.texels.len();
    debug_assert_eq!(light.lambda_texels.len(), table_len);
    let mut lut = Vec::with_capacity(fits.len() * table_len * 2);
    for fit in fits {
        let lagranges = fit.lagranges.map(f64::from);
        for table in [&light.texels, &light.lambda_texels] {
            lut.extend(
                table
                    .iter()
                    .map(|texel| spectral::eval_reflectance(f64::from(texel.x), lagranges) as f32),
            );
        }
    }
    lut
}
