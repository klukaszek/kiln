//! The device-side scene: buffers the trace kernel reads, plus the acceleration structure.
//!
//! Built from a [`Scene`](crate::scene::Scene) snapshot and rebuilt in whole groups when the editor
//! changes something. Each submodule owns one group and knows nothing about the others.

mod accel;
mod geometry;
mod lights;
mod materials;
mod records;
mod tags;
mod textures;

use glam::{Vec3, Vec4};
use kiln_rhi::Device;

use super::upload::{GpuArray, GpuUploadBatch};
use crate::render::{self, Error};
use crate::scene::Scene;
use textures::Textures;

use super::SceneUpdate;
use super::spectrum::{self, Spd};

use accel::SceneAccel;
use geometry::{build_gpu_instances, build_gpu_triangles, build_instance_layout};
use lights::{BuiltLights, LightingArrays, build_lighting_arrays};
use materials::{MaterialArrays, build_material_arrays};

pub(super) use records::{
    GpuEmissiveHit, GpuInstance, GpuLight, GpuMaterial, GpuMeshLightTriangle, GpuTriangle,
};
pub(super) use tags::LightTag;
pub(super) use textures::GpuTextureBinding;

pub(super) struct GpuScene {
    pub(super) accel: SceneAccel,
    pub(super) triangles: GpuArray<GpuTriangle>,
    pub(super) emissive_hits: GpuArray<GpuEmissiveHit>,
    pub(super) instances: GpuArray<GpuInstance>,
    pub(super) materials: GpuArray<GpuMaterial>,
    pub(super) lights: GpuArray<GpuLight>,
    pub(super) mesh_light_triangles: GpuArray<GpuMeshLightTriangle>,
    pub(super) mesh_light_cdf: GpuArray<f32>,
    pub(super) light_spectrum: GpuArray<f32>,
    pub(super) material_emission_spectrum: GpuArray<f32>,
    pub(super) spectrum: GpuArray<Vec4>,
    pub(super) sensor_spectrum: GpuArray<Vec4>,
    pub(super) reflectance: GpuArray<f32>,
    pub(super) texture_bindings: GpuArray<GpuTextureBinding>,
    pub(super) texture_basis: GpuArray<f32>,
    /// Cached because material edits do not change image contents.
    image_averages: Vec<Vec3>,
    baked_spectrum: spectrum::EmissionSpectrum,
    textures: Textures,
}

impl GpuScene {
    /// Rebuild the device buffers a scene edit invalidated.
    ///
    /// Buffer groups are rebuilt whole rather than patched element by element. Every edit calls
    /// `Film::invalidate`, so the trace restarts and re-accumulates thousands of samples; the
    /// milliseconds a targeted patch would save do not pay for the index bookkeeping.
    pub(super) fn update(
        &mut self,
        device: &Device,
        scene: &Scene,
        light_spectrum: &Spd,
        update: SceneUpdate,
    ) -> render::Result<()> {
        // Buffers are replaced outright, so no in-flight frame may still be reading them.
        device.queue().wait_idle();
        if update.topology {
            return self.rebuild_geometry(device, scene, light_spectrum);
        }
        if update.surfaces {
            self.rebuild_materials(device, scene)?;
        }
        // A transform rescales the world-space area of an emissive surface, which sets its radiance.
        if update.emission || update.transforms {
            self.rebuild_lighting(device, scene, light_spectrum)?;
        }
        if update.transforms {
            self.rebuild_instances(device, scene)?;
            self.accel.rebuild_tlas(device, scene)?;
        }
        Ok(())
    }

    /// Emission-dependent buffers: ray triangles, the light list, and the spectral tables.
    fn rebuild_lighting(
        &mut self,
        device: &Device,
        scene: &Scene,
        default_spectrum: &Spd,
    ) -> render::Result<()> {
        let spectrum = default_spectrum.bake(spectrum::DEFAULT_RESOLUTION);
        let LightingArrays {
            emission,
            material_spectrum,
            emissive_hits,
            lights:
                BuiltLights {
                    lights,
                    mesh_triangles,
                    mesh_cdf,
                    spectrum: light_modulation,
                    mesh_ranges: _,
                },
        } = build_lighting_arrays(scene, default_spectrum, &spectrum);
        let triangles = build_gpu_triangles(scene, &emission);

        let mut uploads = GpuUploadBatch::new(device)?;
        let triangles = uploads.upload(&triangles)?;
        let emissive_hits = uploads.upload(&emissive_hits)?;
        let lights = uploads.upload(&lights)?;
        let mesh_triangles = uploads.upload(&mesh_triangles)?;
        let mesh_cdf = uploads.upload(&mesh_cdf)?;
        let light_modulation = uploads.upload(&light_modulation)?;
        let material_spectrum = uploads.upload(&material_spectrum)?;
        uploads.submit()?;

        self.triangles.replace(device, triangles);
        self.emissive_hits.replace(device, emissive_hits);
        self.lights.replace(device, lights);
        self.mesh_light_triangles.replace(device, mesh_triangles);
        self.mesh_light_cdf.replace(device, mesh_cdf);
        self.light_spectrum.replace(device, light_modulation);
        self.material_emission_spectrum
            .replace(device, material_spectrum);
        Ok(())
    }

    /// BSDF parameters and the reflectance tables fitted from them.
    fn rebuild_materials(&mut self, device: &Device, scene: &Scene) -> render::Result<()> {
        let MaterialArrays {
            materials,
            reflectance,
            texture_basis,
        } = build_material_arrays(scene, &self.image_averages, &self.baked_spectrum)?;

        let mut uploads = GpuUploadBatch::new(device)?;
        let materials = uploads.upload(&materials)?;
        let reflectance = uploads.upload(&reflectance)?;
        let texture_basis = uploads.upload(&texture_basis)?;
        uploads.submit()?;

        self.materials.replace(device, materials);
        self.reflectance.replace(device, reflectance);

        self.texture_basis.replace(device, texture_basis);
        Ok(())
    }

    fn rebuild_instances(&mut self, device: &Device, scene: &Scene) -> render::Result<()> {
        let instances = build_gpu_instances(scene, &build_instance_layout(scene));
        let mut uploads = GpuUploadBatch::new(device)?;
        let instances = uploads.upload(&instances)?;
        uploads.submit()?;
        self.instances.replace(device, instances);
        Ok(())
    }

    /// A topology change invalidates everything, including the BLAS geometry.
    fn rebuild_geometry(
        &mut self,
        device: &Device,
        scene: &Scene,
        default_spectrum: &Spd,
    ) -> render::Result<()> {
        if scene.triangle_count() > (u32::MAX as usize) / 3 {
            return Err(Error::Capacity("trace geometry exceeds u32 indexing"));
        }
        let accel = SceneAccel::build(device, scene)?;
        if let Err(error) = self.rebuild_lighting(device, scene, default_spectrum) {
            accel.destroy(device);
            return Err(error);
        }
        if let Err(error) = self.rebuild_instances(device, scene) {
            accel.destroy(device);
            return Err(error);
        }
        std::mem::replace(&mut self.accel, accel).destroy(device);
        Ok(())
    }

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
        let MaterialArrays {
            materials,
            reflectance,
            texture_basis,
        } = build_material_arrays(scene, &image_averages, &spectrum)?;
        let LightingArrays {
            emission,
            material_spectrum: material_emission_spectrum,
            emissive_hits,
            lights:
                BuiltLights {
                    lights,
                    mesh_triangles: mesh_light_triangles,
                    mesh_cdf: mesh_light_cdf,
                    spectrum: light_spectrum_modulation,
                    mesh_ranges: _,
                },
        } = build_lighting_arrays(scene, light_spectrum, &spectrum);
        let triangles = build_gpu_triangles(scene, &emission);
        let instances = build_gpu_instances(scene, &build_instance_layout(scene));
        let accel = SceneAccel::build(device, scene)?;

        eprintln!(
            "spectral GPU scene: {} triangles, {} materials, {} lights, light spectrum {} ({} texels)",
            triangles.len(),
            materials.len(),
            lights.len(),
            spectrum.name,
            spectrum.texels.len()
        );

        // Textures and buffers share one batch, so the whole scene costs a single queue stall.
        let mut uploads = GpuUploadBatch::new(device)?;
        let textures = Textures::upload(device, &mut uploads, &scene.images, &scene.textures)?;

        let storage = Self {
            accel,
            spectrum: uploads.upload(&spectrum.texels)?,
            sensor_spectrum: uploads.upload(&spectrum.sensor_texels)?,
            materials: uploads.upload(&materials)?,
            reflectance: uploads.upload(&reflectance)?,
            triangles: uploads.upload(&triangles)?,
            emissive_hits: uploads.upload(&emissive_hits)?,
            instances: uploads.upload(&instances)?,
            lights: uploads.upload(&lights)?,
            mesh_light_triangles: uploads.upload(&mesh_light_triangles)?,
            mesh_light_cdf: uploads.upload(&mesh_light_cdf)?,
            light_spectrum: uploads.upload(&light_spectrum_modulation)?,
            material_emission_spectrum: uploads.upload(&material_emission_spectrum)?,
            texture_bindings: uploads.upload(textures.bindings())?,
            texture_basis: uploads.upload(&texture_basis)?,
            baked_spectrum: spectrum,
            image_averages,
            textures,
        };
        uploads.submit()?;
        Ok(storage)
    }

    pub(super) fn destroy(self, device: &Device) {
        device.queue().wait_idle();
        self.accel.destroy(device);
        self.triangles.destroy(device);
        self.emissive_hits.destroy(device);
        self.instances.destroy(device);
        self.materials.destroy(device);
        self.lights.destroy(device);
        self.mesh_light_triangles.destroy(device);
        self.mesh_light_cdf.destroy(device);
        self.light_spectrum.destroy(device);
        self.material_emission_spectrum.destroy(device);
        self.spectrum.destroy(device);
        self.sensor_spectrum.destroy(device);
        self.reflectance.destroy(device);
        self.texture_bindings.destroy(device);
        self.texture_basis.destroy(device);
        self.textures.destroy(device);
    }
}

#[cfg(test)]
mod tests {
    use glam::{DMat4, DVec3, Vec3};

    use super::super::spectrum;
    use super::lights::{
        SpectrumResolver, build_analytic_gpu_light, build_material_emission_spectrum,
    };
    use super::materials::texture_basis_colors;
    use super::*;
    use crate::scene::{
        Camera, Geometry, Illuminant, Instance, Material, MaterialId, Mesh, MeshId, Projection,
        SceneData, Vertex,
    };
    use crate::scene::{Light, LightKind, SpectrumSource};

    fn two_triangle_emitter(transform: DMat4) -> Scene {
        let vertices = vec![
            Vertex {
                position: Vec3::ZERO,
                ..Vertex::default()
            },
            Vertex {
                position: Vec3::X,
                ..Vertex::default()
            },
            Vertex {
                position: Vec3::new(1.0, 1.0, 0.0),
                ..Vertex::default()
            },
            Vertex {
                position: Vec3::Y,
                ..Vertex::default()
            },
        ];
        let mesh = Mesh::with_material(vertices, vec![0, 1, 2, 0, 2, 3], MaterialId::DEFAULT);
        let mut scene = Scene::new(SceneData {
            geometry: Geometry::new(
                vec![mesh],
                vec![Instance {
                    mesh: MeshId(0),
                    path: "/Emitter".into(),
                    transform,
                }],
            ),
            lights: Vec::new(),
            nodes: Vec::new(),
            materials: vec![Material {
                emission: Illuminant::luminance(Vec3::ONE, 1.0),
                ..Material::default()
            }],
            images: Vec::new(),
            textures: Vec::new(),
            camera: Camera {
                world: DMat4::IDENTITY,
                projection: Projection {
                    vertical_fov_rad: 1.0,
                    clipping_range: [0.1, 100.0],
                },
            },
            up: DVec3::Y,
        });
        scene.refresh_emissive_components();
        scene
    }

    #[test]
    fn hot_path_records_stay_compact() {
        assert_eq!(size_of::<GpuMaterial>(), 48);
        assert_eq!(size_of::<GpuTriangle>(), 48);
        assert_eq!(size_of::<GpuInstance>(), 64);
    }

    #[test]
    fn texture_basis_fits_rgb_cube_corners() {
        for color in texture_basis_colors() {
            let fit = spectrum::fit_reflectance(color);
            assert!(fit.fit_error < 0.035, "{color:?}: {}", fit.fit_error);
        }
    }

    #[test]
    fn emissive_hits_reference_their_mesh_light() {
        let scene = two_triangle_emitter(DMat4::from_scale(DVec3::splat(2.0)));
        let default = spectrum::d65();
        let baked = default.bake(spectrum::DEFAULT_RESOLUTION);
        let lighting = build_lighting_arrays(&scene, &default, &baked);

        assert_eq!(lighting.emissive_hits.len(), 2);
        assert_eq!(lighting.emissive_hits[0].light_index, 0);
        assert_eq!(lighting.emissive_hits[1].light_index, 0);
        assert!((lighting.lights.lights[0].edge1_area.w - 4.0).abs() < 1e-5);
    }

    #[test]
    fn material_hit_spectrum_matches_light_modulation() {
        let mut scene = two_triangle_emitter(DMat4::IDENTITY);
        scene.materials[0].emission.spectrum = SpectrumSource::Blackbody { kelvin: 3200.0 };
        let default = spectrum::d65();
        let baked = default.bake(spectrum::DEFAULT_RESOLUTION);
        let mut material_resolver = SpectrumResolver::new(&default, &baked);
        let material = build_material_emission_spectrum(&scene, &mut material_resolver, &baked);
        let mut light_resolver = SpectrumResolver::new(&default, &baked);
        let mut light = Vec::new();
        light_resolver.append_modulation(&mut light, &scene.materials[0].emission.spectrum);

        assert_eq!(material, light);
        assert!(material.iter().any(|&scale| (scale - 1.0).abs() > 0.01));
    }

    #[test]
    fn sphere_light_record_preserves_surface_extent() {
        let light = Light {
            kind: LightKind::Sphere { radius: 2.0 },
            ..Light::default()
        };
        let gpu = build_analytic_gpu_light(&light, 3.0, 0.0);

        assert!((gpu.normal.x - 2.0).abs() < 1e-5);
        assert!((gpu.edge1_area.w - 16.0 * std::f32::consts::PI).abs() < 1e-4);
        assert_eq!(gpu.normal.w, LightTag::Sphere.as_f32());
    }
}
