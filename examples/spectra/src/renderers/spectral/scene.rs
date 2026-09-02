//! Device representation of a scene used by spectral tracing.
//!
//! Nothing in this module is part of the shared scene model: the ray geometry, acceleration
//! structure, BSDF parameters, light list, and spectral tables exist only for this backend.

use glam::{DMat4, DVec3, Vec2, Vec3, Vec4};
use kiln_rhi::{Device, GpuAllocation, MAX_FRAMES_IN_FLIGHT, gpu_struct};
use std::collections::HashMap;

use super::spectrum;
use super::spectrum::Spd;
use crate::base::gpu::{
    GpuArray, GpuPatchBatch, GpuTextureBinding, GpuUploadBatch, TextureResources,
};
use crate::base::renderer::{self, Error};
use crate::base::scene::{
    EmitterExtent, IntensityUnit, Light, LightKind, MaterialId, Mesh, Scene, SceneStorage,
    SpectrumSource, Surface, Triangle,
};

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
    pub(super) struct GpuEmissiveHit {
        light_index: u32,
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
    pub(super) struct GpuMeshLightTriangle {
        p0: Vec4,
        edge1: Vec4,
        edge2: Vec4,
        normal_area: Vec4,
    }
}

gpu_struct! {
    pub(super) struct GpuTriangle {
        // Mesh-local; instances apply the normal transform in the shader.
        normal_area: Vec4,
        uv01: Vec4,
        uv2: Vec2,
        material_id: u32,
        emission: f32,
    }
}

gpu_struct! {
    pub(super) struct GpuInstance {
        triangle_base: u32,
        _pad: u32,
        _pad2: u32,
        _pad3: u32,
        // Columns of the inverse-transpose normal matrix.
        normal_x: Vec4,
        normal_y: Vec4,
        normal_z_determinant: Vec4,
    }
}

#[derive(Clone, Copy, Debug)]
struct InstanceLayout {
    triangle_start: usize,
}

#[derive(Clone, Copy, Debug)]
struct MeshLightRange {
    instance_index: usize,
    component_index: usize,
    light_index: usize,
    triangle_start: usize,
    triangle_count: usize,
    cdf_start: usize,
}

struct MeshLightPatch {
    range: MeshLightRange,
    triangles: Vec<GpuMeshLightTriangle>,
    cdf: Vec<f32>,
    light: GpuLight,
}

struct BuiltLights {
    lights: Vec<GpuLight>,
    mesh_triangles: Vec<GpuMeshLightTriangle>,
    mesh_cdf: Vec<f32>,
    spectrum: Vec<f32>,
    mesh_ranges: Vec<MeshLightRange>,
}

struct LightingArrays {
    emission: Vec<f32>,
    material_spectrum: Vec<f32>,
    emissive_hits: Vec<GpuEmissiveHit>,
    lights: BuiltLights,
}

pub(super) struct Storage {
    pub(super) accel: SceneAccel,
    pub(super) triangles: GpuArray<GpuTriangle>,
    pub(super) emissive_hits: GpuArray<GpuEmissiveHit>,
    pub(super) instances: GpuArray<GpuInstance>,
    pub(super) bsdfs: GpuArray<GpuBsdf>,
    pub(super) lights: GpuArray<GpuLight>,
    pub(super) mesh_light_triangles: GpuArray<GpuMeshLightTriangle>,
    pub(super) mesh_light_cdf: GpuArray<f32>,
    pub(super) light_spectrum: GpuArray<f32>,
    pub(super) material_emission_spectrum: GpuArray<f32>,
    pub(super) spectrum: GpuArray<Vec4>,
    pub(super) sensor_spectrum: GpuArray<Vec4>,
    baked_spectrum: spectrum::EmissionSpectrum,
    pub(super) reflectance: GpuArray<f32>,
    pub(super) material_textures: GpuArray<GpuMaterialTexture>,
    pub(super) texture_bindings: GpuArray<GpuTextureBinding>,
    pub(super) texture_basis: GpuArray<f32>,
    // Cached because material edits do not change image contents.
    image_averages: Vec<Vec3>,
    material_texture_ids: Vec<u32>,
    instance_layout: Vec<InstanceLayout>,
    mesh_light_ranges: Vec<MeshLightRange>,
    mesh_light_ranges_by_instance: Vec<Vec<usize>>,
    mesh_light_records: Vec<GpuLight>,
    mesh_light_spectrum: Vec<f32>,
    analytic_light_count: usize,
    pending_staging: [Vec<GpuAllocation>; MAX_FRAMES_IN_FLIGHT],
    textures: TextureResources,
}

impl Storage {
    pub(super) fn begin_frame_update(&mut self, device: &Device, frame_slot: usize) {
        let Some(staging) = self.pending_staging.get_mut(frame_slot) else {
            return;
        };
        for allocation in staging.drain(..) {
            device.free(allocation);
        }
        self.accel.begin_frame(frame_slot);
    }

    fn retire_all_staging(&mut self, device: &Device) {
        for staging in &mut self.pending_staging {
            for allocation in staging.drain(..) {
                device.free(allocation);
            }
        }
    }

    fn wait_and_retire_staging(&mut self, device: &Device) {
        device.queue().wait_idle();
        self.retire_all_staging(device);
    }

    fn retain_staging(
        &mut self,
        device: &Device,
        frame_slot: usize,
        staging: Vec<GpuAllocation>,
    ) -> renderer::Result<()> {
        let Some(slot_staging) = self.pending_staging.get_mut(frame_slot) else {
            for allocation in staging {
                device.free(allocation);
            }
            return Err(Error::Capacity("frame slot is out of bounds"));
        };
        slot_staging.extend(staging);
        Ok(())
    }

    pub(super) fn update_lights(
        &mut self,
        device: &Device,
        scene: &Scene,
        default_spectrum: &Spd,
        light_indices: &[usize],
        frame_slot: usize,
    ) -> renderer::Result<()> {
        if scene.lights.len() != self.analytic_light_count {
            return self.rebuild_light_buffers(device, scene, default_spectrum);
        }
        let spectrum = default_spectrum.bake(spectrum::DEFAULT_RESOLUTION);
        let mut resolver = SpectrumResolver::new(default_spectrum, &spectrum);
        let mut patches = GpuPatchBatch::new(device)?;
        for &light_index in light_indices {
            let Some(light) = scene.lights.get(light_index) else {
                return Err(Error::Capacity("scene light update index is out of bounds"));
            };
            let emission = analytic_light_emission(light, &mut resolver, &spectrum);
            let gpu_light = build_analytic_gpu_light(light, emission, light_index as f32);
            let mut modulation = Vec::with_capacity(spectrum.texels.len());
            resolver.append_modulation(&mut modulation, &light.illuminant.spectrum);
            patches.patch(&self.lights, light_index, std::slice::from_ref(&gpu_light))?;
            patches.patch(
                &self.light_spectrum,
                light_index * spectrum.texels.len(),
                &modulation,
            )?;
        }
        let staging = patches.finish()?;
        self.retain_staging(device, frame_slot, staging)?;
        Ok(())
    }

    fn rebuild_light_buffers(
        &mut self,
        device: &Device,
        scene: &Scene,
        default_spectrum: &Spd,
    ) -> renderer::Result<()> {
        let spectrum = default_spectrum.bake(spectrum::DEFAULT_RESOLUTION);
        let mut resolver = SpectrumResolver::new(default_spectrum, &spectrum);
        let built = build_lights(
            scene,
            &mut resolver,
            &spectrum,
            &[],
            Some((&self.mesh_light_records, &self.mesh_light_spectrum)),
        );
        let (new_lights, new_light_spectrum) =
            upload_analytic_light_data(device, &built.lights, &built.spectrum)?;
        self.retire_all_staging(device);

        let old_lights = std::mem::replace(&mut self.lights, new_lights);
        let old_light_spectrum = std::mem::replace(&mut self.light_spectrum, new_light_spectrum);
        old_lights.destroy(device);
        old_light_spectrum.destroy(device);
        let old_count = self.analytic_light_count;
        self.analytic_light_count = scene.lights.len();
        self.mesh_light_records = built.lights[self.analytic_light_count..].to_vec();
        for range in &mut self.mesh_light_ranges {
            let mesh_index = range
                .light_index
                .checked_sub(old_count)
                .expect("mesh-light range must follow analytic lights");
            range.light_index = self.analytic_light_count + mesh_index;
        }
        Ok(())
    }

    pub(super) fn update_instances(
        &mut self,
        device: &Device,
        scene: &Scene,
        default_spectrum: &Spd,
        instance_indices: &[usize],
        frame_slot: usize,
    ) -> renderer::Result<()> {
        // Total-power emitters derive luminance from world-space area.
        if self.instances_rescale_a_total_power_emitter(scene, instance_indices) {
            return self.update_geometry(device, scene, default_spectrum);
        }
        let mut material_emission = None;
        let mut instance_patches = Vec::with_capacity(instance_indices.len());
        for &instance_index in instance_indices {
            let Some(layout) = self.instance_layout.get(instance_index).copied() else {
                return Err(Error::Capacity(
                    "scene instance update index is out of bounds",
                ));
            };
            let Some(instance) = scene.geometry.instances.get(instance_index) else {
                return Err(Error::Capacity(
                    "scene instance update index is out of bounds",
                ));
            };
            let gpu_instance = build_gpu_instance(layout, instance.transform);
            let mut mesh_patches = Vec::new();
            let Some(range_indices) = self.mesh_light_ranges_by_instance.get(instance_index) else {
                return Err(Error::Capacity(
                    "mesh-light instance index is out of bounds",
                ));
            };
            for &range_index in range_indices {
                let Some(range) = self.mesh_light_ranges.get(range_index).copied() else {
                    return Err(Error::Capacity("mesh-light range index is out of bounds"));
                };
                let cache_index = range
                    .light_index
                    .checked_sub(scene.lights.len())
                    .ok_or(Error::Capacity("mesh-light range precedes analytic lights"))?;
                let spectrum_id = self
                    .mesh_light_records
                    .get(cache_index)
                    .map(|light| light.edge2.w)
                    .ok_or(Error::Capacity("mesh-light cache range is out of bounds"))?;
                let by_material = material_emission.get_or_insert_with(|| {
                    let baked = default_spectrum.bake(spectrum::DEFAULT_RESOLUTION);
                    let mut resolver = SpectrumResolver::new(default_spectrum, &baked);
                    let areas = vec![0.0; scene.materials.len()];
                    build_material_emission(scene, &mut resolver, &baked, &areas)
                });
                let emission = mesh_light_material(scene, range)
                    .and_then(|material| by_material.get(material.0).copied())
                    .unwrap_or(0.0);
                let Some(patch) = build_mesh_light_patch(scene, range, emission, spectrum_id)
                else {
                    return self.update_geometry(device, scene, default_spectrum);
                };
                mesh_patches.push((cache_index, patch));
            }
            instance_patches.push((instance_index, gpu_instance, mesh_patches));
        }

        let mut patches = GpuPatchBatch::new(device)?;
        for (instance_index, gpu_instance, mesh_patches) in &instance_patches {
            patches.patch(
                &self.instances,
                *instance_index,
                std::slice::from_ref(gpu_instance),
            )?;
            for (_, patch) in mesh_patches {
                patches.patch(
                    &self.mesh_light_triangles,
                    patch.range.triangle_start,
                    &patch.triangles,
                )?;
                patches.patch(&self.mesh_light_cdf, patch.range.cdf_start, &patch.cdf)?;
                patches.patch(
                    &self.lights,
                    patch.range.light_index,
                    std::slice::from_ref(&patch.light),
                )?;
            }
        }
        let staging = patches.finish()?;
        self.retain_staging(device, frame_slot, staging)?;
        self.accel
            .update_transforms(device, scene, instance_indices, frame_slot)?;

        for (_, _, mesh_patches) in instance_patches {
            for (cache_index, patch) in mesh_patches {
                self.mesh_light_records[cache_index] = patch.light;
            }
        }
        Ok(())
    }

    fn instances_rescale_a_total_power_emitter(
        &self,
        scene: &Scene,
        instance_indices: &[usize],
    ) -> bool {
        instance_indices.iter().any(|&instance_index| {
            let Some(instance) = scene.geometry.instances.get(instance_index) else {
                return false;
            };
            let Some(mesh) = scene.geometry.meshes.get(instance.mesh.0) else {
                return false;
            };
            mesh.emissive_components.iter().any(|component| {
                scene
                    .materials
                    .get(component.material.0)
                    .is_some_and(|material| material.emission.unit != IntensityUnit::Luminance)
            })
        })
    }

    pub(super) fn update_material_surfaces(
        &mut self,
        device: &Device,
        scene: &Scene,
        material_indices: &[usize],
        frame_slot: usize,
    ) -> renderer::Result<()> {
        if scene.materials.len() != self.material_texture_ids.len() {
            return self.rebuild_material_surfaces(device, scene);
        }
        let spectrum_len = self.baked_spectrum.texels.len();
        let mut lowered = Vec::with_capacity(material_indices.len());
        for &index in material_indices {
            let Some(material) = scene.materials.get(index) else {
                return Err(Error::Capacity(
                    "scene material update index is out of bounds",
                ));
            };
            let representative =
                representative_base_color(material.surface, scene, &self.image_averages);
            let params = bsdf::lower(material, representative)?;
            let old_texture_id =
                self.material_texture_ids
                    .get(index)
                    .copied()
                    .ok_or(Error::Capacity(
                        "scene material update index is out of bounds",
                    ))?;
            if params.base_color_texture.is_some() != (old_texture_id != u32::MAX) {
                return self.rebuild_material_surfaces(device, scene);
            }
            lowered.push((index, params, old_texture_id));
        }

        let mut patches = GpuPatchBatch::new(device)?;
        for (index, params, texture_id) in lowered {
            patches.patch(
                &self.bsdfs,
                index,
                std::slice::from_ref(&bsdf_to_gpu(params, texture_id)),
            )?;
            let fit = spectrum::fit_reflectance(params.base_color);
            let reflectance =
                build_reflectance_lut(std::slice::from_ref(&fit), &self.baked_spectrum);
            patches.patch(&self.reflectance, index * 2 * spectrum_len, &reflectance)?;
            if let Some(binding) = params.base_color_texture {
                let texture = GpuMaterialTexture {
                    factor: params.texture_factor.extend(1.0),
                    binding_id: binding.0 as u32,
                    _pad: [0.0; 3],
                };
                patches.patch(
                    &self.material_textures,
                    texture_id as usize,
                    std::slice::from_ref(&texture),
                )?;
            }
        }
        let staging = patches.finish()?;
        self.retain_staging(device, frame_slot, staging)
    }

    fn rebuild_material_surfaces(
        &mut self,
        device: &Device,
        scene: &Scene,
    ) -> renderer::Result<()> {
        self.wait_and_retire_staging(device);
        let spectrum = &self.baked_spectrum;
        let MaterialArrays {
            bsdfs,
            material_textures,
            reflectance,
            texture_basis,
        } = build_material_arrays(scene, &self.image_averages, spectrum)?;
        let material_texture_ids = bsdfs.iter().map(|bsdf| bsdf.texture_id).collect();

        let mut uploads = GpuUploadBatch::new(device);
        uploads.upload(&bsdfs)?;
        uploads.upload(&reflectance)?;
        uploads.upload(&material_textures)?;
        uploads.upload(&texture_basis)?;
        let [
            bsdf_gpu,
            reflectance_gpu,
            material_texture_gpu,
            texture_basis_gpu,
        ] = uploads.finish()?;

        let old_bsdfs = std::mem::replace(&mut self.bsdfs, GpuArray::new(bsdf_gpu, bsdfs.len()));
        let old_reflectance = std::mem::replace(
            &mut self.reflectance,
            GpuArray::new(reflectance_gpu, reflectance.len()),
        );
        let old_material_textures = std::mem::replace(
            &mut self.material_textures,
            GpuArray::new(material_texture_gpu, material_textures.len()),
        );
        let old_texture_basis = std::mem::replace(
            &mut self.texture_basis,
            GpuArray::new(texture_basis_gpu, texture_basis.len()),
        );
        old_bsdfs.destroy(device);
        old_reflectance.destroy(device);
        old_material_textures.destroy(device);
        old_texture_basis.destroy(device);
        self.material_texture_ids = material_texture_ids;
        Ok(())
    }

    pub(super) fn update_material_emission(
        &mut self,
        device: &Device,
        scene: &Scene,
        default_spectrum: &Spd,
    ) -> renderer::Result<()> {
        self.wait_and_retire_staging(device);
        let spectrum = default_spectrum.bake(spectrum::DEFAULT_RESOLUTION);
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
                    mesh_ranges: mesh_light_ranges,
                },
        } = build_lighting_arrays(scene, default_spectrum, &spectrum);
        let triangles = build_gpu_triangles(scene, &emission);
        let mesh_light_start = scene.lights.len();
        let mesh_light_spectrum_start = mesh_light_start * spectrum.texels.len();
        let mesh_light_records = lights[mesh_light_start..].to_vec();
        let mesh_light_spectrum = light_spectrum_modulation[mesh_light_spectrum_start..].to_vec();

        let mut uploads = GpuUploadBatch::new(device);
        uploads.upload(&triangles)?;
        uploads.upload(&emissive_hits)?;
        uploads.upload(&lights)?;
        uploads.upload(&mesh_light_triangles)?;
        uploads.upload(&mesh_light_cdf)?;
        uploads.upload(&light_spectrum_modulation)?;
        uploads.upload(&material_emission_spectrum)?;
        let [
            triangle_gpu,
            emissive_hits_gpu,
            light_gpu,
            mesh_light_triangle_gpu,
            mesh_light_cdf_gpu,
            light_spectrum_gpu,
            material_emission_spectrum_gpu,
        ] = uploads.finish()?;

        let old_triangles = std::mem::replace(
            &mut self.triangles,
            GpuArray::new(triangle_gpu, triangles.len()),
        );
        let old_emissive_hits = std::mem::replace(
            &mut self.emissive_hits,
            GpuArray::new(emissive_hits_gpu, emissive_hits.len()),
        );
        let old_lights =
            std::mem::replace(&mut self.lights, GpuArray::new(light_gpu, lights.len()));
        let old_mesh_triangles = std::mem::replace(
            &mut self.mesh_light_triangles,
            GpuArray::new(mesh_light_triangle_gpu, mesh_light_triangles.len()),
        );
        let old_mesh_cdf = std::mem::replace(
            &mut self.mesh_light_cdf,
            GpuArray::new(mesh_light_cdf_gpu, mesh_light_cdf.len()),
        );
        let old_light_spectrum = std::mem::replace(
            &mut self.light_spectrum,
            GpuArray::new(light_spectrum_gpu, light_spectrum_modulation.len()),
        );
        let old_material_emission_spectrum = std::mem::replace(
            &mut self.material_emission_spectrum,
            GpuArray::new(
                material_emission_spectrum_gpu,
                material_emission_spectrum.len(),
            ),
        );
        old_triangles.destroy(device);
        old_emissive_hits.destroy(device);
        old_lights.destroy(device);
        old_mesh_triangles.destroy(device);
        old_mesh_cdf.destroy(device);
        old_light_spectrum.destroy(device);
        old_material_emission_spectrum.destroy(device);

        self.mesh_light_ranges = mesh_light_ranges;
        self.mesh_light_ranges_by_instance =
            build_mesh_light_range_index(scene.geometry.instances.len(), &self.mesh_light_ranges);
        self.mesh_light_records = mesh_light_records;
        self.mesh_light_spectrum = mesh_light_spectrum;
        self.analytic_light_count = scene.lights.len();
        Ok(())
    }

    pub(super) fn update_geometry(
        &mut self,
        device: &Device,
        scene: &Scene,
        default_spectrum: &Spd,
    ) -> renderer::Result<()> {
        self.wait_and_retire_staging(device);
        if scene.triangle_count() > (u32::MAX as usize) / 3 {
            return Err(Error::Capacity("trace geometry exceeds u32 indexing"));
        }
        let spectrum = default_spectrum.bake(spectrum::DEFAULT_RESOLUTION);
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
                    mesh_ranges: mesh_light_ranges,
                },
        } = build_lighting_arrays(scene, default_spectrum, &spectrum);
        let triangles = build_gpu_triangles(scene, &emission);
        let instance_layout = build_instance_layout(scene);
        let instances = build_gpu_instances(scene, &instance_layout);
        let mesh_light_start = scene.lights.len();
        let mesh_light_spectrum_start = mesh_light_start * spectrum.texels.len();
        let mesh_light_records = lights[mesh_light_start..].to_vec();
        let mesh_light_spectrum = light_spectrum_modulation[mesh_light_spectrum_start..].to_vec();
        let new_accel = SceneAccel::build(device, scene)?;
        let uploaded = (|| {
            let mut uploads = GpuUploadBatch::new(device);
            uploads.upload(&triangles)?;
            uploads.upload(&emissive_hits)?;
            uploads.upload(&instances)?;
            uploads.upload(&lights)?;
            uploads.upload(&mesh_light_triangles)?;
            uploads.upload(&mesh_light_cdf)?;
            uploads.upload(&light_spectrum_modulation)?;
            uploads.upload(&material_emission_spectrum)?;
            uploads.finish()
        })();
        let [
            triangle_gpu,
            emissive_hits_gpu,
            instance_gpu,
            light_gpu,
            mesh_triangle_gpu,
            mesh_cdf_gpu,
            light_spectrum_gpu,
            material_emission_spectrum_gpu,
        ] = match uploaded {
            Ok(value) => value,
            Err(error) => {
                new_accel.destroy(device);
                return Err(error);
            }
        };

        let new_triangles = GpuArray::new(triangle_gpu, triangles.len());
        let new_emissive_hits = GpuArray::new(emissive_hits_gpu, emissive_hits.len());
        let new_instances = GpuArray::new(instance_gpu, instances.len());
        let new_lights = GpuArray::new(light_gpu, lights.len());
        let new_mesh_triangles = GpuArray::new(mesh_triangle_gpu, mesh_light_triangles.len());
        let new_mesh_cdf = GpuArray::new(mesh_cdf_gpu, mesh_light_cdf.len());
        let new_light_spectrum = GpuArray::new(light_spectrum_gpu, light_spectrum_modulation.len());
        let new_material_emission_spectrum = GpuArray::new(
            material_emission_spectrum_gpu,
            material_emission_spectrum.len(),
        );
        let old_accel = std::mem::replace(&mut self.accel, new_accel);
        old_accel.destroy(device);
        let old_triangles = std::mem::replace(&mut self.triangles, new_triangles);
        let old_emissive_hits = std::mem::replace(&mut self.emissive_hits, new_emissive_hits);
        let old_instances = std::mem::replace(&mut self.instances, new_instances);
        let old_lights = std::mem::replace(&mut self.lights, new_lights);
        let old_mesh_triangles =
            std::mem::replace(&mut self.mesh_light_triangles, new_mesh_triangles);
        let old_mesh_cdf = std::mem::replace(&mut self.mesh_light_cdf, new_mesh_cdf);
        let old_light_spectrum = std::mem::replace(&mut self.light_spectrum, new_light_spectrum);
        let old_material_emission_spectrum = std::mem::replace(
            &mut self.material_emission_spectrum,
            new_material_emission_spectrum,
        );
        old_triangles.destroy(device);
        old_emissive_hits.destroy(device);
        old_instances.destroy(device);
        old_lights.destroy(device);
        old_mesh_triangles.destroy(device);
        old_mesh_cdf.destroy(device);
        old_light_spectrum.destroy(device);
        old_material_emission_spectrum.destroy(device);
        self.instance_layout = instance_layout;
        self.mesh_light_ranges = mesh_light_ranges;
        self.mesh_light_ranges_by_instance =
            build_mesh_light_range_index(scene.geometry.instances.len(), &self.mesh_light_ranges);
        self.mesh_light_records = mesh_light_records;
        self.mesh_light_spectrum = mesh_light_spectrum;
        self.analytic_light_count = scene.lights.len();
        Ok(())
    }
}

impl SceneStorage for Storage {
    type Config = Spd;
    type Error = renderer::Error;

    fn build(
        device: &Device,
        scene: &Scene,
        light_spectrum: &Self::Config,
    ) -> renderer::Result<Self> {
        if scene.triangle_count() > (u32::MAX as usize) / 3 {
            return Err(Error::Capacity("trace geometry exceeds u32 indexing"));
        }
        let spectrum = light_spectrum.bake(spectrum::DEFAULT_RESOLUTION);
        let image_averages = scene
            .images
            .iter()
            .map(|image| image.average_color())
            .collect::<Vec<_>>();
        let materials = build_material_arrays(scene, &image_averages, &spectrum)?;
        let MaterialArrays {
            bsdfs,
            material_textures,
            reflectance,
            texture_basis,
        } = materials;
        let material_texture_ids = bsdfs.iter().map(|bsdf| bsdf.texture_id).collect();
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
                    mesh_ranges: mesh_light_ranges,
                },
        } = build_lighting_arrays(scene, light_spectrum, &spectrum);
        let triangles = build_gpu_triangles(scene, &emission);
        let instance_layout = build_instance_layout(scene);
        let instances = build_gpu_instances(scene, &instance_layout);
        let mesh_light_start = scene.lights.len();
        let mesh_light_spectrum_start = mesh_light_start * spectrum.texels.len();
        let mesh_light_records = lights[mesh_light_start..].to_vec();
        let mesh_light_spectrum = light_spectrum_modulation[mesh_light_spectrum_start..].to_vec();
        let mesh_light_ranges_by_instance =
            build_mesh_light_range_index(scene.geometry.instances.len(), &mesh_light_ranges);
        let accel = SceneAccel::build(device, scene)?;
        let textures = match TextureResources::upload(device, &scene.images, &scene.textures) {
            Ok(textures) => textures,
            Err(error) => {
                accel.destroy(device);
                return Err(error);
            }
        };
        let texture_bindings = textures.bindings();

        let uploaded = (|| {
            let mut uploads = GpuUploadBatch::new(device);
            uploads.upload(&spectrum.texels)?;
            uploads.upload(&spectrum.sensor_texels)?;
            uploads.upload(&bsdfs)?;
            uploads.upload(&reflectance)?;
            uploads.upload(&triangles)?;
            uploads.upload(&emissive_hits)?;
            uploads.upload(&instances)?;
            uploads.upload(&lights)?;
            uploads.upload(&mesh_light_triangles)?;
            uploads.upload(&mesh_light_cdf)?;
            uploads.upload(&light_spectrum_modulation)?;
            uploads.upload(&material_emission_spectrum)?;
            uploads.upload(&material_textures)?;
            uploads.upload(texture_bindings)?;
            uploads.upload(&texture_basis)?;
            uploads.finish()
        })();
        let [
            spectrum_gpu,
            sensor_spectrum_gpu,
            bsdf_gpu,
            reflectance_gpu,
            triangle_gpu,
            emissive_hits_gpu,
            instance_gpu,
            light_gpu,
            mesh_light_triangle_gpu,
            mesh_light_cdf_gpu,
            light_spectrum_gpu,
            material_emission_spectrum_gpu,
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
            "spectral GPU scene: {} triangles, {} materials, {} lights, light spectrum {} ({} texels)",
            triangles.len(),
            bsdfs.len(),
            lights.len(),
            spectrum.name,
            spectrum.texels.len()
        );
        Ok(Self {
            accel,
            triangles: GpuArray::new(triangle_gpu, triangles.len()),
            emissive_hits: GpuArray::new(emissive_hits_gpu, emissive_hits.len()),
            instances: GpuArray::new(instance_gpu, instances.len()),
            bsdfs: GpuArray::new(bsdf_gpu, bsdfs.len()),
            lights: GpuArray::new(light_gpu, lights.len()),
            mesh_light_triangles: GpuArray::new(
                mesh_light_triangle_gpu,
                mesh_light_triangles.len(),
            ),
            mesh_light_cdf: GpuArray::new(mesh_light_cdf_gpu, mesh_light_cdf.len()),
            light_spectrum: GpuArray::new(light_spectrum_gpu, light_spectrum_modulation.len()),
            material_emission_spectrum: GpuArray::new(
                material_emission_spectrum_gpu,
                material_emission_spectrum.len(),
            ),
            spectrum: GpuArray::new(spectrum_gpu, spectrum.texels.len()),
            sensor_spectrum: GpuArray::new(sensor_spectrum_gpu, spectrum.sensor_texels.len()),
            baked_spectrum: spectrum,
            reflectance: GpuArray::new(reflectance_gpu, reflectance.len()),
            material_textures: GpuArray::new(material_texture_gpu, material_textures.len()),
            texture_bindings: GpuArray::new(texture_binding_gpu, texture_bindings.len()),
            texture_basis: GpuArray::new(texture_basis_gpu, texture_basis.len()),
            image_averages,
            material_texture_ids,
            instance_layout,
            mesh_light_ranges,
            mesh_light_ranges_by_instance,
            mesh_light_records,
            mesh_light_spectrum,
            analytic_light_count: scene.lights.len(),
            pending_staging: std::array::from_fn(|_| Vec::new()),
            textures,
        })
    }

    fn destroy(mut self, device: &Device) {
        self.wait_and_retire_staging(device);
        self.accel.destroy(device);
        self.triangles.destroy(device);
        self.emissive_hits.destroy(device);
        self.instances.destroy(device);
        self.bsdfs.destroy(device);
        self.lights.destroy(device);
        self.mesh_light_triangles.destroy(device);
        self.mesh_light_cdf.destroy(device);
        self.light_spectrum.destroy(device);
        self.material_emission_spectrum.destroy(device);
        self.spectrum.destroy(device);
        self.sensor_spectrum.destroy(device);
        self.reflectance.destroy(device);
        self.material_textures.destroy(device);
        self.texture_bindings.destroy(device);
        self.texture_basis.destroy(device);
        self.textures.destroy(device);
    }
}

fn build_gpu_triangles(scene: &Scene, emission: &[f32]) -> Vec<GpuTriangle> {
    let mut triangles = Vec::with_capacity(scene.triangle_count());
    for instance in &scene.geometry.instances {
        let mesh = &scene.geometry.meshes[instance.mesh.0];
        for primitive in &mesh.primitives {
            let indices =
                &mesh.indices[primitive.index_start..primitive.index_start + primitive.index_count];
            for corners in indices.chunks_exact(3) {
                triangles.push(gpu_triangle(
                    Triangle {
                        vertices: [
                            mesh.vertices[corners[0] as usize],
                            mesh.vertices[corners[1] as usize],
                            mesh.vertices[corners[2] as usize],
                        ],
                        material: primitive.material,
                    },
                    emission,
                ));
            }
        }
    }
    triangles
}

fn build_material_emitter_areas(scene: &Scene) -> Vec<f32> {
    let mut material_areas = vec![0.0; scene.materials.len()];
    for instance in &scene.geometry.instances {
        let mesh = &scene.geometry.meshes[instance.mesh.0];
        for component in &mesh.emissive_components {
            let area = component
                .triangles
                .iter()
                .map(|corners| world_triangle_area(instance.transform, mesh, *corners))
                .sum::<f32>();
            material_areas[component.material.0] += area;
        }
    }
    material_areas
}

fn build_emissive_hits(scene: &Scene, ranges: &[MeshLightRange]) -> Vec<GpuEmissiveHit> {
    let invalid = GpuEmissiveHit {
        light_index: u32::MAX,
    };
    let mut hits = vec![invalid; scene.triangle_count()];
    let mut triangle_base = 0;
    for (instance_index, instance) in scene.geometry.instances.iter().enumerate() {
        let mesh = &scene.geometry.meshes[instance.mesh.0];
        let mut flattened = HashMap::new();
        let mut local_index = 0;
        for primitive in &mesh.primitives {
            let range = primitive.index_start..primitive.index_start + primitive.index_count;
            for corners in mesh.indices[range].chunks_exact(3) {
                flattened.insert(
                    [corners[0], corners[1], corners[2]],
                    triangle_base + local_index,
                );
                local_index += 1;
            }
        }

        for range in ranges
            .iter()
            .filter(|range| range.instance_index == instance_index)
        {
            let component = &mesh.emissive_components[range.component_index];
            for &corners in &component.triangles {
                if world_triangle_area(instance.transform, mesh, corners) <= f32::EPSILON {
                    continue;
                }
                hits[flattened[&corners]] = GpuEmissiveHit {
                    light_index: range.light_index as u32,
                };
            }
        }
        triangle_base += local_index;
    }
    hits
}

fn gpu_triangle(triangle: Triangle, emission: &[f32]) -> GpuTriangle {
    let material = triangle.material;
    let (normal, area) = triangle.geometric_normal_and_area();
    let uv0 = triangle.vertices[0].uv.unwrap_or_default();
    let uv1 = triangle.vertices[1].uv.unwrap_or_default();
    let uv2 = triangle.vertices[2].uv.unwrap_or_default();
    GpuTriangle {
        normal_area: normal.extend(area),
        uv01: Vec4::new(uv0.x, uv0.y, uv1.x, uv1.y),
        uv2,
        material_id: material.0 as u32,
        emission: emission.get(material.0).copied().unwrap_or(0.0),
    }
}

fn world_triangle_area(transform: DMat4, mesh: &Mesh, corners: [u32; 3]) -> f32 {
    let vertices = corners
        .map(|index| transform.transform_point3(mesh.vertices[index as usize].position.as_dvec3()));
    let cross = (vertices[1] - vertices[0]).cross(vertices[2] - vertices[0]);
    (0.5 * cross.length()) as f32
}

fn build_material_emission(
    scene: &Scene,
    resolver: &mut SpectrumResolver<'_>,
    spectrum: &spectrum::EmissionSpectrum,
    areas: &[f32],
) -> Vec<f32> {
    scene
        .materials
        .iter()
        .enumerate()
        .map(|(index, material)| {
            if !material.is_emissive() {
                return 0.0;
            }
            let extent = EmitterExtent::Surface(areas[index]);
            let efficacy = resolver.luminous_efficacy(&material.emission.spectrum);
            spectrum.emission_from_luminance(material.emission.emitted_luminance(extent, efficacy))
        })
        .collect()
}

// Dense by material to keep the triangle record at 48 bytes.
fn build_material_emission_spectrum(
    scene: &Scene,
    resolver: &mut SpectrumResolver<'_>,
    spectrum: &spectrum::EmissionSpectrum,
) -> Vec<f32> {
    let mut modulation = Vec::with_capacity(scene.materials.len() * spectrum.texels.len());
    for material in &scene.materials {
        resolver.append_modulation(&mut modulation, &material.emission.spectrum);
    }
    modulation
}

fn build_lighting_arrays(
    scene: &Scene,
    default_spectrum: &Spd,
    spectrum: &spectrum::EmissionSpectrum,
) -> LightingArrays {
    let mut resolver = SpectrumResolver::new(default_spectrum, spectrum);
    let material_areas = build_material_emitter_areas(scene);
    let emission = build_material_emission(scene, &mut resolver, spectrum, &material_areas);
    let material_spectrum = build_material_emission_spectrum(scene, &mut resolver, spectrum);
    let lights = build_lights(scene, &mut resolver, spectrum, &emission, None);
    let emissive_hits = build_emissive_hits(scene, &lights.mesh_ranges);
    LightingArrays {
        emission,
        material_spectrum,
        emissive_hits,
        lights,
    }
}

struct MaterialArrays {
    bsdfs: Vec<GpuBsdf>,
    material_textures: Vec<GpuMaterialTexture>,
    reflectance: Vec<f32>,
    texture_basis: Vec<f32>,
}

fn build_material_arrays(
    scene: &Scene,
    image_averages: &[Vec3],
    spectrum: &spectrum::EmissionSpectrum,
) -> renderer::Result<MaterialArrays> {
    let mut fits = Vec::with_capacity(scene.materials.len());
    let mut lowered = Vec::with_capacity(scene.materials.len());
    for material in &scene.materials {
        let representative = representative_base_color(material.surface, scene, image_averages);
        let params = bsdf::lower(material, representative)?;
        let fit = spectrum::fit_reflectance(params.base_color);
        if fit.fit_error > 0.01 {
            eprintln!(
                "spectral fit for albedo {:?} off by {:.3} (moments {:?})",
                representative_surface_color(material.surface),
                fit.fit_error,
                fit.trig_moments
            );
        }
        fits.push(fit);
        lowered.push(params);
    }

    let mut material_textures = Vec::new();
    let mut bsdfs = Vec::with_capacity(lowered.len());
    for params in lowered {
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
        build_texture_basis(spectrum)
    };
    Ok(MaterialArrays {
        bsdfs,
        material_textures,
        reflectance: build_reflectance_lut(&fits, spectrum),
        texture_basis,
    })
}

fn mesh_light_material(scene: &Scene, range: MeshLightRange) -> Option<MaterialId> {
    let instance = scene.geometry.instances.get(range.instance_index)?;
    let mesh = scene.geometry.meshes.get(instance.mesh.0)?;
    Some(
        mesh.emissive_components
            .get(range.component_index)?
            .material,
    )
}

fn build_mesh_light_patch(
    scene: &Scene,
    range: MeshLightRange,
    emission: f32,
    spectrum_id: f32,
) -> Option<MeshLightPatch> {
    let instance = scene.geometry.instances.get(range.instance_index)?;
    let mesh = scene.geometry.meshes.get(instance.mesh.0)?;
    let component = mesh.emissive_components.get(range.component_index)?;
    let mut triangles = Vec::with_capacity(component.triangles.len());
    let mut cdf = Vec::with_capacity(component.triangles.len());
    let mut total_area = 0.0_f32;
    for corners in &component.triangles {
        let vertices = corners.map(|index| {
            instance
                .transform
                .transform_point3(mesh.vertices[index as usize].position.as_dvec3())
        });
        let p0 = vertices[0];
        let edge1 = vertices[1] - vertices[0];
        let edge2 = vertices[2] - vertices[0];
        let cross = edge1.cross(edge2);
        let area = (0.5 * cross.length()) as f32;
        if area <= f32::EPSILON {
            continue;
        }
        total_area += area;
        triangles.push(GpuMeshLightTriangle {
            p0: p0.as_vec3().extend(0.0),
            edge1: edge1.as_vec3().extend(0.0),
            edge2: edge2.as_vec3().extend(0.0),
            normal_area: cross.normalize().as_vec3().extend(area),
        });
        cdf.push(total_area);
    }
    if triangles.len() != range.triangle_count || total_area <= f32::EPSILON {
        return None;
    }
    for value in &mut cdf {
        *value /= total_area;
    }
    Some(MeshLightPatch {
        range,
        triangles,
        cdf,
        light: GpuLight {
            p0_emission: Vec4::new(0.0, 0.0, 0.0, emission),
            edge1_area: Vec4::new(0.0, 0.0, 0.0, total_area),
            edge2: Vec4::new(
                range.cdf_start as f32,
                range.triangle_count as f32,
                0.0,
                spectrum_id,
            ),
            normal: Vec4::new(0.0, 0.0, 0.0, LIGHT_MESH),
        },
    })
}

fn build_instance_layout(scene: &Scene) -> Vec<InstanceLayout> {
    let mut triangle_base = 0;
    scene
        .geometry
        .instances
        .iter()
        .map(|instance| {
            let mesh = &scene.geometry.meshes[instance.mesh.0];
            let triangle_count = mesh
                .primitives
                .iter()
                .map(|primitive| primitive.index_count / 3)
                .sum::<usize>();
            let layout = InstanceLayout {
                triangle_start: triangle_base,
            };
            triangle_base += triangle_count;
            layout
        })
        .collect()
}

fn build_gpu_instances(scene: &Scene, layout: &[InstanceLayout]) -> Vec<GpuInstance> {
    layout
        .iter()
        .zip(&scene.geometry.instances)
        .map(|(layout, instance)| build_gpu_instance(*layout, instance.transform))
        .collect()
}

fn build_gpu_instance(layout: InstanceLayout, transform: glam::DMat4) -> GpuInstance {
    let normal = transform.inverse().transpose();
    GpuInstance {
        triangle_base: layout.triangle_start as u32,
        _pad: 0,
        _pad2: 0,
        _pad3: 0,
        normal_x: normal.x_axis.truncate().as_vec3().extend(0.0),
        normal_y: normal.y_axis.truncate().as_vec3().extend(0.0),
        normal_z_determinant: normal
            .z_axis
            .truncate()
            .as_vec3()
            .extend(transform.determinant() as f32),
    }
}

fn build_mesh_light_range_index(
    instance_count: usize,
    ranges: &[MeshLightRange],
) -> Vec<Vec<usize>> {
    let mut by_instance = vec![Vec::new(); instance_count];
    for (range_index, range) in ranges.iter().enumerate() {
        if let Some(indices) = by_instance.get_mut(range.instance_index) {
            indices.push(range_index);
        }
    }
    by_instance
}

fn upload_analytic_light_data(
    device: &Device,
    lights: &[GpuLight],
    light_spectrum_modulation: &[f32],
) -> renderer::Result<(GpuArray<GpuLight>, GpuArray<f32>)> {
    let mut uploads = GpuUploadBatch::new(device);
    uploads.upload(lights)?;
    uploads.upload(light_spectrum_modulation)?;
    let [light_gpu, light_spectrum_gpu] = uploads.finish()?;
    Ok((
        GpuArray::new(light_gpu, lights.len()),
        GpuArray::new(light_spectrum_gpu, light_spectrum_modulation.len()),
    ))
}

fn build_analytic_gpu_light(light: &Light, emission: f32, spectrum_id: f32) -> GpuLight {
    match &light.kind {
        LightKind::Rect { width, height } => {
            let edge1 = light
                .transform
                .transform_vector3(DVec3::X * f64::from(width.max(0.0)));
            let edge2 = light
                .transform
                .transform_vector3(DVec3::Y * f64::from(height.max(0.0)));
            let cross = edge1.cross(edge2);
            let area = cross.length() as f32;
            if area <= f32::EPSILON {
                return empty_gpu_light(spectrum_id);
            }
            GpuLight {
                p0_emission: light
                    .transform
                    .transform_point3(DVec3::ZERO)
                    .as_vec3()
                    .extend(emission),
                edge1_area: edge1.as_vec3().extend(area),
                edge2: edge2.as_vec3().extend(spectrum_id),
                normal: cross.normalize().as_vec3().extend(LIGHT_RECT),
            }
        }
        LightKind::Disk { radius } => {
            let edge1 = light
                .transform
                .transform_vector3(DVec3::X * f64::from(radius.max(0.0)));
            let edge2 = light
                .transform
                .transform_vector3(DVec3::Y * f64::from(radius.max(0.0)));
            let cross = edge1.cross(edge2);
            let area = std::f64::consts::PI * cross.length();
            if area <= f64::EPSILON {
                return empty_gpu_light(spectrum_id);
            }
            GpuLight {
                p0_emission: light
                    .transform
                    .transform_point3(DVec3::ZERO)
                    .as_vec3()
                    .extend(emission),
                edge1_area: edge1.as_vec3().extend(area as f32),
                edge2: edge2.as_vec3().extend(spectrum_id),
                normal: cross.normalize().as_vec3().extend(LIGHT_DISK),
            }
        }
        LightKind::Point => analytic_point_light(light, emission, spectrum_id, LIGHT_POINT),
        LightKind::Sphere { radius } => {
            let center = light.transform.transform_point3(DVec3::ZERO);
            let radius = light
                .transform
                .transform_vector3(DVec3::X * f64::from(radius.max(0.0)))
                .length() as f32;
            let area = 4.0 * std::f32::consts::PI * radius * radius;
            if area <= f32::EPSILON {
                return empty_gpu_light(spectrum_id);
            }
            GpuLight {
                p0_emission: center.as_vec3().extend(emission),
                edge1_area: Vec4::new(0.0, 0.0, 0.0, area),
                edge2: Vec4::new(0.0, 0.0, 0.0, spectrum_id),
                normal: Vec4::new(radius, 0.0, 0.0, LIGHT_SPHERE),
            }
        }
        LightKind::Directional { .. } => {
            analytic_point_light(light, emission, spectrum_id, LIGHT_DIRECTIONAL)
        }
        LightKind::Dome => analytic_point_light(light, emission, spectrum_id, LIGHT_DOME),
    }
}

fn analytic_point_light(light: &Light, emission: f32, spectrum_id: f32, kind: f32) -> GpuLight {
    let p0 = light.transform.transform_point3(DVec3::ZERO);
    let direction = light
        .transform
        .transform_vector3(DVec3::NEG_Z)
        .normalize_or(DVec3::NEG_Z);
    GpuLight {
        p0_emission: p0.as_vec3().extend(emission),
        edge1_area: direction.as_vec3().extend(0.0),
        edge2: Vec4::new(0.0, 0.0, 0.0, spectrum_id),
        normal: Vec4::new(0.0, 0.0, 0.0, kind),
    }
}

fn build_lights(
    scene: &Scene,
    resolver: &mut SpectrumResolver<'_>,
    spectrum: &spectrum::EmissionSpectrum,
    material_emission: &[f32],
    mesh_cache: Option<(&[GpuLight], &[f32])>,
) -> BuiltLights {
    let mut lights = Vec::new();
    let mut mesh_light_triangles = Vec::new();
    let mut mesh_light_cdf = Vec::new();
    let mut light_spectrum_modulation = Vec::new();
    let mut mesh_light_ranges = Vec::new();
    for light in &scene.lights {
        let emission = analytic_light_emission(light, resolver, spectrum);
        let spectrum_id =
            resolver.append_modulation(&mut light_spectrum_modulation, &light.illuminant.spectrum);
        lights.push(build_analytic_gpu_light(light, emission, spectrum_id));
    }

    if let Some((mesh_records, mesh_spectrum)) = mesh_cache {
        let mesh_start = lights.len();
        lights.extend(mesh_records.iter().enumerate().map(|(index, record)| {
            let mut record = *record;
            record.edge2.w = (mesh_start + index) as f32;
            record
        }));
        light_spectrum_modulation.extend_from_slice(mesh_spectrum);
        return BuiltLights {
            lights,
            mesh_triangles: mesh_light_triangles,
            mesh_cdf: mesh_light_cdf,
            spectrum: light_spectrum_modulation,
            mesh_ranges: mesh_light_ranges,
        };
    }

    for (instance_index, instance) in scene.geometry.instances.iter().enumerate() {
        let mesh = &scene.geometry.meshes[instance.mesh.0];
        for (component_index, component) in mesh.emissive_components.iter().enumerate() {
            let Some(material) = scene.materials.get(component.material.0) else {
                continue;
            };
            let cdf_start = mesh_light_cdf.len();
            let triangle_start = mesh_light_triangles.len();
            let mut total_area = 0.0_f32;
            for corners in &component.triangles {
                let vertices = corners.map(|index| {
                    instance
                        .transform
                        .transform_point3(mesh.vertices[index as usize].position.as_dvec3())
                });
                let p0 = vertices[0];
                let edge1 = vertices[1] - vertices[0];
                let edge2 = vertices[2] - vertices[0];
                let cross = edge1.cross(edge2);
                let area = (0.5 * cross.length()) as f32;
                if area <= f32::EPSILON {
                    continue;
                }
                total_area += area;
                mesh_light_triangles.push(GpuMeshLightTriangle {
                    p0: p0.as_vec3().extend(0.0),
                    edge1: edge1.as_vec3().extend(0.0),
                    edge2: edge2.as_vec3().extend(0.0),
                    normal_area: cross.normalize().as_vec3().extend(area),
                });
                mesh_light_cdf.push(total_area);
            }
            let triangle_count = mesh_light_triangles.len() - triangle_start;
            if triangle_count == 0 || total_area <= f32::EPSILON {
                continue;
            }
            for value in &mut mesh_light_cdf[cdf_start..] {
                *value /= total_area;
            }
            let emission = material_emission
                .get(component.material.0)
                .copied()
                .unwrap_or(0.0);
            let spectrum_id = resolver
                .append_modulation(&mut light_spectrum_modulation, &material.emission.spectrum);
            let light_index = lights.len();
            lights.push(GpuLight {
                p0_emission: Vec4::new(0.0, 0.0, 0.0, emission),
                edge1_area: Vec4::new(0.0, 0.0, 0.0, total_area),
                edge2: Vec4::new(cdf_start as f32, triangle_count as f32, 0.0, spectrum_id),
                normal: Vec4::new(0.0, 0.0, 0.0, LIGHT_MESH),
            });
            mesh_light_ranges.push(MeshLightRange {
                instance_index,
                component_index,
                light_index,
                triangle_start,
                triangle_count,
                cdf_start,
            });
        }
    }

    BuiltLights {
        lights,
        mesh_triangles: mesh_light_triangles,
        mesh_cdf: mesh_light_cdf,
        spectrum: light_spectrum_modulation,
        mesh_ranges: mesh_light_ranges,
    }
}

fn empty_gpu_light(spectrum_id: f32) -> GpuLight {
    GpuLight {
        p0_emission: Vec4::ZERO,
        edge1_area: Vec4::ZERO,
        edge2: Vec4::new(0.0, 0.0, 0.0, spectrum_id),
        normal: Vec4::new(0.0, 0.0, 0.0, LIGHT_POINT),
    }
}

struct ResolvedSpectrum {
    // None is an exact identity modulation.
    spd: Option<Spd>,
    scale_ratio: f32,
    luminous_efficacy: f32,
}

struct SpectrumResolver<'a> {
    default_spd: &'a Spd,
    default_baked: &'a spectrum::EmissionSpectrum,
    resolved: HashMap<String, ResolvedSpectrum>,
}

impl<'a> SpectrumResolver<'a> {
    fn new(default_spd: &'a Spd, default_baked: &'a spectrum::EmissionSpectrum) -> Self {
        Self {
            default_spd,
            default_baked,
            resolved: HashMap::new(),
        }
    }

    fn entry(&mut self, source: &SpectrumSource) -> &ResolvedSpectrum {
        let key = source.key();
        if !self.resolved.contains_key(&key) {
            let resolved = resolve_spectrum(self.default_spd, self.default_baked, &key);
            self.resolved.insert(key.clone(), resolved);
        }
        &self.resolved[&key]
    }

    fn luminous_efficacy(&mut self, source: &SpectrumSource) -> f32 {
        self.entry(source).luminous_efficacy
    }

    fn append_modulation(&mut self, modulation: &mut Vec<f32>, source: &SpectrumSource) -> f32 {
        let (default_spd, default_baked) = (self.default_spd, self.default_baked);
        let spectrum_id = (modulation.len() / default_baked.texels.len()) as f32;
        let entry = self.entry(source);
        let spd = entry.spd.as_ref().unwrap_or(default_spd);
        modulation.extend(default_baked.texels.iter().map(|texel| {
            let default_power = default_spd.power(texel.y);
            if default_power <= 1e-9 {
                0.0
            } else {
                entry.scale_ratio * spd.power(texel.y) / default_power
            }
        }));
        spectrum_id
    }
}

fn resolve_spectrum(
    default_spd: &Spd,
    default_baked: &spectrum::EmissionSpectrum,
    key: &str,
) -> ResolvedSpectrum {
    let neutral = || ResolvedSpectrum {
        spd: None,
        scale_ratio: 1.0,
        luminous_efficacy: default_spd.luminous_efficacy(),
    };
    let Some(spd) = spectrum::named(key) else {
        return neutral();
    };
    if spd.name == default_spd.name {
        return neutral();
    }
    // The baked and aggregate RGB normalizations are intentionally different.
    let scale_ratio = spd.bake(spectrum::DEFAULT_RESOLUTION).luminance_scale()
        / default_baked.luminance_scale().max(1e-9);
    ResolvedSpectrum {
        luminous_efficacy: spd.luminous_efficacy(),
        spd: Some(spd),
        scale_ratio,
    }
}

fn analytic_light_emission(
    light: &Light,
    resolver: &mut SpectrumResolver<'_>,
    spectrum: &spectrum::EmissionSpectrum,
) -> f32 {
    let efficacy = resolver.luminous_efficacy(&light.illuminant.spectrum);
    let luminance = light.illuminant.emitted_luminance(light.extent(), efficacy);
    spectrum.emission_from_luminance(luminance)
}

const LIGHT_RECT: f32 = 1.0;
const LIGHT_DISK: f32 = 2.0;
const LIGHT_POINT: f32 = 3.0;
const LIGHT_DIRECTIONAL: f32 = 4.0;
const LIGHT_DOME: f32 = 5.0;
const LIGHT_SPHERE: f32 = 6.0;
const LIGHT_MESH: f32 = 7.0;

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
        return representative_surface_color(surface);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::scene::{
        Camera, Geometry, Illuminant, Instance, Material, MeshId, Projection, SceneData, Vertex,
    };

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
        assert_eq!(size_of::<GpuBsdf>(), 32);
        assert_eq!(size_of::<GpuTriangle>(), 48);
        assert_eq!(size_of::<GpuInstance>(), 64);
        assert_eq!(size_of::<GpuMaterialTexture>(), 32);
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
        assert_eq!(gpu.normal.w, LIGHT_SPHERE);
    }
}
