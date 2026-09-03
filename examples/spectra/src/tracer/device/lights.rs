//! The light list the integrator samples: analytic lights, emissive mesh components, and the
//! spectral tables that give each one its colour.
//!
//! Every emitter reaches the GPU as a `GpuLight` regardless of whether it came from a `UsdLux` prim
//! or an emissive material, so next-event estimation has one array to sample.

use std::collections::HashMap;

use glam::{DVec3, Vec4};

use crate::scene::{EmitterExtent, Light, LightKind, Scene, SpectrumSource};

use super::super::spectrum::{self, Spd};
use super::geometry::world_triangle_area;
use super::records::{GpuEmissiveHit, GpuLight, GpuMeshLightTriangle};
use super::tags::LightTag;

/// Which scene emissive component a mesh light came from, so emissive hits can point back at it.
#[derive(Clone, Copy, Debug)]
pub(super) struct MeshLightRange {
    pub(super) instance_index: usize,
    pub(super) component_index: usize,
    pub(super) light_index: usize,
}

pub(super) struct BuiltLights {
    pub(super) lights: Vec<GpuLight>,
    pub(super) mesh_triangles: Vec<GpuMeshLightTriangle>,
    pub(super) mesh_cdf: Vec<f32>,
    pub(super) spectrum: Vec<f32>,
    /// Where each mesh light's triangles landed, used to resolve emissive hits at build time.
    pub(super) mesh_ranges: Vec<MeshLightRange>,
}

pub(super) struct LightingArrays {
    pub(super) emission: Vec<f32>,
    pub(super) material_spectrum: Vec<f32>,
    pub(super) emissive_hits: Vec<GpuEmissiveHit>,
    pub(super) lights: BuiltLights,
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
pub(super) fn build_material_emission_spectrum(
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

pub(super) fn build_lighting_arrays(
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

pub(super) fn build_analytic_gpu_light(light: &Light, emission: f32, spectrum_id: f32) -> GpuLight {
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
                normal: cross.normalize().as_vec3().extend(LightTag::Rect.as_f32()),
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
                normal: cross.normalize().as_vec3().extend(LightTag::Disk.as_f32()),
            }
        }
        LightKind::Point => {
            analytic_point_light(light, emission, spectrum_id, LightTag::Point.as_f32())
        }
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
                normal: Vec4::new(radius, 0.0, 0.0, LightTag::Sphere.as_f32()),
            }
        }
        LightKind::Directional { .. } => {
            analytic_point_light(light, emission, spectrum_id, LightTag::Directional.as_f32())
        }
        LightKind::Dome => {
            analytic_point_light(light, emission, spectrum_id, LightTag::Dome.as_f32())
        }
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
                normal: Vec4::new(0.0, 0.0, 0.0, LightTag::Mesh.as_f32()),
            });
            mesh_light_ranges.push(MeshLightRange {
                instance_index,
                component_index,
                light_index,
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
        normal: Vec4::new(0.0, 0.0, 0.0, LightTag::Point.as_f32()),
    }
}

pub(super) struct ResolvedSpectrum {
    // None is an exact identity modulation.
    spd: Option<Spd>,
    scale_ratio: f32,
    luminous_efficacy: f32,
}

pub(super) struct SpectrumResolver<'a> {
    default_spd: &'a Spd,
    default_baked: &'a spectrum::EmissionSpectrum,
    resolved: HashMap<String, ResolvedSpectrum>,
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

impl<'a> SpectrumResolver<'a> {
    pub(super) fn new(default_spd: &'a Spd, default_baked: &'a spectrum::EmissionSpectrum) -> Self {
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

    pub(super) fn append_modulation(
        &mut self,
        modulation: &mut Vec<f32>,
        source: &SpectrumSource,
    ) -> f32 {
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
