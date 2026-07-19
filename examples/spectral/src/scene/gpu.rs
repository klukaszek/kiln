//! Renderer-facing GPU scene data.
//!
//! [`GpuGeometry`] contains the vertex buffer and optional ray-tracing
//! acceleration shared by renderers. [`SpectralGpuScene`] contains only path
//! transport data and wavelength tables.

use glam::{Vec3, Vec4};
use kiln_rhi::{
    AccelerationStructure, BlasDesc, BlasMeshDesc, BuildAccelFlags, Device, GeometryFlags,
    GeometryType, GpuAddress, GpuAllocation, MemoryType, TlasDesc, TlasInstance, gpu_struct,
};
use std::sync::atomic::{AtomicU64, Ordering};

use super::spectral::{self, Spd};
use super::{Material, Scene, Vertex};

static NEXT_GPU_REVISION: AtomicU64 = AtomicU64::new(1);

gpu_struct! {
    pub struct GpuBsdf {
        alpha: f32,
        alpha2: f32,
        metallic: f32,
        f0_dielectric: f32,
        spec_prob: f32,
        _pad: [f32; 3],
    }
}

gpu_struct! {
    pub struct GpuLight {
        p0_emission: Vec4, // xyz: first vertex, w: spectral emission scale
        edge1_area: Vec4,  // xyz: p1 - p0, w: triangle area
        edge2: Vec4,       // xyz: p2 - p0
        normal: Vec4,      // xyz: unit geometric normal
    }
}

gpu_struct! {
    pub struct GpuTriangle {
        normal_area: Vec4, // xyz: unit geometric normal, w: triangle area
        material_id: u32,
        emission: f32,
        _pad: [f32; 2],
    }
}

/// Ray-tracing acceleration over the scene. The instance buffer and BLAS are held
/// alive here because the TLAS references their GPU memory.
pub struct SceneAccel {
    instance_buffer: GpuAllocation,
    blas: AccelerationStructure,
    pub tlas: AccelerationStructure,
}

pub struct GpuGeometry {
    pub vertex_buffer: GpuAllocation,
    pub accel: Option<SceneAccel>,
    pub triangle_count: u32,
    revision: u64,
}

pub struct SpectralGpuScene {
    pub triangle_buffer: GpuAllocation,
    pub bsdf_buffer: GpuAllocation,
    pub light_buffer: GpuAllocation,
    pub spectrum_buffer: GpuAllocation,
    pub lambda_buffer: GpuAllocation,
    pub reflectance_buffer: GpuAllocation,
    pub spectrum_len: u32,
    pub light_count: u32,
    pub material_count: u32,
    revision: u64,
}

impl GpuGeometry {
    pub fn build(device: &Device, scene: &Scene) -> anyhow::Result<Self> {
        let triangle_count = u32::try_from(scene.triangle_count())?;
        anyhow::ensure!(triangle_count > 0, "scene has no triangles");
        let vertex_buffer = device.upload_slice(&scene.vertices)?;
        let accel = match build_accel(device, scene, &vertex_buffer) {
            Ok(accel) => Some(accel),
            Err(e) => {
                eprintln!("scene acceleration structures unavailable: {e}");
                None
            }
        };
        Ok(Self {
            vertex_buffer,
            accel,
            triangle_count,
            revision: NEXT_GPU_REVISION.fetch_add(1, Ordering::Relaxed),
        })
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn destroy(self, device: &Device) {
        if let Some(accel) = self.accel {
            accel.destroy(device);
        }
        device.free(self.vertex_buffer);
    }
}

impl SpectralGpuScene {
    pub fn build(device: &Device, scene: &Scene, light_spectrum: &Spd) -> anyhow::Result<Self> {
        let triangle_count = scene.triangle_count();
        anyhow::ensure!(triangle_count > 0, "scene has no triangles");
        anyhow::ensure!(
            scene.vertices.len().is_multiple_of(3)
                && scene.triangle_materials.len() == triangle_count,
            "triangle material count does not match geometry"
        );
        anyhow::ensure!(
            scene
                .triangle_materials
                .iter()
                .all(|&material| (material as usize) < scene.materials.len()),
            "triangle references an invalid material"
        );

        let baked = light_spectrum.bake(spectral::DEFAULT_RESOLUTION);
        let material_fits: Vec<spectral::ReflectanceSpectrum> = scene
            .materials
            .iter()
            .map(|material| spectral::fit_reflectance(material.base_color))
            .collect();
        for (material, fit) in scene.materials.iter().zip(&material_fits) {
            if fit.fit_error > 0.01 {
                eprintln!(
                    "spectral fit for albedo {:?} off by {:.3} (moments {:?})",
                    material.base_color, fit.fit_error, fit.trig_moments
                );
            }
        }
        let gpu_bsdfs: Vec<GpuBsdf> = scene.materials.iter().map(bsdf_to_gpu).collect();
        let emission_scales: Vec<f32> = scene
            .materials
            .iter()
            .map(|material| {
                if material.is_emissive() {
                    baked.emission_scale(material.emission)
                } else {
                    0.0
                }
            })
            .collect();
        let reflectance_lut = build_reflectance_lut(&material_fits, &baked);
        let triangles: Vec<GpuTriangle> = scene
            .vertices
            .chunks_exact(3)
            .zip(&scene.triangle_materials)
            .map(|(vertices, &mat)| {
                let (normal, area) = triangle_normal_area(vertices);
                GpuTriangle {
                    normal_area: normal.extend(area),
                    material_id: mat,
                    emission: emission_scales[mat as usize],
                    _pad: [0.0; 2],
                }
            })
            .collect();
        let lights: Vec<GpuLight> = scene
            .triangle_materials
            .iter()
            .enumerate()
            .filter(|&(_, &mat)| scene.materials[mat as usize].is_emissive())
            .map(|(tri, &mat)| {
                let vertices = &scene.vertices[tri * 3..tri * 3 + 3];
                let p0 = vertices[0].pos.truncate();
                let edge1 = vertices[1].pos.truncate() - p0;
                let edge2 = vertices[2].pos.truncate() - p0;
                let (normal, area) = triangle_normal_area(vertices);
                GpuLight {
                    p0_emission: p0.extend(emission_scales[mat as usize]),
                    edge1_area: edge1.extend(area),
                    edge2: edge2.extend(0.0),
                    normal: normal.extend(0.0),
                }
            })
            .collect();

        let spectrum_len = u32::try_from(baked.texels.len())?;
        let light_count = u32::try_from(lights.len())?;
        let material_count = u32::try_from(gpu_bsdfs.len())?;
        let spectrum_buffer = device.upload_slice(&baked.texels)?;
        let lambda_buffer = device.upload_slice(&baked.lambda_texels)?;
        let bsdf_buffer = device.upload_slice(&gpu_bsdfs)?;
        let reflectance_buffer = device.upload_slice(&reflectance_lut)?;
        let triangle_buffer = device.upload_slice(&triangles)?;
        let light_buffer = device.upload_slice(&lights)?;

        eprintln!(
            "spectral gpu scene: {triangle_count} triangles, {} materials, {} emissive triangles, light spectrum {} ({} texels)",
            gpu_bsdfs.len(),
            lights.len(),
            baked.name,
            baked.texels.len(),
        );

        Ok(Self {
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

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn destroy(self, device: &Device) {
        device.free(self.triangle_buffer);
        device.free(self.bsdf_buffer);
        device.free(self.light_buffer);
        device.free(self.spectrum_buffer);
        device.free(self.lambda_buffer);
        device.free(self.reflectance_buffer);
    }
}

impl SceneAccel {
    fn destroy(self, device: &Device) {
        let Self {
            instance_buffer,
            blas,
            tlas,
        } = self;
        drop(tlas);
        drop(blas);
        device.free(instance_buffer);
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

fn build_accel(
    device: &Device,
    scene: &Scene,
    vertex_buffer: &GpuAllocation,
) -> anyhow::Result<SceneAccel> {
    let vertex_count = u32::try_from(scene.vertices.len())?;
    let blas_desc = BlasDesc {
        meshes: vec![BlasMeshDesc {
            geometry_type: GeometryType::Triangles,
            flags: GeometryFlags::OPAQUE,
            vertex_buffer: vertex_buffer.gpu(),
            vertex_stride: std::mem::size_of::<Vertex>() as u64,
            vertex_count,
            index_buffer: GpuAddress(0),
            index_count: 0,
            aabb_buffer: GpuAddress(0),
            aabb_count: 0,
        }],
        flags: BuildAccelFlags::PREFER_FAST_TRACE,
    };
    let blas = device.create_blas(&blas_desc)?;
    {
        let mut cmd = device.create_command_buffer()?;
        cmd.build_blas(&blas, &blas_desc);
        cmd.end();
        let queue = device.queue();
        queue.submit(cmd)?;
        queue.wait_idle();
    }

    let instance_buffer =
        device.malloc(device.tlas_instance_stride() as u64, MemoryType::Default)?;
    device.write_tlas_instance(
        &instance_buffer,
        0,
        &TlasInstance {
            transform: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            instance_custom_index_and_mask: 0xFF << 24,
            instance_sbt_offset_and_flags: 0,
            acceleration_structure_reference: blas.gpu(),
        },
    )?;

    let tlas_desc = TlasDesc {
        instance_buffer: instance_buffer.gpu(),
        instance_count: 1,
        flags: BuildAccelFlags::PREFER_FAST_TRACE,
    };
    let tlas = device.create_tlas(&tlas_desc)?;
    {
        let mut cmd = device.create_command_buffer()?;
        cmd.build_tlas(&tlas, &tlas_desc);
        cmd.end();
        let queue = device.queue();
        queue.submit(cmd)?;
        queue.wait_idle();
    }

    Ok(SceneAccel {
        instance_buffer,
        blas,
        tlas,
    })
}
