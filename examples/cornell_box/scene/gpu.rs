//! The scene, resident on the GPU — buffers, spectral conversions, and
//! acceleration structures, independent of any render pipeline.
//!
//! [`GpuScene`] is built once per loaded stage and shared by every renderer: the
//! raster preview reads the vertex buffer, the path tracer reads everything.
//! This is also where host colours become spectral: each material's albedo is
//! fitted to moment-based reflectance coefficients ([`spectral::fit_reflectance`])
//! and the chosen light spectrum is baked into its wavelength-sampling table,
//! so the shaders only ever see ready-to-evaluate spectral data.

use glam::{UVec4, Vec3, Vec4};
use kiln_rhi::{
    AccelerationStructure, BlasDesc, BlasMeshDesc, BuildAccelFlags, Device, GeometryFlags,
    GeometryType, GpuAddress, GpuAllocation, MemoryType, TlasDesc, TlasInstance, gpu_struct,
};

use super::spectral::{self, Spd};
use super::{Material, Scene, Vertex};

gpu_struct! {
    pub struct GpuMaterial {
        base_roughness: Vec4 as "float4",
        emission_metallic: Vec4 as "float4",
        specular_ior: Vec4 as "float4",
        coat_opacity: Vec4 as "float4",
        flags: UVec4 as "uint4",
        // xyz: Lagrange multipliers of the moment-based reflectance spectrum
        // (prep done on the CPU; the shader only evaluates). w: the emitter
        // scalar for spectral transport (0 for non-emissive materials).
        lagrange_emission: Vec4 as "float4",
    }
}

/// Ray-tracing acceleration over the scene. The instance buffer and BLAS are held
/// alive here because the TLAS references their GPU memory.
pub struct SceneAccel {
    _instance_buffer: GpuAllocation,
    _blas: AccelerationStructure,
    pub tlas: AccelerationStructure,
}

pub struct GpuScene {
    pub vertex_buffer: GpuAllocation,
    pub triangle_material_buffer: GpuAllocation,
    pub material_buffer: GpuAllocation,
    pub light_triangle_buffer: GpuAllocation,
    /// The light's baked wavelength-sampling table ([`spectral::EmissionSpectrum`]
    /// texels: rgb = sensor weight, w = phase). One spectrum per scene for now;
    /// per-light spectra need per-light tables plus flux data for MIS.
    pub spectrum_buffer: GpuAllocation,
    pub spectrum_len: u32,
    pub accel: Option<SceneAccel>,
    pub triangle_count: u32,
    pub light_count: u32,
    pub material_count: u32,
}

impl GpuScene {
    pub fn build(device: &Device, scene: &Scene, light_spectrum: &Spd) -> anyhow::Result<Self> {
        let triangle_count = scene.triangle_count();
        anyhow::ensure!(triangle_count > 0, "scene has no triangles");

        let vertex_buffer = upload_slice(device, &scene.vertices, "vertex buffer");
        let triangle_material_buffer = upload_slice(
            device,
            &scene.triangle_materials,
            "triangle material buffer",
        );

        let baked = light_spectrum.bake(spectral::DEFAULT_RESOLUTION);
        let spectrum_buffer = upload_slice(device, &baked.texels, "light spectrum table");

        let gpu_materials: Vec<GpuMaterial> = scene
            .materials
            .iter()
            .map(|material| material_to_gpu(material, &baked))
            .collect();
        let material_buffer = upload_slice(device, &gpu_materials, "material buffer");

        let light_triangles: Vec<u32> = scene
            .triangle_materials
            .iter()
            .enumerate()
            .filter_map(|(tri, &mat)| {
                scene
                    .materials
                    .get(mat as usize)
                    .is_some_and(|m| m.is_emissive())
                    .then_some(tri as u32)
            })
            .collect();
        let light_triangle_buffer = upload_slice(device, &light_triangles, "light triangle buffer");

        let accel = match build_accel(device, scene, &vertex_buffer) {
            Ok(accel) => Some(accel),
            Err(e) => {
                eprintln!("scene acceleration structures unavailable: {e}");
                None
            }
        };

        eprintln!(
            "cornell gpu scene: {triangle_count} triangles, {} materials, {} emissive triangles, light spectrum {} ({} texels), accel={}",
            gpu_materials.len(),
            light_triangles.len(),
            baked.name,
            baked.texels.len(),
            if accel.is_some() { "yes" } else { "no" },
        );

        Ok(Self {
            vertex_buffer,
            triangle_material_buffer,
            material_buffer,
            light_triangle_buffer,
            spectrum_buffer,
            spectrum_len: baked.texels.len() as u32,
            accel,
            triangle_count,
            light_count: light_triangles.len() as u32,
            material_count: gpu_materials.len() as u32,
        })
    }
}

fn material_to_gpu(material: &Material, light: &spectral::EmissionSpectrum) -> GpuMaterial {
    let fit = spectral::fit_reflectance(material.base_color);
    if fit.fit_error > 0.01 {
        eprintln!(
            "spectral fit for albedo {:?} off by {:.3} (moments {:?})",
            material.base_color, fit.fit_error, fit.trig_moments
        );
    }
    let emission_scale = if material.is_emissive() {
        light.emission_scale(material.emission)
    } else {
        0.0
    };

    GpuMaterial {
        base_roughness: material.base_color.extend(material.roughness),
        emission_metallic: material.emission.extend(material.metallic),
        specular_ior: material.specular_color.extend(material.ior),
        coat_opacity: Vec4::new(
            material.clearcoat,
            material.clearcoat_roughness,
            material.opacity,
            material.opacity_threshold,
        ),
        flags: UVec4::new(material.use_specular_workflow as u32, 0, 0, 0),
        lagrange_emission: Vec3::from_array(fit.lagranges).extend(emission_scale),
    }
}

/// Build a single-instance BLAS + TLAS over the scene's triangle soup.
fn build_accel(
    device: &Device,
    scene: &Scene,
    vertex_buffer: &GpuAllocation,
) -> anyhow::Result<SceneAccel> {
    let blas_desc = BlasDesc {
        meshes: vec![BlasMeshDesc {
            geometry_type: GeometryType::Triangles,
            flags: GeometryFlags::OPAQUE,
            vertex_buffer: vertex_buffer.gpu(),
            vertex_stride: std::mem::size_of::<Vertex>() as u64,
            vertex_count: scene.vertices.len() as u32,
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

    let instance_buffer = device
        .malloc(device.tlas_instance_stride() as u64, MemoryType::Default)
        .expect("alloc TLAS instance buffer");
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
        _instance_buffer: instance_buffer,
        _blas: blas,
        tlas,
    })
}

fn upload_slice<T: kiln_rhi::GpuPod>(device: &Device, data: &[T], label: &str) -> GpuAllocation {
    let size = std::mem::size_of_val(data).max(1) as u64;
    let buffer = device.malloc(size, MemoryType::Default).expect(label);
    if !data.is_empty() {
        buffer.upload_slice(data).expect(label);
    }
    buffer
}
