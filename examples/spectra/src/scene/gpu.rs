//! Renderer-facing GPU scene data.
//!
//! [`GpuGeometry`] contains the vertex buffer and optional ray-tracing
//! acceleration shared by renderers. [`SpectralGpuScene`] contains only path
//! transport data and wavelength tables.

mod accel;
mod spectral_scene;

use glam::Vec4;
use kiln_rhi::{Device, GpuAllocation, RhiError, gpu_struct};
use std::sync::atomic::{AtomicU64, Ordering};

pub use accel::SceneAccel;

use super::Scene;

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
        let accel = match SceneAccel::build(device, scene, &vertex_buffer) {
            Ok(accel) => Some(accel),
            Err(error)
                if matches!(
                    error.downcast_ref::<RhiError>(),
                    Some(RhiError::Unsupported(_))
                ) =>
            {
                eprintln!("scene acceleration structures unavailable: {error}");
                None
            }
            Err(error) => {
                device.free(vertex_buffer);
                return Err(error);
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
    pub fn build(
        device: &Device,
        scene: &Scene,
        light_spectrum: &super::spectral::Spd,
    ) -> anyhow::Result<Self> {
        spectral_scene::build(device, scene, light_spectrum)
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
