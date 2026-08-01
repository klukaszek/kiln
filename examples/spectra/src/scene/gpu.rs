//! Renderer-facing GPU scene data.
//!
//! [`GpuGeometry`] contains the vertex buffer and optional ray-tracing
//! acceleration shared by renderers. [`SpectralGpuScene`] contains only path
//! transport data and wavelength tables.

mod accel;
mod spectral_scene;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use glam::Vec4;
use kiln_rhi::{Device, GpuAllocation, GpuPod, MemoryType, RhiError, gpu_struct};

pub use accel::SceneAccel;

use super::{Scene, Vertex};

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
    /// Compact position-only stream consumed by the ray-tracing BLAS. The raster
    /// preview still uses `vertex_buffer`, which also carries normals and color.
    ray_vertex_buffer: Option<GpuAllocation>,
    ray_index_buffer: Option<GpuAllocation>,
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
        let (ray_vertices, ray_indices) = build_ray_geometry(&scene.vertices);
        let ray_vertex_count = u32::try_from(ray_vertices.len())?;
        let ray_index_count = u32::try_from(ray_indices.len())?;
        let mut uploads = GpuUploadBatch::new(device);
        uploads.upload(&scene.vertices)?;
        uploads.upload(&ray_vertices)?;
        uploads.upload(&ray_indices)?;
        let [vertex_buffer, ray_vertex_buffer, ray_index_buffer] = uploads.finish()?;
        let (accel, ray_vertex_buffer, ray_index_buffer) = match SceneAccel::build(
            device,
            &ray_vertex_buffer,
            ray_vertex_count,
            &ray_index_buffer,
            ray_index_count,
        ) {
            Ok(accel) => (Some(accel), Some(ray_vertex_buffer), Some(ray_index_buffer)),
            Err(error)
                if matches!(
                    error.downcast_ref::<RhiError>(),
                    Some(RhiError::Unsupported(_))
                ) =>
            {
                eprintln!("scene acceleration structures unavailable: {error}");
                device.free(ray_vertex_buffer);
                device.free(ray_index_buffer);
                (None, None, None)
            }
            Err(error) => {
                device.free(ray_vertex_buffer);
                device.free(ray_index_buffer);
                device.free(vertex_buffer);
                return Err(error);
            }
        };
        Ok(Self {
            vertex_buffer,
            ray_vertex_buffer,
            ray_index_buffer,
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
        if let Some(ray_vertex_buffer) = self.ray_vertex_buffer {
            device.free(ray_vertex_buffer);
        }
        if let Some(ray_index_buffer) = self.ray_index_buffer {
            device.free(ray_index_buffer);
        }
        device.free(self.vertex_buffer);
    }
}

/// Stage immutable scene data through host-visible memory, then keep the final
/// allocations device-local. All uploads share one copy submission.
pub(super) struct GpuUploadBatch<'a> {
    device: &'a Device,
    allocations: Vec<GpuAllocation>,
    staging: Vec<GpuAllocation>,
}

impl<'a> GpuUploadBatch<'a> {
    pub(super) fn new(device: &'a Device) -> Self {
        Self {
            device,
            allocations: Vec::new(),
            staging: Vec::new(),
        }
    }

    pub(super) fn upload<T: GpuPod>(&mut self, data: &[T]) -> anyhow::Result<()> {
        let size = std::mem::size_of_val(data).max(1) as u64;
        let allocation = self.device.malloc(size, MemoryType::GpuOnly)?;
        let staging = match self.device.upload_slice(data) {
            Ok(staging) => staging,
            Err(error) => {
                self.device.free(allocation);
                return Err(error.into());
            }
        };
        self.allocations.push(allocation);
        self.staging.push(staging);
        Ok(())
    }

    pub(super) fn finish<const N: usize>(mut self) -> anyhow::Result<[GpuAllocation; N]> {
        let mut cmd = self.device.create_command_buffer()?;
        for (destination, staging) in self.allocations.iter().zip(&self.staging) {
            cmd.memcpy(destination.gpu(), staging.gpu(), destination.size());
        }
        cmd.end();
        self.device.queue().submit(cmd)?;
        self.device.queue().wait_idle();

        for staging in self.staging.drain(..) {
            self.device.free(staging);
        }

        match std::mem::take(&mut self.allocations).try_into() {
            Ok(allocations) => Ok(allocations),
            Err(allocations) => {
                self.allocations = allocations;
                anyhow::bail!("internal GPU upload count mismatch")
            }
        }
    }
}

impl Drop for GpuUploadBatch<'_> {
    fn drop(&mut self) {
        for allocation in self.allocations.drain(..) {
            self.device.free(allocation);
        }
        for staging in self.staging.drain(..) {
            self.device.free(staging);
        }
    }
}

/// Multiply-xor hasher for the 12-byte position keys used when welding ray vertices. `SipHash` is
/// keyed and DoS-resistant, neither of which matters for local mesh data, and costs several rounds
/// per vertex; float bit patterns are well distributed enough for one multiply-rotate per word.
#[derive(Default, Clone, Copy)]
struct PositionHasher(u64);

impl std::hash::Hasher for PositionHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.write_u32(u32::from(byte));
        }
    }

    fn write_u32(&mut self, value: u32) {
        // FxHash constant; the rotate keeps high bits alive when the map masks to a bucket.
        self.0 = (self.0.rotate_left(5) ^ u64::from(value)).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

type PositionBuildHasher = std::hash::BuildHasherDefault<PositionHasher>;

/// Build the ray-facing representation independently of the raster vertex stream.
/// Raster vertices are intentionally duplicated for per-corner normals, while the
/// ray tracer uses only geometric triangle normals and can share identical positions.
fn build_ray_geometry(vertices: &[Vertex]) -> (Vec<[f32; 3]>, Vec<u32>) {
    let mut positions = Vec::with_capacity(vertices.len());
    let mut indices = Vec::with_capacity(vertices.len());
    let mut position_indices: HashMap<[u32; 3], u32, PositionBuildHasher> =
        HashMap::with_capacity_and_hasher(vertices.len(), PositionBuildHasher::default());

    for vertex in vertices {
        let position = vertex.pos.truncate().to_array();
        let key = position.map(f32::to_bits);
        let index = match position_indices.entry(key) {
            std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let index = u32::try_from(positions.len()).expect("ray vertex count exceeds u32");
                positions.push(position);
                entry.insert(index);
                index
            }
        };
        indices.push(index);
    }

    (positions, indices)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn vertex(position: [f32; 3]) -> Vertex {
        Vertex {
            pos: Vec4::from_array([position[0], position[1], position[2], 1.0]),
            normal: Vec4::ZERO,
            color: Vec4::ZERO,
        }
    }

    #[test]
    fn ray_geometry_deduplicates_positions_in_triangle_order() {
        let vertices = [
            vertex([0.0, 0.0, 0.0]),
            vertex([1.0, 0.0, 0.0]),
            vertex([0.0, 1.0, 0.0]),
            vertex([0.0, 0.0, 0.0]),
            vertex([0.0, 1.0, 0.0]),
            vertex([1.0, 1.0, 0.0]),
        ];
        let (positions, indices) = build_ray_geometry(&vertices);

        assert_eq!(positions.len(), 4);
        assert_eq!(indices, [0, 1, 2, 0, 2, 3]);
    }
}
