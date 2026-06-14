//! Per-pass GPU argument layouts and the camera basis the trace kernel
//! consumes. Roots are transient — bump-allocated per frame slot, never
//! persistent — and the [`gpu_struct!`] layouts here are mirrored into Slang via
//! their generated `SLANG` constants. Also holds the tiny film-clear shader,
//! which is paired with [`ClearRoot`].

use glam::{DVec3, UVec2, UVec4, Vec4};
use kiln_rhi::{GpuAddress, gpu_struct};

use crate::scene::Scene;
/// Zero the spectral film on the GPU (one thread per f32 — bins and counts
/// alike). Recorded ahead of the trace pass whenever the film resets, so
/// history invalidation needs no CPU writes and stays ordered against in-flight
/// frames by queue submission order.
pub const CLEAR_SOURCE: &str = /*slang*/
    r#"
[shader("compute")]
[numthreads(256, 1, 1)]
void clearMain(uint3 tid : SV_DispatchThreadID, uniform ClearRoot* r)
{
    if (tid.x < r.count) {
        r.film[tid.x] = 0.0;
    }
}
"#;

pub const CLEAR_THREADS: u32 = 256;

gpu_struct! {
    pub struct ClearRoot {
        film: GpuAddress as "float*",
        count: u32, // number of f32 in the film (stride * pixels)
        _pad: u32,
    }
}

gpu_struct! {
    pub struct TraceRoot {
        cam_pos: Vec4,
        cam_right: Vec4,
        cam_up: Vec4,
        cam_forward: Vec4,
        lens: Vec4,
        film: GpuAddress as "float*", // spectral film: per pixel [bin_0..bin_N, count]
        verts: GpuAddress as "Vertex*",
        triangle_materials: GpuAddress as "uint*",
        materials: GpuAddress as "GpuMaterial*",
        light_triangles: GpuAddress as "uint*",
        spectrum: GpuAddress as "float4*", // CDF table: (phase, wavelength, flux_shape, p_light)
        lambda: GpuAddress as "float4*", // uniform-λ MIS table, same texel layout
        // Bindless TLAS handle. As the 8th 8-byte slot it also keeps the struct 16-byte-aligned
        // for the trailing UVec4s (so gpu_struct needs no explicit pad). The handle is the AS
        // device address (Vulkan) / gpuResourceID (Metal), both from `accel.tlas.gpu()`.
        // See docs/design/vulkan-binding-convention.md.
        tlas: GpuAddress as "DescriptorHandle<RaytracingAccelerationStructure>",
        dims0: UVec4, // film width, film height, sample_index, max_spp
        dims1: UVec4, // tri_count, light_count, samples_per_frame, spectrum_len
        dims2: UVec4, // spectral_bins, n_light_lanes, n_uniform_lanes, 0
    }
}

gpu_struct! {
    pub struct DisplayRoot {
        dims: UVec4, // display width, display height, spectral_bins, target_is_srgb
        film_dims: UVec4, // film width, film height, film_stride, 0
        film: GpuAddress as "float*", // spectral film: per pixel [bin_0..bin_N, count]
        cmf: GpuAddress as "float4*", // per-bin linear-sRGB sensor response
    }
}

/// Camera basis in the shape the trace kernel consumes.
pub struct CameraGpu {
    pub pos: Vec4,
    pub right: Vec4,
    pub up: Vec4,
    pub forward: Vec4,
    pub lens: Vec4,
}

impl CameraGpu {
    pub fn from_scene(scene: &Scene, extent: UVec2) -> Self {
        // The world matrix is column-vector glam; its x/y/z columns are the
        // camera's right/up/back axes, w its position.
        let world = &scene.camera.world;
        let aspect = extent.x as f32 / extent.y.max(1) as f32;
        let tan_half_fovy = (scene.camera.usd.vertical_fov_rad() * 0.5).tan();
        let basis = |axis: glam::DVec4| axis.truncate().normalize_or(DVec3::Z).as_vec3().extend(0.0);

        Self {
            pos: world.w_axis.as_vec4(),
            right: basis(world.x_axis),
            up: basis(world.y_axis),
            forward: basis(-world.z_axis),
            lens: Vec4::new(aspect, tan_half_fovy, 0.0, 0.0),
        }
    }

    /// Fold every accumulated-sample-invalidating camera input into a film key.
    pub fn film_key(&self, seed: u64) -> u64 {
        let mut key = seed;
        for vec in [self.pos, self.right, self.up, self.forward, self.lens] {
            for component in vec.to_array() {
            // FNV-1a over the raw bits: cheap, stable, and exact-equality
            // semantics (any camera change at all restarts accumulation).
                key ^= component.to_bits() as u64;
                key = key.wrapping_mul(0x100000001b3);
            }
        }
        key
    }
}
