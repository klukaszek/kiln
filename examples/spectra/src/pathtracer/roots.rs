//! GPU argument layouts shared with the path-tracing shaders.

use glam::{DVec3, UVec2, Vec4};
use kiln_rhi::{AccelHandle, GpuAddress, gpu_struct};

use crate::scene::Scene;

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
        film: GpuAddress as "float*", // spectral film: per pixel [bin_0..bin_N]
        triangles: GpuAddress as "Ptr<GpuTriangle, Access.Read>",
        bsdfs: GpuAddress as "Ptr<GpuBsdf, Access.Read>",
        lights: GpuAddress as "Ptr<GpuLight, Access.Read>",
        spectrum: GpuAddress as "Ptr<float4, Access.Read>", // CDF table: (phase, wavelength, flux_shape, p_light)
        lambda: GpuAddress as "Ptr<float4, Access.Read>", // uniform-λ MIS table, same texel layout
        reflectance: GpuAddress as "Ptr<float, Access.Read>", // [material][light/uniform table][wavelength entry]
        tlas: AccelHandle,
        film_width: u32,
        film_height: u32,
        pass_start: u32,
        pass_count: u32,
        target_passes: u32,
        _pad0: u32,
        _pad1: u32,
        _pad2: u32,
    }
}

gpu_struct! {
    pub struct DisplayRoot {
        film: GpuAddress as "Ptr<float4, Access.Read>",
        display_width: u32,
        display_height: u32,
        film_width: u32,
        film_height: u32,
        film_stride: u32,
        completed_samples: u32,
        remaining_phases: u32,
        target_is_srgb: u32,
        _pad: UVec2,
    }
}

#[derive(Clone, Copy, PartialEq)]
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
        let basis =
            |axis: glam::DVec4| axis.truncate().normalize_or(DVec3::Z).as_vec3().extend(0.0);

        Self {
            pos: world.w_axis.as_vec4(),
            right: basis(world.x_axis),
            up: basis(world.y_axis),
            forward: basis(-world.z_axis),
            lens: Vec4::new(aspect, tan_half_fovy, 0.0, 0.0),
        }
    }
}
