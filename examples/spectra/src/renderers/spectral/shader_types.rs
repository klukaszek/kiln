//! Host and shader argument layouts for spectral tracing.

use glam::{UVec2, UVec4, Vec4};
use kiln_rhi::{AccelHandle, gpu_struct};

use crate::base::gpu::GpuTextureBinding;
use crate::base::scene::Camera;
use crate::renderers::spectral::scene::{
    GpuBsdf, GpuEmissiveHit, GpuInstance, GpuLight, GpuMaterialTexture, GpuMeshLightTriangle,
    GpuTriangle,
};

pub(super) const CLEAR_SOURCE: &str = /*slang*/
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

pub(super) const CLEAR_THREADS: u32 = 256;

gpu_struct! {
    pub(super) struct ClearRoot {
        film: GpuPtr<f32> as "float*",
        count: u32, // number of f32 in the film (stride * pixels)
        _pad: u32,
    }
}

gpu_struct! {
    pub(super) struct TraceRoot {
        cam_pos: Vec4,
        cam_right: Vec4,
        cam_up: Vec4,
        cam_forward: Vec4,
        lens: Vec4,
        film: GpuPtr<f32> as "float*",
        triangles: GpuPtr<GpuTriangle> as "Ptr<GpuTriangle, Access.Read>",
        emissive_hits: GpuPtr<GpuEmissiveHit> as "Ptr<GpuEmissiveHit, Access.Read>",
        instances: GpuPtr<GpuInstance> as "Ptr<GpuInstance, Access.Read>",
        bsdfs: GpuPtr<GpuBsdf> as "Ptr<GpuBsdf, Access.Read>",
        lights: GpuPtr<GpuLight> as "Ptr<GpuLight, Access.Read>",
        mesh_light_triangles: GpuPtr<GpuMeshLightTriangle> as "Ptr<GpuMeshLightTriangle, Access.Read>",
        mesh_light_cdf: GpuPtr<f32> as "Ptr<float, Access.Read>",
        light_spectrum: GpuPtr<f32> as "Ptr<float, Access.Read>",
        material_emission_spectrum: GpuPtr<f32> as "Ptr<float, Access.Read>",
        spectrum: GpuPtr<Vec4> as "Ptr<float4, Access.Read>", // CDF table: (phase, wavelength, flux_shape, p_light)
        sensor_spectrum: GpuPtr<Vec4> as "Ptr<float4, Access.Read>",
        reflectance: GpuPtr<f32> as "Ptr<float, Access.Read>", // [material][wavelength entry]
        material_textures: GpuPtr<GpuMaterialTexture> as "Ptr<GpuMaterialTexture, Access.Read>",
        texture_bindings: GpuPtr<GpuTextureBinding> as "Ptr<GpuTextureBinding, Access.Read>",
        texture_basis: GpuPtr<f32> as "Ptr<float, Access.Read>",
        tlas: AccelHandle,
        film_width: u32,
        film_height: u32,
        pass_start: u32,
        pass_count: u32,
        settings: UVec4,
        _pad: UVec2,
    }
}

gpu_struct! {
    pub(super) struct DisplayRoot {
        film: GpuPtr<Vec4> as "Ptr<float4, Access.Read>",
        display_width: u32,
        display_height: u32,
        film_width: u32,
        film_height: u32,
        film_stride: u32,
        completed_samples: u32,
        remaining_phases: u32,
        target_is_srgb: u32,
        pixel_stride: u32,
        spectral_capture: u32,
    }
}

#[derive(Clone, Copy, PartialEq)]
pub(super) struct CameraGpu {
    pub pos: Vec4,
    pub right: Vec4,
    pub up: Vec4,
    pub forward: Vec4,
    pub lens: Vec4,
}

impl CameraGpu {
    pub(super) fn new(camera: &Camera, extent: UVec2) -> Self {
        // The world matrix is column-vector glam; its x/y/z columns are the
        // camera's right/up/back axes, w its position.
        let world = &camera.world;
        let aspect = extent.x as f32 / extent.y.max(1) as f32;
        let tan_half_fovy = (camera.projection.vertical_fov_rad * 0.5).tan();
        let basis = |axis: glam::DVec4| axis.truncate().normalize().as_vec3().extend(0.0);

        Self {
            pos: world.w_axis.as_vec4(),
            right: basis(world.x_axis),
            up: basis(world.y_axis),
            forward: basis(-world.z_axis),
            lens: Vec4::new(aspect, tan_half_fovy, 0.0, 0.0),
        }
    }
}
