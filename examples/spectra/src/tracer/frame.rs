//! What one frame costs the GPU: the root structs the shaders read, and the commands that fill and
//! dispatch them.
//!
//! A root is rebuilt from scratch each frame into a per-frame arena. There is no persistent binding
//! state; every buffer the shader touches is a pointer in one of these structs.

use glam::{UVec2, UVec4, Vec4};
use kiln_rhi::{AccelHandle, CommandBuffer, StageFlags, gpu_struct};

use crate::render::{self, RenderFrame};
use crate::scene::Camera;
use crate::tracer::device::GpuTextureBinding;

use super::device::{
    GpuEmissiveHit, GpuInstance, GpuLight, GpuMaterial, GpuMeshLightTriangle, GpuTriangle,
};
use super::film::FilmSignature;
use super::program::CLEAR_THREADS;
use super::sampling::TraceBatch;
use super::{FILM_STRIDE, PathTracer, program};

gpu_struct! {
    pub(super) struct ClearRoot {
        film: GpuPtr<f32>,
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
        film: GpuPtr<f32>,
        triangles: GpuPtr<GpuTriangle, Read>,
        emissive_hits: GpuPtr<GpuEmissiveHit, Read>,
        instances: GpuPtr<GpuInstance, Read>,
        materials: GpuPtr<GpuMaterial, Read>,
        lights: GpuPtr<GpuLight, Read>,
        mesh_light_triangles: GpuPtr<GpuMeshLightTriangle, Read>,
        mesh_light_cdf: GpuPtr<f32, Read>,
        light_spectrum: GpuPtr<f32, Read>,
        material_emission_spectrum: GpuPtr<f32, Read>,
        spectrum: GpuPtr<Vec4, Read>, // CDF table: (phase, wavelength, flux_shape, p_light)
        sensor_spectrum: GpuPtr<Vec4, Read>,
        reflectance: GpuPtr<f32, Read>, // [material][wavelength entry]
        texture_bindings: GpuPtr<GpuTextureBinding, Read>,
        texture_basis: GpuPtr<f32, Read>,
        // Sobol' direction table, flattened [table][byte][value].
        sobol: GpuPtr<u32, Read>,
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
        film: GpuPtr<Vec4, Read>,
        // Mean sensor response per spectral bin, resolving a capture to linear sRGB.
        cmf: GpuPtr<Vec4, Read>,
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

struct TraceDispatch {
    camera: CameraGpu,
    extent: UVec2,
    batch: TraceBatch,
}

impl PathTracer {
    pub fn record_iteration(
        &mut self,
        ctx: &RenderFrame<'_>,
        cmd: &mut CommandBuffer,
        camera: &Camera,
    ) -> render::Result<()> {
        self.frame_arenas.reset(ctx.slot);

        let extent = self.film_extent(ctx.extent);
        let camera = CameraGpu::new(camera, extent);
        let resized = self.film.extent != extent;
        let needs_clear = self.film.prepare(
            ctx.device,
            extent,
            FilmSignature::new(camera, self.schedule.pixel_stride),
        )?;
        let batch = if self.scene.lights.len() > 0 {
            self.schedule.next_batch(self.film.pass_count)
        } else {
            None
        };
        if !needs_clear && batch.is_none() {
            return Ok(());
        }

        // Order this frame's film writes after the previous in-flight frame's
        // trace writes and display reads. Recorded outside an encoder, this
        // becomes a queue-scoped barrier at the next compute encoder.
        cmd.barrier(
            StageFlags::COMPUTE | StageFlags::PIXEL_SHADER,
            StageFlags::COMPUTE,
        );

        if needs_clear {
            self.record_film_clear(ctx, cmd);
            if resized {
                eprintln!(
                    "spectral path tracer reset: {}x{} film ({}x{} display), target spp={}, passes/frame={}, materials={}, lights={}",
                    extent.x,
                    extent.y,
                    ctx.extent.x,
                    ctx.extent.y,
                    self.schedule.target_spp,
                    self.schedule.passes_per_frame,
                    self.scene.materials.len(),
                    self.scene.lights.len()
                );
            }
        }
        if let Some(batch) = batch {
            self.record_trace(
                ctx,
                cmd,
                TraceDispatch {
                    camera,
                    extent,
                    batch,
                },
            );
        }
        Ok(())
    }

    pub fn record_display(&mut self, ctx: &RenderFrame<'_>, cmd: &mut CommandBuffer) {
        let accum = self.film.accum();

        let film = self.film.extent;
        let progress = self.schedule.progress(self.film.pass_count);
        let root = self.frame_arenas.upload(
            ctx.slot,
            &DisplayRoot {
                film: accum.gpu().cast(),
                cmf: self.cmf.gpu(),
                display_width: ctx.extent.x,
                display_height: ctx.extent.y,
                film_width: film.x,
                film_height: film.y,
                film_stride: FILM_STRIDE,
                completed_samples: progress.completed_samples,
                remaining_phases: progress.remaining_phases,
                target_is_srgb: u32::from(self.display_target_is_srgb),
                pixel_stride: self.schedule.pixel_stride,
                spectral_capture: u32::from(self.spectral_capture),
            },
        );

        cmd.set_pipeline(&self.pipelines.display);
        cmd.draw(root, 3, 1, 0, 0);
    }

    fn record_trace(
        &mut self,
        ctx: &RenderFrame<'_>,
        cmd: &mut CommandBuffer,
        dispatch: TraceDispatch,
    ) {
        let TraceDispatch {
            camera,
            extent,
            batch,
            ..
        } = dispatch;
        let resources = &self.scene;
        let accum = self.film.accum();
        let root = self.frame_arenas.upload(
            ctx.slot,
            &TraceRoot {
                cam_pos: camera.pos,
                cam_right: camera.right,
                cam_up: camera.up,
                cam_forward: camera.forward,
                lens: camera.lens,
                film: accum.gpu().cast(),
                triangles: resources.triangles.gpu(),
                emissive_hits: resources.emissive_hits.gpu(),
                instances: resources.instances.gpu(),
                materials: resources.materials.gpu(),
                lights: resources.lights.gpu(),
                mesh_light_triangles: resources.mesh_light_triangles.gpu(),
                mesh_light_cdf: resources.mesh_light_cdf.gpu(),
                light_spectrum: resources.light_spectrum.gpu(),
                material_emission_spectrum: resources.material_emission_spectrum.gpu(),
                spectrum: resources.spectrum.gpu(),
                sensor_spectrum: resources.sensor_spectrum.gpu(),
                reflectance: resources.reflectance.gpu(),
                texture_bindings: resources.texture_bindings.gpu(),
                texture_basis: resources.texture_basis.gpu(),
                sobol: self.sobol.gpu(),
                tlas: resources.accel.tlas.gpu(),
                film_width: extent.x,
                film_height: extent.y,
                pass_start: batch.start,
                pass_count: batch.count,
                settings: glam::UVec4::new(
                    self.scene.lights.len(),
                    self.schedule.pixel_stride,
                    self.schedule.pixel_stride * self.schedule.pixel_stride,
                    u32::from(self.spectral_capture),
                ),
                _pad: UVec2::ZERO,
            },
        );

        cmd.set_pipeline(&self.pipelines.trace);
        let stride = self.schedule.pixel_stride;
        cmd.dispatch(
            root,
            extent
                .x
                .div_ceil(stride)
                .div_ceil(program::TRACE_THREADS[0]),
            extent
                .y
                .div_ceil(stride)
                .div_ceil(program::TRACE_THREADS[1]),
            1,
        );
        self.film.pass_count += batch.count;
        self.log_progress();
        cmd.barrier(StageFlags::COMPUTE, StageFlags::PIXEL_SHADER);
    }

    fn record_film_clear(&mut self, ctx: &RenderFrame<'_>, cmd: &mut CommandBuffer) {
        let accum = self.film.accum();
        let float_count = self.film.element_count;
        let root = self.frame_arenas.upload(
            ctx.slot,
            &ClearRoot {
                film: accum.gpu().cast(),
                count: float_count,
                _pad: 0,
            },
        );

        cmd.set_pipeline(&self.pipelines.clear);
        cmd.dispatch(root, float_count.div_ceil(CLEAR_THREADS), 1, 1);
        // Each pipeline switch opens a new compute encoder. Including the
        // pixel stage forces a queue-scoped barrier, ordering both the trace
        // and display work after the clear.
        cmd.barrier(
            StageFlags::COMPUTE,
            StageFlags::COMPUTE | StageFlags::PIXEL_SHADER,
        );
    }
}
