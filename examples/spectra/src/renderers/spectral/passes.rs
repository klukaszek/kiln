//! Per-frame command encoding for the spectral path tracer.

use glam::UVec2;
use kiln_rhi::{CommandBuffer, StageFlags};

use crate::base::renderer::{self as render, RenderFrame};
use crate::base::scene::Camera;

use super::film::FilmSignature;
use super::schedule::TraceBatch;
use super::shader_types::{CLEAR_THREADS, CameraGpu, ClearRoot, DisplayRoot, TraceRoot};
use super::{FILM_STRIDE, PathTracer, integrator};

struct TraceDispatch {
    camera: CameraGpu,
    extent: UVec2,
    batch: TraceBatch,
}

impl PathTracer {
    pub(super) fn record_iteration(
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
        let batch = if self.storage.lights.len() > 0 {
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
                    self.storage.bsdfs.len(),
                    self.storage.lights.len()
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

    pub(super) fn record_display(&mut self, ctx: &RenderFrame<'_>, cmd: &mut CommandBuffer) {
        let accum = self.film.accum();

        let film = self.film.extent;
        let progress = self.schedule.progress(self.film.pass_count);
        let root = self.frame_arenas.upload(
            ctx.slot,
            &DisplayRoot {
                film: accum.ptr(),
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

        cmd.set_graphics_pipeline(&self.pipelines.display);
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
        let resources = &self.storage;
        let accum = self.film.accum();
        let root = self.frame_arenas.upload(
            ctx.slot,
            &TraceRoot {
                cam_pos: camera.pos,
                cam_right: camera.right,
                cam_up: camera.up,
                cam_forward: camera.forward,
                lens: camera.lens,
                film: accum.ptr(),
                triangles: resources.triangles.gpu(),
                emissive_hits: resources.emissive_hits.gpu(),
                instances: resources.instances.gpu(),
                bsdfs: resources.bsdfs.gpu(),
                lights: resources.lights.gpu(),
                mesh_light_triangles: resources.mesh_light_triangles.gpu(),
                mesh_light_cdf: resources.mesh_light_cdf.gpu(),
                light_spectrum: resources.light_spectrum.gpu(),
                material_emission_spectrum: resources.material_emission_spectrum.gpu(),
                spectrum: resources.spectrum.gpu(),
                sensor_spectrum: resources.sensor_spectrum.gpu(),
                reflectance: resources.reflectance.gpu(),
                material_textures: resources.material_textures.gpu(),
                texture_bindings: resources.texture_bindings.gpu(),
                texture_basis: resources.texture_basis.gpu(),
                tlas: resources.accel.tlas.gpu(),
                film_width: extent.x,
                film_height: extent.y,
                pass_start: batch.start,
                pass_count: batch.count,
                settings: glam::UVec4::new(
                    self.storage.lights.len(),
                    self.schedule.pixel_stride,
                    self.schedule.pixel_stride * self.schedule.pixel_stride,
                    u32::from(self.spectral_capture),
                ),
                _pad: glam::UVec2::ZERO,
            },
        );

        cmd.set_compute_pipeline(&self.pipelines.trace);
        let stride = self.schedule.pixel_stride;
        cmd.dispatch(
            root,
            extent.x.div_ceil(stride).div_ceil(integrator::THREADS_X),
            extent.y.div_ceil(stride).div_ceil(integrator::THREADS_Y),
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
                film: accum.ptr(),
                count: float_count,
                _pad: 0,
            },
        );

        cmd.set_compute_pipeline(&self.pipelines.clear);
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
