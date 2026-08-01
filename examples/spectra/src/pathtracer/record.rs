//! Per-frame command encoding for [`PathTracer`](super::PathTracer).

use glam::UVec2;
use kiln_app::FrameCtx;
use kiln_rhi::{CommandBuffer, StageFlags};

use crate::scene::Scene;
use crate::scene::gpu::{GpuGeometry, SceneAccel, SpectralGpuScene};

use super::film::FilmSignature;
use super::roots::{CLEAR_THREADS, CameraGpu, ClearRoot, DisplayRoot, TraceRoot};
use super::schedule::TraceBatch;
use super::{FILM_STRIDE, PathTracer, integrator};

impl PathTracer {
    pub fn pre_render(
        &mut self,
        ctx: &FrameCtx,
        cmd: &mut CommandBuffer,
        scene: &Scene,
        geometry: &GpuGeometry,
        gpu_scene: &SpectralGpuScene,
    ) {
        let Some(accel) = geometry.accel.as_ref() else {
            return;
        };
        self.frame_arenas.reset(ctx.slot);

        let extent = self.film_extent(ctx.extent);
        let camera = CameraGpu::from_scene(scene, extent);
        let resized = self.film.extent() != extent;
        let needs_clear = self
            .film
            .prepare(
                ctx.device,
                extent,
                FilmSignature::new(
                    camera,
                    geometry.revision(),
                    gpu_scene.revision(),
                    self.schedule.pixel_stride(),
                    FILM_STRIDE,
                ),
            )
            .expect("prepare spectral film");
        let batch = if gpu_scene.light_count > 0 {
            self.schedule.next_batch(self.film.pass_count())
        } else {
            None
        };
        if !needs_clear && batch.is_none() {
            return;
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
                    self.schedule.target_spp(),
                    self.schedule.passes_per_frame(),
                    gpu_scene.material_count,
                    gpu_scene.light_count
                );
            }
        }
        if let Some(batch) = batch {
            self.record_trace(ctx, cmd, accel, gpu_scene, camera, extent, batch);
        }
    }

    pub fn render(&mut self, ctx: &FrameCtx, cmd: &mut CommandBuffer) {
        let Some(accum) = self.film.accum() else {
            return;
        };
        if self.film.extent() != self.film_extent(ctx.extent) {
            return;
        }

        let film = self.film.extent();
        let progress = self.schedule.progress(self.film.pass_count());
        let root = self.frame_arenas.upload(
            ctx.slot,
            &DisplayRoot {
                film: accum.gpu(),
                display_width: ctx.extent.x,
                display_height: ctx.extent.y,
                film_width: film.x,
                film_height: film.y,
                film_stride: FILM_STRIDE,
                completed_samples: progress.completed_samples,
                remaining_phases: progress.remaining_phases,
                target_is_srgb: u32::from(self.display_target_is_srgb),
                _pad: UVec2::ZERO,
            },
        );

        cmd.set_graphics_pipeline(&self.pipelines.display);
        cmd.draw(root, 3, 1, 0, 0);
    }

    fn record_trace(
        &mut self,
        ctx: &FrameCtx,
        cmd: &mut CommandBuffer,
        accel: &SceneAccel,
        gpu_scene: &SpectralGpuScene,
        camera: CameraGpu,
        extent: UVec2,
        batch: TraceBatch,
    ) {
        let accum = self.film.accum().expect("film prepared");
        let root = self.frame_arenas.upload(
            ctx.slot,
            &TraceRoot {
                cam_pos: camera.pos,
                cam_right: camera.right,
                cam_up: camera.up,
                cam_forward: camera.forward,
                lens: camera.lens,
                film: accum.gpu(),
                triangles: gpu_scene.triangle_buffer.gpu(),
                bsdfs: gpu_scene.bsdf_buffer.gpu(),
                lights: gpu_scene.light_buffer.gpu(),
                spectrum: gpu_scene.spectrum_buffer.gpu(),
                lambda: gpu_scene.lambda_buffer.gpu(),
                reflectance: gpu_scene.reflectance_buffer.gpu(),
                tlas: accel.tlas.gpu(),
                film_width: extent.x,
                film_height: extent.y,
                pass_start: batch.start,
                pass_count: batch.count,
                target_passes: batch.target,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            },
        );

        cmd.set_compute_pipeline(&self.pipelines.trace);
        let stride = self.schedule.pixel_stride();
        cmd.dispatch(
            root,
            extent.x.div_ceil(stride).div_ceil(integrator::THREADS_X),
            extent.y.div_ceil(stride).div_ceil(integrator::THREADS_Y),
            1,
        );
        self.film.add_passes(batch.count);
        self.log_progress();
        cmd.barrier(StageFlags::COMPUTE, StageFlags::PIXEL_SHADER);
    }

    fn record_film_clear(&mut self, ctx: &FrameCtx, cmd: &mut CommandBuffer) {
        let accum = self.film.accum().expect("film prepared");
        let film = self.film.extent();
        let float_count = film
            .x
            .checked_mul(film.y)
            .and_then(|pixels| pixels.checked_mul(FILM_STRIDE))
            .expect("spectral film element count exceeds u32");
        let root = self.frame_arenas.upload(
            ctx.slot,
            &ClearRoot {
                film: accum.gpu(),
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
