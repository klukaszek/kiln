//! Progressive spectral GPU path tracer.
//!
//! [`PathTracer`] owns scene-independent pipelines, scheduling, and the
//! progressive film. Geometry and spectral scene data remain externally owned.

mod display;
mod film;
mod integrator;
mod pipelines;
mod roots;
mod sampler;
mod schedule;

use glam::{UVec2, Vec3, Vec4};
use kiln_rhi::{
    BufferDesc, BumpAllocator, CommandBuffer, CompareOp, DepthFlags, DepthStencilState, Device,
    Format, GpuAllocation, MAX_FRAMES_IN_FLIGHT, MemoryType, StageFlags,
};

use kiln_app::FrameCtx;

use crate::scene::Scene;
use crate::scene::gpu::{GpuGeometry, SpectralGpuScene};
use crate::scene::spectral;
use film::{Film, FilmSignature};
use pipelines::Pipelines;
use roots::{CLEAR_THREADS, CameraGpu, ClearRoot, DisplayRoot, TraceRoot};
use schedule::SpatialSchedule;

pub const DEFAULT_TARGET_SPP: u32 = 1024;
/// One spatial pass per present.
pub const DEFAULT_PASSES_PER_FRAME: u32 = 1;

const FILM_STRIDE: u32 = spectral::SPECTRAL_BINS as u32;

/// Light-importance and uniform wavelength lanes carried by each path.
const N_LIGHT_LANES: u32 = 2;
const N_UNIFORM_LANES: u32 = 2;

const FRAME_ARENA_SIZE: u64 = 4096;

pub struct PathTracer {
    pipelines: Pipelines,
    frame_arenas: [BumpAllocator; MAX_FRAMES_IN_FLIGHT],
    film: Film,
    schedule: SpatialSchedule,
    render_scale: u32,
    cmf_bins: GpuAllocation,
    cmf_bins_cpu: Vec<Vec3>,
    display_target_is_srgb: bool,
}

impl PathTracer {
    pub fn new(
        device: &Device,
        color_format: Format,
        target_spp: u32,
        passes_per_frame: u32,
        render_scale: u32,
        pixel_stride: u32,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(render_scale > 0, "render scale must be greater than zero");
        let schedule = SpatialSchedule::new(target_spp, passes_per_frame, pixel_stride)?;
        let pipelines = Pipelines::new(device, color_format)?;

        let frame_arenas = std::array::from_fn(|slot| {
            BumpAllocator::new(
                device
                    .create_buffer(&BufferDesc {
                        size: FRAME_ARENA_SIZE,
                        memory: MemoryType::Default,
                        label: Some(format!("spectral-frame-arena-{slot}")),
                    })
                    .expect("create frame arena"),
            )
        });

        let cmf_bins_cpu = spectral::cmf_bins_linear_srgb();
        let cmf_padded: Vec<Vec4> = cmf_bins_cpu.iter().map(|c| c.extend(0.0)).collect();
        let cmf_bins = device.upload_slice(&cmf_padded)?;

        Ok(Self {
            pipelines,
            frame_arenas,
            film: Film::new(FILM_STRIDE),
            schedule,
            render_scale,
            cmf_bins,
            cmf_bins_cpu,
            display_target_is_srgb: display::format_is_srgb(color_format),
        })
    }

    fn film_extent(&self, display: UVec2) -> UVec2 {
        UVec2::new(
            display.x.div_ceil(self.render_scale).max(1),
            display.y.div_ceil(self.render_scale).max(1),
        )
    }

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
        self.frame_arenas[ctx.slot].reset();

        let film_extent = self.film_extent(ctx.extent);
        let camera = CameraGpu::from_scene(scene, film_extent);
        let signature = FilmSignature::new(
            camera,
            geometry.revision(),
            gpu_scene.revision(),
            self.schedule.pixel_stride(),
            FILM_STRIDE,
        );
        let resized = self.film.extent() != film_extent;
        let needs_clear = self
            .film
            .prepare(ctx.device, film_extent, signature)
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
        // trace writes and display reads. The barrier is recorded outside any
        // encoder, so it lands queue-scoped at the head of the next compute
        // encoder — which is why it must only be emitted when compute work
        // follows. Without it, the film clear recorded on every camera move
        // races the prior frame's display pass still reading the buffer.
        cmd.barrier(
            StageFlags::COMPUTE | StageFlags::PIXEL_SHADER,
            StageFlags::COMPUTE,
        );

        if needs_clear {
            self.record_film_clear(ctx, cmd);
            if resized {
                eprintln!(
                    "spectral path tracer reset: {}x{} film ({}x{} display), target spp={}, passes/frame={}, materials={}, lights={}",
                    film_extent.x,
                    film_extent.y,
                    ctx.extent.x,
                    ctx.extent.y,
                    self.schedule.target_spp(),
                    self.schedule.passes_per_frame(),
                    gpu_scene.material_count,
                    gpu_scene.light_count
                );
            }
        }
        let Some(batch) = batch else {
            return;
        };
        let accum = self.film.accum().expect("film prepared");
        let root = self.frame_arenas[ctx.slot]
            .alloc(std::mem::size_of::<TraceRoot>() as u64, 16)
            .expect("trace root from frame arena");
        root.upload(&TraceRoot {
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
            film_width: film_extent.x,
            film_height: film_extent.y,
            pass_start: batch.start,
            pass_count: batch.count,
            target_passes: batch.target,
            phase_count: self.schedule.phase_count(),
            pixel_stride: self.schedule.pixel_stride(),
            light_count: gpu_scene.light_count,
            spectrum_len: gpu_scene.spectrum_len,
            spectral_bins: FILM_STRIDE,
            light_lane_count: N_LIGHT_LANES,
            uniform_lane_count: N_UNIFORM_LANES,
        })
        .expect("upload trace root");

        cmd.set_compute_pipeline(&self.pipelines.trace);
        cmd.dispatch(
            root.gpu,
            film_extent
                .x
                .div_ceil(self.schedule.pixel_stride())
                .div_ceil(integrator::THREADS_X),
            film_extent
                .y
                .div_ceil(self.schedule.pixel_stride())
                .div_ceil(integrator::THREADS_Y),
            1,
        );
        self.film.add_passes(batch.count);
        self.log_progress();
        cmd.barrier(StageFlags::COMPUTE, StageFlags::PIXEL_SHADER);
    }

    pub fn render(&mut self, ctx: &FrameCtx, cmd: &mut CommandBuffer) {
        let Some(accum) = self.film.accum() else {
            return;
        };
        if self.film.extent() != self.film_extent(ctx.extent) {
            return;
        }

        let root = self.frame_arenas[ctx.slot]
            .alloc(std::mem::size_of::<DisplayRoot>() as u64, 16)
            .expect("display root from frame arena");
        let film = self.film.extent();
        let progress = self.schedule.progress(self.film.pass_count());
        root.upload(&DisplayRoot {
            film: accum.gpu(),
            cmf: self.cmf_bins.gpu(),
            display_width: ctx.extent.x,
            display_height: ctx.extent.y,
            film_width: film.x,
            film_height: film.y,
            film_stride: FILM_STRIDE,
            spectral_bins: FILM_STRIDE,
            pixel_stride: self.schedule.pixel_stride(),
            completed_samples: progress.completed_samples,
            remaining_phases: progress.remaining_phases,
            target_is_srgb: u32::from(self.display_target_is_srgb),
            _pad: UVec2::ZERO,
        })
        .expect("upload display root");

        cmd.set_graphics_pipeline(&self.pipelines.display);
        cmd.set_depth_stencil_state(&DepthStencilState {
            depth_mode: DepthFlags::empty(),
            depth_test: CompareOp::Always,
            stencil_read_mask: 0,
            stencil_write_mask: 0,
            ..Default::default()
        });
        cmd.draw(None, root.gpu, 3, 1, 0, 0);
    }

    pub fn is_complete(&self) -> bool {
        self.schedule.is_complete(self.film.pass_count())
    }

    pub fn sample_count(&self) -> u32 {
        self.schedule.sample_count(self.film.pass_count())
    }

    pub fn pass_count(&self) -> u32 {
        self.film.pass_count()
    }

    pub fn target_spp(&self) -> u32 {
        self.schedule.target_spp()
    }

    pub fn passes_per_frame(&self) -> u32 {
        self.schedule.passes_per_frame()
    }

    pub fn extent(&self) -> UVec2 {
        self.film.extent()
    }

    pub fn tonemapped_rgba8(&self, device: &Device) -> anyhow::Result<Vec<u8>> {
        let rows = self.film.readback(device)?;
        Ok(display::film_to_rgba8(
            &rows,
            spectral::SPECTRAL_BINS,
            self.film.extent(),
            self.schedule,
            self.film.pass_count(),
            &self.cmf_bins_cpu,
        ))
    }

    /// Per-pixel band-integrated spectral radiance, row-major
    /// `[height][width][SPECTRAL_BINS]` — the raw spectral capture.
    pub fn spectral_bands(&self, device: &Device) -> anyhow::Result<Vec<f32>> {
        let rows = self.film.readback(device)?;
        Ok(display::film_to_bands(
            &rows,
            spectral::SPECTRAL_BINS,
            self.film.extent(),
            self.schedule,
            self.film.pass_count(),
        ))
    }

    /// Centre wavelength (nm) of each spectral bin, for labelling exports.
    pub fn spectral_bin_centers() -> Vec<f32> {
        (0..spectral::SPECTRAL_BINS)
            .map(spectral::spectral_bin_center)
            .collect()
    }

    pub fn destroy(self, device: &Device) {
        let Self {
            frame_arenas,
            film,
            cmf_bins,
            ..
        } = self;
        film.destroy(device);
        device.free(cmf_bins);
        for arena in frame_arenas {
            device.destroy_buffer(arena.into_buffer());
        }
    }

    fn record_film_clear(&mut self, ctx: &FrameCtx, cmd: &mut CommandBuffer) {
        let accum = self.film.accum().expect("film prepared");
        let film = self.film.extent();
        let float_count = film
            .x
            .checked_mul(film.y)
            .and_then(|pixels| pixels.checked_mul(FILM_STRIDE))
            .expect("spectral film element count exceeds u32");
        let root = self.frame_arenas[ctx.slot]
            .alloc(std::mem::size_of::<ClearRoot>() as u64, 16)
            .expect("clear root from frame arena");
        root.upload(&ClearRoot {
            film: accum.gpu(),
            count: float_count,
            _pad: 0,
        })
        .expect("upload clear root");

        cmd.set_compute_pipeline(&self.pipelines.clear);
        cmd.dispatch(root.gpu, float_count.div_ceil(CLEAR_THREADS), 1, 1);
        // The trace dispatch lands in a *different* compute encoder (every
        // set_compute_pipeline opens a fresh one), and a COMPUTE→COMPUTE
        // barrier inside this encoder is encoder-scoped — it would not order
        // the trace after the clear at all. Including PIXEL_SHADER in the
        // destination forces the queue-scoped form, ordering everything that
        // follows (trace and display) after the clear.
        cmd.barrier(
            StageFlags::COMPUTE,
            StageFlags::COMPUTE | StageFlags::PIXEL_SHADER,
        );
    }

    fn log_progress(&self) {
        let sample_count = self.sample_count();
        if sample_count == self.schedule.target_spp()
            || (sample_count >= 64 && sample_count.is_power_of_two())
        {
            eprintln!(
                "spectral path tracer progress: {}/{} spp",
                sample_count,
                self.schedule.target_spp()
            );
        }
    }
}
