//! Progressive spectral GPU path tracer.
//!
//! [`PathTracer`] owns scene-independent pipelines, scheduling, and the
//! progressive film. Geometry and spectral scene data remain externally owned.

mod display;
mod film;
mod integrator;
mod pipelines;
mod record;
mod roots;
mod sampler;
mod schedule;

use glam::{UVec2, Vec3, Vec4};
use kiln_rhi::{Device, Format, GpuAllocation};

use crate::frame_arena::FrameArenas;
use crate::scene::spectral;
use film::Film;
use pipelines::Pipelines;
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
    frame_arenas: FrameArenas,
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

        let cmf_bins_cpu = spectral::cmf_bins_linear_srgb();
        let cmf_padded: Vec<Vec4> = cmf_bins_cpu.iter().map(|c| c.extend(0.0)).collect();
        let cmf_bins = device.upload_slice(&cmf_padded)?;
        let frame_arenas = match FrameArenas::new(device, FRAME_ARENA_SIZE, "spectral-frame-arena")
        {
            Ok(arenas) => arenas,
            Err(error) => {
                device.free(cmf_bins);
                return Err(error);
            }
        };

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
        frame_arenas.destroy(device);
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
