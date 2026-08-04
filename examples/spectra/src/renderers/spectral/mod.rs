//! Progressive spectral path tracer.

mod acceleration;
mod bsdf;
mod display;
mod film;
mod integrator;
mod passes;
mod pipelines;
mod sampler;
mod scene;
mod schedule;
mod shader_types;
pub mod spectrum;

use glam::{UVec2, Vec3};
use kiln_rhi::{CommandBuffer, Device, Format};

use self::spectrum::Spd;
use crate::base::gpu::FrameArenas;
use crate::base::renderer::{self as render, Error, PresentRenderer, RenderFrame, Renderer};
use crate::base::scene::{Camera, CpuStorage, Scene};

use film::Film;
use pipelines::Pipelines;
use scene::Storage;
use schedule::SpatialSchedule;

pub const DEFAULT_TARGET_SPP: u32 = 1024;
pub const DEFAULT_PASSES_PER_FRAME: u32 = 1;

const FILM_STRIDE: u32 = spectrum::SPECTRAL_BINS as u32;
const N_LIGHT_LANES: u32 = 2;
const N_UNIFORM_LANES: u32 = 2;
const FRAME_ARENA_SIZE: u64 = 4096;

#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub target_spp: u32,
    pub passes_per_frame: u32,
    pub render_scale: u32,
    pub pixel_stride: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            target_spp: DEFAULT_TARGET_SPP,
            passes_per_frame: DEFAULT_PASSES_PER_FRAME,
            render_scale: 1,
            pixel_stride: 1,
        }
    }
}

/// Progressive spectral path tracer and all resources required to render its scene.
pub struct PathTracer {
    scene: Scene<Storage>,
    pipelines: Pipelines,
    frame_arenas: FrameArenas,
    film: Film,
    schedule: SpatialSchedule,
    render_scale: u32,
    cmf_bins_cpu: Vec<Vec3>,
    display_target_is_srgb: bool,
}

impl PathTracer {
    pub fn new(
        device: &Device,
        color_format: Format,
        scene: &Scene<CpuStorage>,
        light_spectrum: &Spd,
        settings: Settings,
    ) -> render::Result<Self> {
        if settings.render_scale == 0 {
            return Err(Error::Settings("render scale must be non-zero"));
        }
        let schedule = SpatialSchedule::new(
            settings.target_spp,
            settings.passes_per_frame,
            settings.pixel_stride,
        )?;
        let trace_scene = scene.prepare::<Storage>(device, light_spectrum)?;
        let pipelines = match Pipelines::new(
            device,
            color_format,
            schedule.pixel_stride,
            trace_scene.storage.lights.len(),
            trace_scene.storage.spectrum.len(),
        ) {
            Ok(pipelines) => pipelines,
            Err(error) => {
                trace_scene.destroy(device);
                return Err(error);
            }
        };
        let frame_arenas = match FrameArenas::new(device, FRAME_ARENA_SIZE, "spectral-frame-arena")
        {
            Ok(frame_arenas) => frame_arenas,
            Err(error) => {
                trace_scene.destroy(device);
                return Err(error);
            }
        };

        Ok(Self {
            scene: trace_scene,
            pipelines,
            frame_arenas,
            film: Film::new(FILM_STRIDE),
            schedule,
            render_scale: settings.render_scale,
            cmf_bins_cpu: spectrum::cmf_bins_linear_srgb(),
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
        if self.scene.storage.lights.len() == 0 {
            self.film.extent != UVec2::ZERO
        } else {
            self.schedule.is_complete(self.film.pass_count)
        }
    }

    pub fn sample_count(&self) -> u32 {
        self.schedule.sample_count(self.film.pass_count)
    }

    pub fn pass_count(&self) -> u32 {
        self.film.pass_count
    }

    pub fn target_spp(&self) -> u32 {
        self.schedule.target_spp
    }

    pub fn passes_per_frame(&self) -> u32 {
        self.schedule.passes_per_frame
    }

    pub fn extent(&self) -> UVec2 {
        self.film.extent
    }

    pub fn tonemapped_rgba8(&self, device: &Device) -> render::Result<Vec<u8>> {
        let rows = self.film.readback(device)?;
        Ok(display::film_to_rgba8(
            &rows,
            spectrum::SPECTRAL_BINS,
            self.film.extent,
            self.schedule,
            self.film.pass_count,
            &self.cmf_bins_cpu,
        ))
    }

    /// Per-pixel band-integrated spectral radiance, row-major
    /// `[height][width][SPECTRAL_BINS]`.
    pub fn spectral_bands(&self, device: &Device) -> render::Result<Vec<f32>> {
        let rows = self.film.readback(device)?;
        Ok(display::film_to_bands(
            &rows,
            spectrum::SPECTRAL_BINS,
            self.film.extent,
            self.schedule,
            self.film.pass_count,
        ))
    }

    pub fn spectral_bin_centers() -> Vec<f32> {
        (0..spectrum::SPECTRAL_BINS)
            .map(spectrum::spectral_bin_center)
            .collect()
    }

    pub fn destroy(self, device: &Device) {
        let Self {
            scene,
            frame_arenas,
            film,
            ..
        } = self;
        film.destroy(device);
        frame_arenas.destroy(device);
        scene.destroy(device);
    }

    fn log_progress(&self) {
        let sample_count = self.sample_count();
        if sample_count == self.schedule.target_spp
            || (sample_count >= 64 && sample_count.is_power_of_two())
        {
            eprintln!(
                "spectral path tracer progress: {}/{} spp",
                sample_count, self.schedule.target_spp
            );
        }
    }
}

impl Renderer for PathTracer {
    fn encode(
        &mut self,
        frame: &RenderFrame<'_>,
        commands: &mut CommandBuffer,
        camera: &Camera,
    ) -> render::Result<()> {
        self.record_iteration(frame, commands, camera)
    }

    fn destroy(self: Box<Self>, device: &Device) {
        (*self).destroy(device);
    }
}

impl PresentRenderer for PathTracer {
    fn encode_present(&mut self, frame: &RenderFrame<'_>, commands: &mut CommandBuffer) {
        self.record_display(frame, commands)
    }
}
