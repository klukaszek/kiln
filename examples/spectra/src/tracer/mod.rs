//! Progressive spectral path tracer.

mod arena;
mod device;
mod film;
mod frame;
mod program;
mod sampling;
pub mod spectrum;
mod upload;

use glam::{UVec2, Vec3, Vec4};
use kiln_rhi::{Device, Format};

use self::spectrum::Spd;
use crate::render::{self as render, Error};
use crate::scene::Scene;

use arena::FrameArenas;
use device::GpuScene;
use film::Film;
use program::Pipelines;
use sampling::SpatialSchedule;
use upload::{GpuArray, GpuUploadBatch};

pub const DEFAULT_TARGET_SPP: u32 = 1024;
pub const DEFAULT_PASSES_PER_FRAME: u32 = 1;

const FILM_STRIDE: u32 = spectrum::SPECTRAL_BINS as u32;
const FRAME_ARENA_SIZE: u64 = 4096;

#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub target_spp: u32,
    pub passes_per_frame: u32,
    pub render_scale: u32,
    pub pixel_stride: u32,
    pub spectral_capture: bool,
}

/// Which groups of device buffers a scene edit invalidated.
///
/// Flags rather than dirty index lists: the tracer rebuilds each group whole, because every edit
/// restarts the progressive film anyway.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SceneUpdate {
    pub transforms: bool,
    pub emission: bool,
    pub surfaces: bool,
    pub topology: bool,
}

impl SceneUpdate {
    /// Geometry or instance count changed; the BLAS geometry is no longer valid.
    pub fn topology() -> Self {
        Self {
            topology: true,
            ..Self::default()
        }
    }

    pub fn transforms() -> Self {
        Self {
            transforms: true,
            ..Self::default()
        }
    }

    /// A light, or a material's emission. Both feed the same light list.
    pub fn emission() -> Self {
        Self {
            emission: true,
            ..Self::default()
        }
    }

    pub fn surfaces() -> Self {
        Self {
            surfaces: true,
            ..Self::default()
        }
    }

    /// A material changed in both its surface and its emission, or was added or removed.
    pub fn material() -> Self {
        Self {
            surfaces: true,
            emission: true,
            ..Self::default()
        }
    }

    pub fn merge(&mut self, other: Self) {
        self.transforms |= other.transforms;
        self.emission |= other.emission;
        self.surfaces |= other.surfaces;
        self.topology |= other.topology;
    }

    /// Nothing is dirty, so `update_scene` would be a no-op that still resets the film.
    pub fn is_empty(self) -> bool {
        self == Self::default()
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            target_spp: DEFAULT_TARGET_SPP,
            passes_per_frame: DEFAULT_PASSES_PER_FRAME,
            render_scale: 1,
            pixel_stride: 1,
            spectral_capture: false,
        }
    }
}

/// Progressive spectral path tracer and all resources required to render its scene.
pub struct PathTracer {
    scene: GpuScene,
    pipelines: Pipelines,
    frame_arenas: FrameArenas,
    /// Both are constant for the life of the device, so they are built once rather than per scene.
    sobol: GpuArray<u32>,
    cmf: GpuArray<Vec4>,
    film: Film,
    schedule: SpatialSchedule,
    render_scale: u32,
    cmf_bins_cpu: Vec<Vec3>,
    display_target_is_srgb: bool,
    spectral_capture: bool,
}

impl PathTracer {
    pub fn new(
        device: &Device,
        color_format: Format,
        scene: &Scene,
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
        let scene = GpuScene::build(device, scene, light_spectrum)?;
        let (pipelines, frame_arenas, sobol, cmf) =
            match Self::device_resources(device, color_format) {
                Ok(resources) => resources,
                Err(error) => {
                    scene.destroy(device);
                    return Err(error);
                }
            };

        Ok(Self {
            scene,
            pipelines,
            frame_arenas,
            sobol,
            cmf,
            film: Film::new(FILM_STRIDE),
            schedule,
            render_scale: settings.render_scale,
            cmf_bins_cpu: spectrum::cmf_bins_linear_srgb(),
            display_target_is_srgb: film::format_is_srgb(color_format),
            spectral_capture: settings.spectral_capture,
        })
    }

    /// Scene-independent device resources. Ordered so only the last step can strand an earlier
    /// one: pipeline objects free themselves on drop, so the Sobol table is the only cleanup.
    fn device_resources(
        device: &Device,
        color_format: Format,
    ) -> render::Result<(Pipelines, FrameArenas, GpuArray<u32>, GpuArray<Vec4>)> {
        let pipelines = Pipelines::new(device, color_format)?;

        let mut uploads = GpuUploadBatch::new(device)?;
        let sobol = uploads.upload(&sampling::table())?;
        let cmf = uploads.upload(&cmf_texels())?;
        uploads.submit()?;

        match FrameArenas::new(device, FRAME_ARENA_SIZE, "spectral-frame-arena") {
            Ok(frame_arenas) => Ok((pipelines, frame_arenas, sobol, cmf)),
            Err(error) => {
                sobol.destroy(device);
                cmf.destroy(device);
                Err(error)
            }
        }
    }

    fn film_extent(&self, display: UVec2) -> UVec2 {
        UVec2::new(
            display.x.div_ceil(self.render_scale).max(1),
            display.y.div_ceil(self.render_scale).max(1),
        )
    }

    pub fn is_complete(&self) -> bool {
        if self.scene.lights.len() == 0 {
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

    pub fn update_scene(
        &mut self,
        device: &Device,
        scene: &Scene,
        light_spectrum: &Spd,
        update: SceneUpdate,
    ) -> render::Result<()> {
        self.scene.update(device, scene, light_spectrum, update)?;
        self.film.invalidate();
        Ok(())
    }

    pub fn update_settings(&mut self, settings: Settings) -> render::Result<()> {
        if settings.render_scale == 0 {
            return Err(Error::Settings("render scale must be non-zero"));
        }
        self.schedule = SpatialSchedule::new(
            settings.target_spp,
            settings.passes_per_frame,
            settings.pixel_stride,
        )?;
        self.render_scale = settings.render_scale;
        self.spectral_capture = settings.spectral_capture;
        self.film.invalidate();
        Ok(())
    }

    pub fn tonemapped_rgba8(&self, device: &Device) -> render::Result<Vec<u8>> {
        let rows = self.film.readback(device)?;
        Ok(film::to_rgba8(
            &rows,
            spectrum::SPECTRAL_BINS,
            self.film.extent,
            self.schedule,
            self.film.pass_count,
            &self.cmf_bins_cpu,
            self.spectral_capture,
        ))
    }

    /// Per-pixel band-integrated spectral radiance, row-major
    /// `[height][width][SPECTRAL_BINS]`.
    pub fn spectral_bands(&self, device: &Device) -> render::Result<Vec<f32>> {
        if !self.spectral_capture {
            return Err(Error::Settings(
                "spectral bands require spectral capture mode",
            ));
        }
        let rows = self.film.readback(device)?;
        Ok(film::to_bands(
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
            sobol,
            cmf,
            film,
            ..
        } = self;
        film.destroy(device);
        frame_arenas.destroy(device);
        sobol.destroy(device);
        cmf.destroy(device);
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

/// The display shader's sensor table, padded to `float4` for the GPU.
fn cmf_texels() -> Vec<Vec4> {
    spectrum::cmf_bins_linear_srgb()
        .into_iter()
        .map(|c| c.extend(0.0))
        .collect()
}
