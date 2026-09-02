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
use crate::base::scene::{Camera, Scene, SceneStorage};

use film::Film;
use pipelines::Pipelines;
use scene::Storage;
use schedule::SpatialSchedule;

pub const DEFAULT_TARGET_SPP: u32 = 1024;
pub const DEFAULT_PASSES_PER_FRAME: u32 = 1;

const FILM_STRIDE: u32 = spectrum::SPECTRAL_BINS as u32;
const WAVELENGTH_LANES: u32 = 4;
const FRAME_ARENA_SIZE: u64 = 4096;

#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub target_spp: u32,
    pub passes_per_frame: u32,
    pub render_scale: u32,
    pub pixel_stride: u32,
    pub spectral_capture: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SceneUpdate {
    instance_transforms: Vec<usize>,
    lights: Vec<usize>,
    material_surfaces: Vec<usize>,
    material_emissions: Vec<usize>,
    topology_changed: bool,
}

impl SceneUpdate {
    /// Mark an edit that changes the geometry/instance topology and therefore requires the
    /// backend's full scene-resource fallback.
    pub fn topology() -> Self {
        Self {
            topology_changed: true,
            ..Self::default()
        }
    }

    pub fn instance_transform(index: usize) -> Self {
        Self {
            instance_transforms: vec![index],
            ..Self::default()
        }
    }

    pub fn light(index: usize) -> Self {
        Self {
            lights: vec![index],
            ..Self::default()
        }
    }

    pub fn material(index: usize) -> Self {
        Self {
            material_surfaces: vec![index],
            material_emissions: vec![index],
            ..Self::default()
        }
    }

    pub fn material_surface(index: usize) -> Self {
        Self {
            material_surfaces: vec![index],
            ..Self::default()
        }
    }

    pub fn material_emission(index: usize) -> Self {
        Self {
            material_emissions: vec![index],
            ..Self::default()
        }
    }

    pub fn merge(&mut self, other: Self) {
        for index in other.instance_transforms {
            if !self.instance_transforms.contains(&index) {
                self.instance_transforms.push(index);
            }
        }
        for index in other.lights {
            if !self.lights.contains(&index) {
                self.lights.push(index);
            }
        }
        for index in other.material_surfaces {
            if !self.material_surfaces.contains(&index) {
                self.material_surfaces.push(index);
            }
        }
        for index in other.material_emissions {
            if !self.material_emissions.contains(&index) {
                self.material_emissions.push(index);
            }
        }
        self.topology_changed |= other.topology_changed;
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
    storage: Storage,
    pipelines: Pipelines,
    frame_arenas: FrameArenas,
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
        let storage = scene.prepare::<Storage>(device, light_spectrum)?;
        let pipelines = match Pipelines::new(device, color_format, storage.spectrum.len()) {
            Ok(pipelines) => pipelines,
            Err(error) => {
                storage.destroy(device);
                return Err(error);
            }
        };
        let frame_arenas = match FrameArenas::new(device, FRAME_ARENA_SIZE, "spectral-frame-arena")
        {
            Ok(frame_arenas) => frame_arenas,
            Err(error) => {
                storage.destroy(device);
                return Err(error);
            }
        };

        Ok(Self {
            storage,
            pipelines,
            frame_arenas,
            film: Film::new(FILM_STRIDE),
            schedule,
            render_scale: settings.render_scale,
            cmf_bins_cpu: spectrum::cmf_bins_linear_srgb(),
            display_target_is_srgb: display::format_is_srgb(color_format),
            spectral_capture: settings.spectral_capture,
        })
    }

    fn film_extent(&self, display: UVec2) -> UVec2 {
        UVec2::new(
            display.x.div_ceil(self.render_scale).max(1),
            display.y.div_ceil(self.render_scale).max(1),
        )
    }

    pub fn is_complete(&self) -> bool {
        if self.storage.lights.len() == 0 {
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
        update: &SceneUpdate,
        frame_slot: usize,
    ) -> render::Result<()> {
        if !update.material_emissions.is_empty() {
            self.storage
                .update_material_emission(device, scene, light_spectrum)?;
        }
        self.storage.begin_frame_update(device, frame_slot);
        if !update.material_surfaces.is_empty() {
            self.storage.update_material_surfaces(
                device,
                scene,
                &update.material_surfaces,
                frame_slot,
            )?;
        }
        if update.topology_changed {
            self.storage
                .update_geometry(device, scene, light_spectrum)?;
        } else {
            if !update.instance_transforms.is_empty() {
                self.storage.update_instances(
                    device,
                    scene,
                    light_spectrum,
                    &update.instance_transforms,
                    frame_slot,
                )?;
            }
            if update.material_emissions.is_empty() && !update.lights.is_empty() {
                self.storage.update_lights(
                    device,
                    scene,
                    light_spectrum,
                    &update.lights,
                    frame_slot,
                )?;
            }
        }
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
        Ok(display::film_to_rgba8(
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
            storage,
            frame_arenas,
            film,
            ..
        } = self;
        film.destroy(device);
        frame_arenas.destroy(device);
        storage.destroy(device);
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

    fn samples_drawn(&self) -> Option<u32> {
        Some(self.sample_count())
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

#[cfg(test)]
mod scene_update_tests {
    use super::SceneUpdate;

    #[test]
    fn merges_dirty_ranges_without_duplicates() {
        let mut update = SceneUpdate::instance_transform(4);
        update.merge(SceneUpdate::instance_transform(4));
        update.merge(SceneUpdate::light(2));
        update.merge(SceneUpdate::material(7));
        update.merge(SceneUpdate::topology());

        assert_eq!(update.instance_transforms, vec![4]);
        assert_eq!(update.lights, vec![2]);
        assert_eq!(update.material_surfaces, vec![7]);
        assert_eq!(update.material_emissions, vec![7]);
        assert!(update.topology_changed);
    }

    #[test]
    fn surface_and_emission_updates_stay_independent() {
        let surface = SceneUpdate::material_surface(2);
        assert_eq!(surface.material_surfaces, vec![2]);
        assert!(surface.material_emissions.is_empty());

        let emission = SceneUpdate::material_emission(3);
        assert!(emission.material_surfaces.is_empty());
        assert_eq!(emission.material_emissions, vec![3]);
    }
}
