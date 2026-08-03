//! Windowed application mode.

use std::sync::OnceLock;

use kiln_app::{Example, FrameCtx};
use kiln_rhi::{CommandBuffer, Device, Format};
use winit::event::WindowEvent;

use spectra::path_tracer::{PathTracer, Settings};
use spectra::preview::PreviewRenderer;
use spectra::render::{PresentRenderer, RenderFrame};
use spectra::scene::Scene;

use super::Result;
use super::config::Config;
use super::controls::CameraController;

static CONFIG: OnceLock<Config> = OnceLock::new();

pub fn run(config: Config) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let title = format!("Kiln · Spectral — {}", config.scene_name());
    let harness = config.harness.clone();
    CONFIG
        .set(config)
        .expect("viewer configured more than once");
    kiln_app::run_with::<App>(&title, [0.02, 0.02, 0.03, 1.0], harness)
}

struct App {
    scene: Scene,
    renderer: Box<dyn PresentRenderer>,
    controls: CameraController,
}

impl App {
    fn try_new(device: &Device, color_format: Format) -> Result<Self> {
        let config = CONFIG.get().expect("viewer config not installed");
        let asset = config.scene_path()?;
        let scene = spectra::usd::load(&asset)?;
        let light_spectrum = config.light_spectrum()?;
        let settings = Settings {
            target_spp: config.spp,
            passes_per_frame: config.passes_per_frame,
            render_scale: config.render_scale,
            pixel_stride: config.pixel_stride,
        };
        let renderer: Box<dyn PresentRenderer> =
            match PathTracer::new(device, color_format, &scene, &light_spectrum, settings) {
                Ok(renderer) => Box::new(renderer),
                Err(error) => {
                    eprintln!("spectral path tracer unavailable; using raster preview: {error:#}");
                    Box::new(PreviewRenderer::new(device, color_format, &scene)?)
                }
            };

        let controls = CameraController::new(scene.camera.world, scene.up);
        Ok(Self {
            scene,
            renderer,
            controls,
        })
    }
}

impl Example for App {
    fn depth_format(&self) -> Option<Format> {
        self.renderer.depth_format()
    }

    fn new(device: &Device, color_format: Format) -> Self {
        Self::try_new(device, color_format).unwrap_or_else(|error| {
            eprintln!("{error:#}");
            std::process::exit(1);
        })
    }

    fn window_event(&mut self, event: &WindowEvent) {
        self.controls.window_event(event);
    }

    fn pre_render(&mut self, ctx: &FrameCtx, cmd: &mut CommandBuffer) {
        if let Some(world) = self.controls.update() {
            self.scene.camera.world = world;
        }
        let frame = RenderFrame {
            device: ctx.device,
            extent: ctx.extent,
            slot: ctx.slot,
        };
        if let Err(error) = self.renderer.encode(&frame, cmd, &self.scene.camera) {
            eprintln!("renderer pre-render failed: {error:#}");
            std::process::exit(1);
        }
    }

    fn render(&mut self, ctx: &FrameCtx, cmd: &mut CommandBuffer) {
        let frame = RenderFrame {
            device: ctx.device,
            extent: ctx.extent,
            slot: ctx.slot,
        };
        self.renderer.encode_present(&frame, cmd);
    }

    fn destroy(self, device: &Device) {
        self.renderer.destroy(device);
    }
}
