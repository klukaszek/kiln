use std::sync::OnceLock;

use anyhow::Context;
use kiln_app::{Example, FrameCtx};
use kiln_rhi::{CommandBuffer, Device, Format};
use winit::event::WindowEvent;

use crate::config::Config;
use crate::controls::CameraController;
use crate::pathtracer::PathTracer;
use crate::raster::RasterPreview;
use crate::scene::Scene;
use crate::scene::gpu::{GpuGeometry, SpectralGpuScene};
use crate::scene::spectral::Spd;

static CONFIG: OnceLock<Config> = OnceLock::new();

pub fn run(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let title = format!("Kiln · Spectral — {}", config.scene_name());
    let harness = config.harness.clone();
    CONFIG
        .set(config)
        .expect("viewer configured more than once");
    kiln_app::run_with::<App>(&title, [0.02, 0.02, 0.03, 1.0], harness)
}

struct App {
    scene: Scene,
    geometry: GpuGeometry,
    renderer: Renderer,
    controls: CameraController,
}

enum Renderer {
    PathTraced {
        scene: SpectralGpuScene,
        tracer: Box<PathTracer>,
    },
    Raster(RasterPreview),
}

impl App {
    fn try_new(device: &Device, color_format: Format) -> anyhow::Result<Self> {
        let config = CONFIG.get().expect("viewer config not installed");
        let asset = config.scene_path().context("invalid --scene")?;
        let scene =
            crate::scene::load(&asset).with_context(|| format!("loading {}", asset.display()))?;
        let light_spectrum = config
            .light_spectrum()
            .context("invalid --light-spectrum")?;
        let geometry = GpuGeometry::build(device, &scene).context("uploading geometry")?;
        let renderer = Renderer::build(
            device,
            color_format,
            &scene,
            &geometry,
            config,
            &light_spectrum,
        )?;

        let controls = CameraController::new(&scene.camera.world, scene.up);
        Ok(Self {
            scene,
            geometry,
            renderer,
            controls,
        })
    }
}

impl Renderer {
    fn build(
        device: &Device,
        color_format: Format,
        scene: &Scene,
        geometry: &GpuGeometry,
        config: &Config,
        light_spectrum: &Spd,
    ) -> anyhow::Result<Self> {
        if geometry.accel.is_some() {
            let spectral_scene = SpectralGpuScene::build(device, scene, light_spectrum)
                .context("uploading spectral scene")?;
            match PathTracer::new(
                device,
                color_format,
                config.spp,
                config.passes_per_frame,
                config.render_scale,
                config.pixel_stride,
            ) {
                Ok(tracer) => {
                    return Ok(Self::PathTraced {
                        scene: spectral_scene,
                        tracer: Box::new(tracer),
                    });
                }
                Err(error) => {
                    eprintln!("spectral path tracer disabled: {error}");
                    spectral_scene.destroy(device);
                }
            }
        }

        RasterPreview::build(device, color_format, geometry)
            .map(Self::Raster)
            .context("building raster preview")
    }

    fn destroy(self, device: &Device) {
        match self {
            Self::PathTraced { scene, tracer } => {
                (*tracer).destroy(device);
                scene.destroy(device);
            }
            Self::Raster(raster) => raster.destroy(device),
        }
    }
}

impl Example for App {
    fn depth_format() -> Option<Format> {
        Some(Format::D32Float)
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
        self.controls.update(&mut self.scene.camera.world);
        if let Renderer::PathTraced { scene, tracer } = &mut self.renderer {
            tracer.pre_render(ctx, cmd, &self.scene, &self.geometry, scene);
        }
    }

    fn render(&mut self, ctx: &FrameCtx, cmd: &mut CommandBuffer) {
        match &mut self.renderer {
            Renderer::PathTraced { tracer, .. } => tracer.render(ctx, cmd),
            Renderer::Raster(raster) => raster.render(ctx, cmd, &self.scene, &self.geometry),
        }
    }

    fn destroy(self, device: &Device) {
        self.renderer.destroy(device);
        self.geometry.destroy(device);
    }
}
