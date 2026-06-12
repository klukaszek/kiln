//! Cornell box, progressively path traced on the GPU through the RHI — with
//! spectral transport.
//!
//! Loads `examples/assets/cornell-box.usda` with the pure-Rust `openusd` crate
//! and renders it with a compute path tracer: Owen-scrambled Sobol' sampling,
//! NEE + MIS, one importance-sampled wavelength per path from a physically based
//! light spectrum, and moment-based reflectance spectra fitted to the USD
//! albedos (Peters 2019). Devices without ray-query support fall back to a
//! mesh-shader raster preview of the same triangle soup.
//!
//! This file is the application: CLI, the windowed harness glue, and the
//! headless render-to-PNG path. The domain code lives in [`scene`] (USD loading,
//! spectra, GPU upload), [`pathtracer`], and [`raster`].
//!
//! Run with: `cargo run --example cornell_box -- --spp 1024 --light-spectrum A`
//! (needs `slangc` on PATH).

#[path = "../common/mod.rs"]
mod common;

mod pathtracer;
mod png;
mod raster;
mod scene;

use std::sync::OnceLock;

use clap::Parser;
use glam::UVec2;
use kiln_rhi::{CommandBuffer, Device, DeviceDesc, Format};

use common::{Example, FrameCtx};
use pathtracer::PathTracer;
use raster::RasterPreview;
use scene::gpu::GpuScene;
use scene::{Scene, spectral};

const ASSET: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/examples/assets/cornell-box.usda"
);

/// Progressive spectral path tracer for the Cornell box USD stage.
#[derive(Parser, Clone)]
struct Config {
    /// Target samples per pixel for the progressive render
    #[arg(long, default_value_t = pathtracer::DEFAULT_TARGET_SPP)]
    spp: u32,
    /// Path samples accumulated per frame
    #[arg(long, visible_alias = "spf", default_value_t = pathtracer::DEFAULT_SAMPLES_PER_FRAME)]
    samples_per_frame: u32,
    /// Render offscreen at WxH and write a PNG under target/test-images
    #[arg(long, value_name = "WxH", value_parser = parse_resolution)]
    headless: Option<UVec2>,
    /// Light emission spectrum: A, D50, D65, E, FL2, FL7, FL11, <T>K
    /// (blackbody), <λ>nm (monochromatic), or a path to an LSPDD CSV
    #[arg(long, default_value = "A")]
    light_spectrum: String,
}

impl Config {
    /// Resolve `--light-spectrum` to an SPD: built-in names first, then an
    /// LSPDD CSV path.
    fn light_spectrum(&self) -> anyhow::Result<spectral::Spd> {
        if let Some(spd) = spectral::named(&self.light_spectrum) {
            return Ok(spd);
        }
        spectral::from_lspdd_csv(std::path::Path::new(&self.light_spectrum))
    }
}

fn parse_resolution(value: &str) -> Result<UVec2, String> {
    let (w, h) = value
        .split_once(['x', 'X'])
        .ok_or_else(|| format!("expected a resolution in WxH form, got {value:?}"))?;
    let parse = |s: &str| {
        s.parse::<u32>()
            .ok()
            .filter(|&v| v > 0)
            .ok_or_else(|| format!("expected a positive integer, got {s:?}"))
    };
    Ok(UVec2::new(parse(w)?, parse(h)?))
}

/// The windowed harness constructs the example through the no-argument
/// [`Example::new`], so `main` stashes the parsed config here.
static CONFIG: OnceLock<Config> = OnceLock::new();

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::parse();
    if let Some(resolution) = config.headless {
        run_headless(&config, resolution)?;
        return Ok(());
    }

    let _ = CONFIG.set(config);
    common::run::<App>("Kiln · Cornell box", [0.02, 0.02, 0.03, 1.0])
}

// ---------------------------------------------------------------------------
// Windowed application: glue between the harness, the scene, and the two
// renderers. Prefers the path tracer; raster preview when the device can't
// trace.
// ---------------------------------------------------------------------------

struct App {
    scene: Scene,
    gpu_scene: GpuScene,
    raster: RasterPreview,
    tracer: Option<PathTracer>,
}

impl Example for App {
    fn depth_format() -> Option<Format> {
        Some(Format::D32Float)
    }

    fn new(device: &Device, color_format: Format) -> Self {
        let config = CONFIG.get().cloned().unwrap_or_else(|| Config::parse_from(["cornell_box"]));
        let fail = |message: String| -> ! {
            eprintln!("{message}");
            std::process::exit(1);
        };

        let scene = scene::load(ASSET)
            .unwrap_or_else(|e| fail(format!("failed to load {ASSET}: {e}")));
        let light_spectrum = config
            .light_spectrum()
            .unwrap_or_else(|e| fail(format!("invalid --light-spectrum: {e}")));
        let gpu_scene = GpuScene::build(device, &scene, &light_spectrum)
            .unwrap_or_else(|e| fail(format!("failed to upload scene: {e}")));
        let raster = RasterPreview::build(device, color_format, &scene);

        let tracer = if gpu_scene.accel.is_some() {
            match PathTracer::new(device, color_format, config.spp, config.samples_per_frame) {
                Ok(tracer) => Some(tracer),
                Err(e) => {
                    eprintln!("cornell path tracer disabled: {e}");
                    None
                }
            }
        } else {
            None
        };

        Self {
            scene,
            gpu_scene,
            raster,
            tracer,
        }
    }

    fn pre_render(&mut self, ctx: &FrameCtx, cmd: &mut CommandBuffer) {
        if let Some(tracer) = &mut self.tracer {
            tracer.pre_render(ctx, cmd, &self.scene, &self.gpu_scene);
        }
    }

    fn render(&mut self, ctx: &FrameCtx, cmd: &mut CommandBuffer) {
        if let Some(tracer) = &mut self.tracer {
            tracer.render(ctx, cmd);
            return;
        }
        self.raster.render(ctx, cmd, &self.scene, &self.gpu_scene);
    }
}

// ---------------------------------------------------------------------------
// Headless: trace to the target sample count and write a PNG.
// ---------------------------------------------------------------------------

fn run_headless(config: &Config, resolution: UVec2) -> anyhow::Result<()> {
    let device = Device::new(&DeviceDesc {
        validation: false,
        label: Some("cornell-box-headless".into()),
        ..Default::default()
    })?;

    let scene = scene::load(ASSET)?;
    let light_spectrum = config.light_spectrum()?;
    let gpu_scene = GpuScene::build(&device, &scene, &light_spectrum)?;
    anyhow::ensure!(
        gpu_scene.accel.is_some(),
        "headless render needs ray tracing support"
    );
    let mut tracer = PathTracer::new(
        &device,
        Format::B8G8R8A8Srgb,
        config.spp,
        config.samples_per_frame,
    )?;
    eprintln!(
        "cornell headless: {}x{}, target spp={}, samples/frame={}, light spectrum {}",
        resolution.x,
        resolution.y,
        tracer.target_spp(),
        tracer.samples_per_frame(),
        light_spectrum.name,
    );

    // Each iteration submits and drains, so reusing frame slot 0 is safe here.
    let ctx = FrameCtx {
        device: &device,
        extent: resolution,
        slot: 0,
    };
    while !tracer.is_complete() {
        let before = tracer.sample_count();
        let mut cmd = device.create_command_buffer()?;
        tracer.pre_render(&ctx, &mut cmd, &scene, &gpu_scene);
        cmd.end();
        let queue = device.queue();
        queue.submit(cmd)?;
        queue.wait_idle();

        anyhow::ensure!(
            tracer.sample_count() > before,
            "path tracer made no progress; lights={}",
            gpu_scene.light_count
        );
    }

    let rgba = tracer.tonemapped_rgba8()?;
    let extent = tracer.extent();
    let name = format!("cornell_box_{}x{}_{}spp", extent.x, extent.y, tracer.sample_count());
    let path = png::save_rgba_png(&name, extent.x, extent.y, &rgba)?;
    eprintln!("cornell headless wrote {}", path.display());
    Ok(())
}
