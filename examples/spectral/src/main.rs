//! Spectral path tracer: a USD stage progressively traced on the GPU through
//! the RHI.
//!
//! Loads the `--scene` (a named `.usda` under `examples/spectral/assets`, or any USD
//! file path) with the pure-Rust `openusd` crate and renders it with a compute
//! path tracer: Owen-scrambled Sobol' sampling, NEE + MIS, one
//! importance-sampled wavelength per path from a physically based light
//! spectrum, and moment-based reflectance spectra fitted to the USD albedos
//! (Peters 2019). Devices without ray-query support fall back to a mesh-shader
//! raster preview of the same triangle soup.
//!
//! This file is the application: CLI, the windowed harness glue, and the
//! headless render-to-PNG path. The domain code lives in [`scene`] (USD loading,
//! spectra, GPU upload), [`pathtracer`], and [`raster`].
//!
//! Run with: `cargo run -p spectral -- --spp 1024 --light-spectrum A`
//! (needs `slangc` on PATH).
//!
//! Windowed controls: WASD + Q/E to fly (Shift speeds up), left-drag to look,
//! R to return to the authored USD camera. Moving restarts accumulation.

mod controls;
mod export;
mod pathtracer;
mod png;
mod raster;
mod scene;

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use clap::Parser;
use glam::UVec2;
use kiln_rhi::{CommandBuffer, Device, DeviceDesc, Format};
use winit::event::WindowEvent;

use kiln_app::{Example, FrameCtx};
use controls::{CameraController, debug_camera_roundtrip};
use pathtracer::PathTracer;
use raster::RasterPreview;
use scene::gpu::GpuScene;
use scene::{Scene, spectral};

/// Bundled scenes (and the LSPDD drop folder) live here (next to this crate's `Cargo.toml`).
const ASSETS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/assets");

/// Progressive spectral path tracer for USD stages.
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
    /// (blackbody), <λ>nm (monochromatic), a path to an LSPDD CSV, or the
    /// stem of a CSV dropped into examples/spectral/assets/lspdd
    #[arg(long, default_value = "A")]
    light_spectrum: String,
    /// Scene to render: the stem of a .usda under examples/spectral/assets
    /// (e.g. cornell-box, cornell-box-copper) or a path to any USD file
    #[arg(long, default_value = "cornell-box")]
    scene: String,
    /// Windowed mode: trace at display resolution divided by this (the blit
    /// upscales). 2 quarters the per-frame path count on retina displays.
    #[arg(long, default_value_t = 1)]
    render_scale: u32,
    /// Headless: write the full per-pixel spectral film to this path as a
    /// float32 `.npy` of shape (height, width, SPECTRAL_BINS) — band-integrated
    /// radiance per pixel.
    #[arg(long, value_name = "PATH")]
    spectral_dump: Option<PathBuf>,
    /// Headless: print one pixel's spectrum (X,Y in film pixels) to stderr.
    #[arg(long, value_name = "X,Y", value_parser = parse_pixel)]
    spectral_probe: Option<(u32, u32)>,
    /// Harness options (e.g. --validation), parsed here and forwarded to `run_with`.
    #[command(flatten)]
    harness: kiln_app::HarnessOpts,
}

fn parse_pixel(value: &str) -> Result<(u32, u32), String> {
    let (x, y) = value
        .split_once(',')
        .ok_or_else(|| format!("expected X,Y, got {value:?}"))?;
    Ok((
        x.trim().parse().map_err(|_| format!("bad X in {value:?}"))?,
        y.trim().parse().map_err(|_| format!("bad Y in {value:?}"))?,
    ))
}

impl Config {
    /// Resolve `--light-spectrum` to an SPD: built-in names first, then a
    /// literal CSV path, then a stem under `examples/spectral/assets/lspdd`.
    fn light_spectrum(&self) -> anyhow::Result<spectral::Spd> {
        if let Some(spd) = spectral::named(&self.light_spectrum) {
            return Ok(spd);
        }
        let literal = PathBuf::from(&self.light_spectrum);
        if literal.is_file() {
            return spectral::from_lspdd_csv(&literal);
        }
        let dropped = Path::new(ASSETS_DIR)
            .join("lspdd")
            .join(&self.light_spectrum)
            .with_extension("csv");
        if dropped.is_file() {
            return spectral::from_lspdd_csv(&dropped);
        }
        anyhow::bail!(
            "no spectrum named {:?}: not a built-in, not a CSV path, and {} does not exist",
            self.light_spectrum,
            dropped.display()
        )
    }

    /// Resolve `--scene` to a USD file: a literal path first, then the stem of
    /// a `.usda` bundled under `examples/spectral/assets`.
    fn scene_path(&self) -> anyhow::Result<PathBuf> {
        let literal = PathBuf::from(&self.scene);
        if literal.is_file() {
            return Ok(literal);
        }
        let bundled = Path::new(ASSETS_DIR).join(&self.scene).with_extension("usda");
        if bundled.is_file() {
            return Ok(bundled);
        }
        anyhow::bail!(
            "no scene named {:?}: not a file, and not one of the bundled scenes ({})",
            self.scene,
            bundled_scenes().join(", ")
        )
    }

    /// The scene's stem, for the window title and headless PNG name.
    fn scene_name(&self) -> String {
        self.scene_path()
            .ok()
            .and_then(|p| p.file_stem().map(|s| s.to_string_lossy().into_owned()))
            .unwrap_or_else(|| self.scene.clone())
    }
}

/// Stems of the `.usda` stages bundled under `examples/spectral/assets`.
fn bundled_scenes() -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(ASSETS_DIR) else {
        return Vec::new();
    };
    let mut scenes: Vec<String> = entries
        .filter_map(|e| {
            let path = e.ok()?.path();
            (path.extension()? == "usda")
                .then(|| path.file_stem())?
                .map(|s| s.to_string_lossy().into_owned())
        })
        .collect();
    scenes.sort();
    scenes
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

    let title = format!("Kiln · Spectral — {}", config.scene_name());
    let harness = config.harness.clone();
    let _ = CONFIG.set(config);
    kiln_app::run_with::<App>(&title, [0.02, 0.02, 0.03, 1.0], harness)
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
    controls: CameraController,
}

impl Example for App {
    fn depth_format() -> Option<Format> {
        Some(Format::D32Float)
    }

    fn new(device: &Device, color_format: Format) -> Self {
        let config = CONFIG.get().cloned().unwrap_or_else(|| Config::parse_from(["spectral"]));
        let fail = |message: String| -> ! {
            eprintln!("{message}");
            std::process::exit(1);
        };

        let asset = config
            .scene_path()
            .unwrap_or_else(|e| fail(format!("invalid --scene: {e}")));
        let scene = scene::load(&asset)
            .unwrap_or_else(|e| fail(format!("failed to load {}: {e}", asset.display())));
        let light_spectrum = config
            .light_spectrum()
            .unwrap_or_else(|e| fail(format!("invalid --light-spectrum: {e}")));
        let gpu_scene = GpuScene::build(device, &scene, &light_spectrum)
            .unwrap_or_else(|e| fail(format!("failed to upload scene: {e}")));
        let raster = RasterPreview::build(device, color_format, &scene);

        let tracer = if gpu_scene.accel.is_some() {
            match PathTracer::new(
                device,
                color_format,
                config.spp,
                config.samples_per_frame,
                config.render_scale,
            ) {
                Ok(tracer) => Some(tracer),
                Err(e) => {
                    eprintln!("spectral path tracer disabled: {e}");
                    None
                }
            }
        } else {
            None
        };

        let controls = CameraController::new(&scene.camera.world, scene.up);
        Self {
            scene,
            gpu_scene,
            raster,
            tracer,
            controls,
        }
    }

    fn window_event(&mut self, event: &WindowEvent) {
        self.controls.window_event(event);
    }

    fn pre_render(&mut self, ctx: &FrameCtx, cmd: &mut CommandBuffer) {
        self.controls.update(&mut self.scene.camera.world);
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
        label: Some("spectral-headless".into()),
        ..Default::default()
    })?;

    let scene = scene::load(&config.scene_path()?)?;
    if std::env::var_os("SPECTRAL_DEBUG_CAMERA").is_some() {
        debug_camera_roundtrip(&scene);
    }
    let light_spectrum = config.light_spectrum()?;
    let gpu_scene = GpuScene::build(&device, &scene, &light_spectrum)?;
    anyhow::ensure!(
        gpu_scene.accel.is_some(),
        "headless render needs ray tracing support"
    );
    // Headless renders exactly the requested resolution: no render scaling.
    let mut tracer = PathTracer::new(
        &device,
        Format::B8G8R8A8Srgb,
        config.spp,
        config.samples_per_frame,
        1,
    )?;
    eprintln!(
        "spectral headless: {}x{}, target spp={}, samples/frame={}, light spectrum {}",
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
    // Time only the trace loop (each iteration waits idle, so wall time is GPU
    // time): clean ms/spp, free of startup/slangc/BVH-build noise.
    let trace_start = std::time::Instant::now();
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
    let trace_ms = trace_start.elapsed().as_secs_f64() * 1e3;
    let spp = tracer.sample_count().max(1) as f64;
    eprintln!(
        "spectral trace: {:.1} ms for {} spp = {:.2} ms/spp ({:.2} Mpath/s)",
        trace_ms,
        tracer.sample_count(),
        trace_ms / spp,
        (resolution.x * resolution.y) as f64 * spp / (trace_ms * 1e3),
    );

    let rgba = tracer.tonemapped_rgba8()?;
    let extent = tracer.extent();
    let name = format!(
        "{}_{}x{}_{}spp",
        config.scene_name(),
        extent.x,
        extent.y,
        tracer.sample_count()
    );
    let path = png::save_rgba_png(&name, extent.x, extent.y, &rgba)?;
    eprintln!("spectral headless wrote {}", path.display());

    export::emit(
        &tracer,
        extent,
        config.spectral_probe,
        config.spectral_dump.as_deref(),
    )?;
    Ok(())
}
