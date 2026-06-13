//! Spectral path tracer: a USD stage progressively traced on the GPU through
//! the RHI.
//!
//! Loads the `--scene` (a named `.usda` under `examples/assets`, or any USD
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
//! Run with: `cargo run --example spectral -- --spp 1024 --light-spectrum A`
//! (needs `slangc` on PATH).
//!
//! Windowed controls: WASD + Q/E to fly (Shift speeds up), left-drag to look,
//! R to return to the authored USD camera. Moving restarts accumulation.

#[path = "../common/mod.rs"]
mod common;

mod pathtracer;
mod png;
mod raster;
mod scene;

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;

use clap::Parser;
use glam::{DMat4, DQuat, DVec2, DVec3, UVec2};
use kiln_rhi::{CommandBuffer, Device, DeviceDesc, Format};
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

use common::{Example, FrameCtx};
use pathtracer::PathTracer;
use raster::RasterPreview;
use scene::gpu::GpuScene;
use scene::{Scene, spectral};

/// Bundled scenes (and the LSPDD drop folder) live here.
const ASSETS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/assets");

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
    /// stem of a CSV dropped into examples/assets/lspdd
    #[arg(long, default_value = "A")]
    light_spectrum: String,
    /// Scene to render: the stem of a .usda under examples/assets
    /// (e.g. cornell-box, cornell-box-copper) or a path to any USD file
    #[arg(long, default_value = "cornell-box")]
    scene: String,
}

impl Config {
    /// Resolve `--light-spectrum` to an SPD: built-in names first, then a
    /// literal CSV path, then a stem under `examples/assets/lspdd`.
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
    /// a `.usda` bundled under `examples/assets`.
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

/// Stems of the `.usda` stages bundled under `examples/assets`.
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
    let _ = CONFIG.set(config);
    common::run::<App>(&title, [0.02, 0.02, 0.03, 1.0])
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
            match PathTracer::new(device, color_format, config.spp, config.samples_per_frame) {
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
// Interactive fly camera. WASD moves in the view plane, Q/E descends/climbs
// along the scene's up axis, Shift speeds up, dragging with the left mouse
// button looks around, and R returns to the authored USD camera. The path
// tracer keys its film off the camera basis, so any motion restarts
// progressive accumulation by itself.
//
// Yaw/pitch live in a "levelled" local frame whose Y is the *stage's* up axis
// (`Scene::up`) — a Z-up stage steered with Y-up controls yaws around the view
// axis (i.e. rolls) and starts at the gimbal pole, where decomposing the
// authored matrix turns numerical noise into a finite roll. The controller
// also never writes the camera until the first actual input, so the authored
// USD view survives loading bit-exact (and the film key stays stable).
// ---------------------------------------------------------------------------

/// Base fly speed in scene units/second (the Cornell box is ~5.5 units tall).
const FLY_SPEED: f64 = 2.5;
const FLY_SPEED_BOOST: f64 = 4.0;
/// Look sensitivity in radians per pixel of drag.
const LOOK_SPEED: f64 = 0.004;

/// Keys that take the camera over from the authored USD transform.
const MOVEMENT_KEYS: [KeyCode; 6] = [
    KeyCode::KeyW,
    KeyCode::KeyA,
    KeyCode::KeyS,
    KeyCode::KeyD,
    KeyCode::KeyQ,
    KeyCode::KeyE,
];

struct CameraController {
    /// The authored camera world transform, restored by R.
    home: DMat4,
    /// Rotation taking the controller's Y-up local frame to world space.
    frame: DQuat,
    position: DVec3,
    yaw: f64,
    pitch: f64,
    /// False until the first movement/look input: while false, `update` leaves
    /// the camera untouched (the authored transform, roll and all).
    active: bool,
    reset_requested: bool,
    held: HashSet<KeyCode>,
    dragging: bool,
    cursor: Option<DVec2>,
    last_tick: Instant,
}

impl CameraController {
    fn new(world: &DMat4, up: DVec3) -> Self {
        let frame = DQuat::from_rotation_arc(DVec3::Y, up.normalize_or(DVec3::Y));
        let (position, yaw, pitch) = Self::decompose(world, frame);
        Self {
            home: *world,
            frame,
            position,
            yaw,
            pitch,
            active: false,
            reset_requested: false,
            held: HashSet::new(),
            dragging: false,
            cursor: None,
            last_tick: Instant::now(),
        }
    }

    /// Position plus yaw/pitch of the camera's view axis in the levelled local
    /// frame. Any authored roll is dropped — the controller keeps the horizon
    /// level once it takes over.
    fn decompose(world: &DMat4, frame: DQuat) -> (DVec3, f64, f64) {
        let forward = frame.inverse() * (-world.z_axis.truncate()).normalize_or(DVec3::NEG_Z);
        (
            world.w_axis.truncate(),
            (-forward.x).atan2(-forward.z),
            forward.y.clamp(-1.0, 1.0).asin(),
        )
    }

    fn window_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                let PhysicalKey::Code(code) = event.physical_key else {
                    return;
                };
                match event.state {
                    ElementState::Pressed => {
                        if code == KeyCode::KeyR {
                            self.reset_requested = true;
                        }
                        self.held.insert(code);
                    }
                    ElementState::Released => {
                        self.held.remove(&code);
                    }
                }
            }
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state,
                ..
            } => {
                self.dragging = *state == ElementState::Pressed;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let position = DVec2::new(position.x, position.y);
                if self.dragging && let Some(last) = self.cursor {
                    let delta = position - last;
                    if delta != DVec2::ZERO {
                        self.active = true;
                    }
                    self.yaw -= delta.x * LOOK_SPEED;
                    self.pitch = (self.pitch - delta.y * LOOK_SPEED).clamp(
                        -std::f64::consts::FRAC_PI_2 + 0.01,
                        std::f64::consts::FRAC_PI_2 - 0.01,
                    );
                }
                self.cursor = Some(position);
            }
            WindowEvent::Focused(false) => {
                self.held.clear();
                self.dragging = false;
            }
            _ => {}
        }
    }

    /// Integrate held keys over the elapsed frame time and write the camera's
    /// world transform (once the user has taken over).
    fn update(&mut self, world: &mut DMat4) {
        let dt = self.last_tick.elapsed().as_secs_f64().min(0.1);
        self.last_tick = Instant::now();

        if self.reset_requested {
            self.reset_requested = false;
            self.active = false;
            *world = self.home;
            (self.position, self.yaw, self.pitch) = Self::decompose(world, self.frame);
            return;
        }
        if !self.active {
            // Key takeover is derived from held state here (not from the press
            // event) so movement keys still held across an R reset resume
            // flying immediately instead of waiting for an OS key repeat.
            self.active = MOVEMENT_KEYS.iter().any(|key| self.held.contains(key));
            if !self.active {
                return;
            }
        }

        let rotation = self.frame
            * DQuat::from_rotation_y(self.yaw)
            * DQuat::from_rotation_x(self.pitch);
        let mut wish = DVec3::ZERO;
        let held = |code| self.held.contains(&code) as i32 as f64;
        wish += (rotation * DVec3::NEG_Z) * (held(KeyCode::KeyW) - held(KeyCode::KeyS));
        wish += (rotation * DVec3::X) * (held(KeyCode::KeyD) - held(KeyCode::KeyA));
        wish += (self.frame * DVec3::Y) * (held(KeyCode::KeyE) - held(KeyCode::KeyQ));
        if wish != DVec3::ZERO {
            let boost = if self.held.contains(&KeyCode::ShiftLeft)
                || self.held.contains(&KeyCode::ShiftRight)
            {
                FLY_SPEED_BOOST
            } else {
                1.0
            };
            self.position += wish.normalize() * (FLY_SPEED * boost * dt);
        }

        *world = DMat4::from_rotation_translation(rotation, self.position);
    }
}

/// `SPECTRAL_DEBUG_CAMERA=1`: print the authored camera world matrix next to
/// the controller's takeover rebuild, to validate the decompose math per scene.
fn debug_camera_roundtrip(scene: &Scene) {
    let authored = scene.camera.world;
    let mut controls = CameraController::new(&authored, scene.up);
    controls.active = true;
    let mut rebuilt = authored;
    controls.update(&mut rebuilt);
    eprintln!("up axis:  {:?}", scene.up);
    eprintln!("authored: {authored:.6}");
    eprintln!("rebuilt:  {rebuilt:.6}");
    let drift = (rebuilt - authored).abs().to_cols_array().into_iter().fold(0.0, f64::max);
    eprintln!("max abs drift: {drift:.2e}");
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
    let mut tracer = PathTracer::new(
        &device,
        Format::B8G8R8A8Srgb,
        config.spp,
        config.samples_per_frame,
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
    let name = format!(
        "{}_{}x{}_{}spp",
        config.scene_name(),
        extent.x,
        extent.y,
        tracer.sample_count()
    );
    let path = png::save_rgba_png(&name, extent.x, extent.y, &rgba)?;
    eprintln!("spectral headless wrote {}", path.display());
    Ok(())
}
