//! Windowed application: the harness lifecycle, the clock, and the scene rebuilt each frame.

use std::sync::OnceLock;
use std::time::Instant;

use kiln_app::{Example, FrameCtx, PerformanceStats};
use kiln_rhi::{CommandBuffer, Device, Format};

use crate::cascades::{self, HrcRenderer, Settings};
use crate::scene::{self, Prim};
use crate::ui::{Controls, Readout};

/// The CLI's opening state. `Example::new` takes only a device, so the options parsed in `main`
/// reach the app through here rather than through a constructor argument.
static STARTUP: OnceLock<(u32, usize)> = OnceLock::new();

pub fn configure(probes: u32, scene: usize) {
    STARTUP
        .set((probes, scene))
        .expect("app configured more than once");
}

pub struct App {
    renderer: HrcRenderer,
    controls: Controls,
    /// Animation clock, advanced only while playing, so pausing freezes the scene rather than
    /// jumping when it resumes.
    time: f32,
    last_tick: Instant,
    stats: PerformanceStats,
    prims: Vec<Prim>,
}

impl App {
    fn try_new(device: &Device, color_format: Format) -> kiln_rhi::RhiResult<Self> {
        let (probes, scene) = *STARTUP.get().unwrap_or(&(0, 0));
        let defaults = Controls::default();
        let controls = Controls {
            scene,
            settings: Settings {
                probes_thousands: if probes > 0 {
                    probes
                } else {
                    defaults.settings.probes_thousands
                },
                ..defaults.settings
            },
            ..defaults
        };
        Ok(Self {
            renderer: HrcRenderer::new(device, color_format)?,
            controls,
            time: 0.0,
            last_tick: Instant::now(),
            stats: PerformanceStats {
                cpu_ms: 0.0,
                gpu_ms: 0.0,
                wait_ms: 0.0,
                backend: device.backend(),
            },
            prims: Vec::new(),
        })
    }

    fn tick(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_tick).as_secs_f32();
        self.last_tick = now;
        if self.controls.animate {
            self.time += dt;
        }
    }
}

impl Example for App {
    fn new(device: &Device, color_format: Format) -> Self {
        Self::try_new(device, color_format).unwrap_or_else(|error| {
            eprintln!("{error}");
            std::process::exit(1);
        })
    }

    fn ui(&mut self, ui: &mut egui::Ui, stats: PerformanceStats) {
        // Kept for the next frame's `pre_render`, which runs after the overlay is built.
        self.stats = stats;
        let readout = Readout {
            stats,
            res: self.renderer.resolution(),
            out_res: self.renderer.output_resolution(),
            levels: self.renderer.levels_per_axis(),
            rays: self.renderer.traced_rays,
            plan: self.renderer.plan,
        };
        self.controls.show(ui, &readout);
    }

    fn pre_render(&mut self, ctx: &FrameCtx, cmd: &mut CommandBuffer) {
        self.tick();

        let (_, build) = scene::SCENES[self.controls.scene];
        self.prims = build(self.time);
        if self.controls.cursor_light {
            self.prims.push(scene::cursor_light(
                self.controls.cursor.x,
                self.controls.cursor.y,
            ));
        }

        let (res, _) =
            cascades::plan_resolution(ctx.extent, self.controls.settings.probes_thousands);
        let frame = cascades::Frame {
            device: ctx.device,
            slot: ctx.slot,
            res,
            out_res: ctx.extent,
        };
        if let Err(error) = self
            .renderer
            .record(&frame, cmd, &self.prims, self.controls.settings)
        {
            eprintln!("cascade frame failed: {error}");
        }
    }

    fn render(&mut self, _ctx: &FrameCtx, cmd: &mut CommandBuffer) {
        self.renderer.record_resolve(cmd);
    }

    fn destroy(self, device: &Device) {
        self.renderer.destroy(device);
    }
}
