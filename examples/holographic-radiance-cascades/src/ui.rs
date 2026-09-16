//! The control window, and the pointer and key handling that goes with it.
//!
//! The overlay is the harness's: this crate turns on `kiln-app`'s `egui` feature and fills
//! [`kiln_app::Example::ui`], so the painter is `kiln-egui` and the context is the one the harness
//! already drives. Everything here is presentation — it reads the renderer's settings, writes them
//! back, and never touches the renderer itself.

use glam::{UVec2, Vec2};
use kiln_app::PerformanceStats;

use crate::cascades::{DIRECT_TRACE_LEVELS, Plan, Settings, ViewMode};
use crate::scene::SCENES;

/// Everything the window owns, and what the app reads back out of it each frame.
pub struct Controls {
    pub open: bool,
    pub settings: Settings,
    pub scene: usize,
    pub animate: bool,
    pub cursor_light: bool,
    /// Pointer-driven emitter, in the normalised square the scenes are authored in.
    pub cursor: Vec2,
}

impl Default for Controls {
    fn default() -> Self {
        Self {
            open: true,
            settings: Settings::default(),
            scene: 0,
            animate: true,
            cursor_light: true,
            cursor: Vec2::new(0.5, 0.18),
        }
    }
}

/// What the window displays but does not own.
pub struct Readout {
    pub stats: PerformanceStats,
    /// Probe grid the light field is solved on.
    pub res: UVec2,
    /// Swapchain resolution the image is resolved at.
    pub out_res: UVec2,
    /// Cascade levels per axis, for the two gather parities.
    pub levels: (u32, u32),
    pub rays: u64,
    pub plan: Plan,
}

impl Controls {
    pub fn show(&mut self, ui: &mut egui::Ui, readout: &Readout) {
        self.keys(ui.ctx());
        self.drag(ui.ctx(), readout);
        let mut open = self.open;
        egui::Window::new("Radiance Cascades")
            .open(&mut open)
            .default_pos([16.0, 16.0])
            .default_width(300.0)
            .show(ui.ctx(), |ui| {
                self.readout(ui, readout);
                ui.separator();

                egui::ComboBox::from_label("Scene")
                    .selected_text(SCENES[self.scene].0)
                    .show_ui(ui, |ui| {
                        for (index, (name, _)) in SCENES.iter().enumerate() {
                            ui.selectable_value(&mut self.scene, index, *name);
                        }
                    });
                ui.checkbox(&mut self.animate, "Animate");
                ui.checkbox(&mut self.cursor_light, "Cursor light (drag)");

                ui.separator();
                ui.add(
                    egui::Slider::new(&mut self.settings.probes_thousands, 30..=2000)
                        .text("Probes (thousands)"),
                );
                ui.add(
                    egui::Slider::new(&mut self.settings.blur_passes, 0..=4)
                        .text("Cross blur passes"),
                );
                ui.add(
                    egui::Slider::new(&mut self.settings.surface_falloff, 0.002..=0.15)
                        .text("Depth falloff"),
                );
                // The bounce loop is only stable for gain <= 1; above that a closed box with a high
                // albedo diverges until the half stores saturate.
                ui.add(
                    egui::Slider::new(&mut self.settings.bounce, 0.0..=1.0).text("Diffuse bounce"),
                );
                ui.add(egui::Slider::new(&mut self.settings.exposure, 0.05..=6.0).text("Exposure"));
                ui.add(egui::Slider::new(&mut self.settings.sky, 0.0..=1.0).text("Sky"));

                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("View");
                    ui.selectable_value(&mut self.settings.view_mode, ViewMode::Final, "Final");
                    ui.selectable_value(
                        &mut self.settings.view_mode,
                        ViewMode::LightField,
                        "Light field",
                    );
                });
                ui.checkbox(&mut self.settings.surface_shading, "Surface shading");
                ui.weak("keys: 1-4 scene, space pause, esc quit");
            });
        self.open = open;
    }

    fn readout(&self, ui: &mut egui::Ui, readout: &Readout) {
        let Readout {
            stats,
            res,
            out_res,
            levels,
            rays,
            plan,
        } = readout;
        // The harness reports the queue time it brackets, which is the honest number for a frame
        // that is almost entirely compute. Wall time includes the present wait.
        let wall = stats.cpu_ms + stats.wait_ms;
        let gigarays = *rays as f64 / stats.gpu_ms.max(1e-6) / 1.0e6;

        ui.label(format!(
            "{:.2} ms gpu ({:.0} fps, {gigarays:.2} Gray/s)",
            stats.gpu_ms,
            1000.0 / wall.max(1e-3)
        ));
        ui.weak(format!(
            "{:?} · cpu {:.2} ms · present wait {:.2} ms",
            stats.backend, stats.cpu_ms, stats.wait_ms
        ));
        ui.weak(format!(
            "resolve {}x{} · light {}x{} · {}/{} levels",
            out_res.x, out_res.y, res.x, res.y, levels.0, levels.1
        ));
        ui.weak(format!(
            "{:.2} M rays/frame (levels 0-{}) · cascades {} MiB",
            *rays as f64 / 1.0e6,
            DIRECT_TRACE_LEVELS - 1,
            plan.bytes() / (1024 * 1024)
        ));
    }

    /// `1`-`4` pick a scene, space pauses. Esc is the harness's.
    fn keys(&mut self, ctx: &egui::Context) {
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        ctx.input(|input| {
            for (index, key) in [
                egui::Key::Num1,
                egui::Key::Num2,
                egui::Key::Num3,
                egui::Key::Num4,
            ]
            .into_iter()
            .enumerate()
            {
                if input.key_pressed(key) && index < SCENES.len() {
                    self.scene = index;
                }
            }
            if input.key_pressed(egui::Key::Space) {
                self.animate = !self.animate;
            }
        });
    }

    /// Dragging anywhere the window is not moves the cursor emitter.
    fn drag(&mut self, ctx: &egui::Context, readout: &Readout) {
        if ctx.egui_wants_pointer_input() || readout.out_res.min_element() == 0 {
            return;
        }
        let Some(pos) = ctx.pointer_interact_pos() else {
            return;
        };
        if !ctx.input(|input| input.pointer.primary_down()) {
            return;
        }
        // egui works in points; the light field is indexed in swapchain pixels.
        let pixels = Vec2::new(pos.x, pos.y) * ctx.pixels_per_point();
        let probe = pixels * readout.res.as_vec2() / readout.out_res.as_vec2();
        // Scenes are authored in the unit square, centred in the grid's shorter axis.
        let side = readout.res.min_element() as f32;
        self.cursor = (probe - 0.5 * (readout.res.as_vec2() - Vec2::splat(side))) / side;
    }
}
