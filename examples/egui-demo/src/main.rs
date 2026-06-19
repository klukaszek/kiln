//! Windowed egui demo rendered through the Kiln RHI.
//!
//! Unlike the other demos, this one's content *is* the egui overlay: its crate turns on
//! `kiln-app`'s `egui` feature, so the harness drives the overlay and this file just fills
//! [`Example::ui`], leaving [`Example::render`] empty — the harness clears the screen and paints
//! the UI on top. It proves the end-to-end egui path: font-atlas sampling through the bindless
//! heap, premultiplied-alpha blending, and per-primitive scissor clipping.
//!
//! Run with: `cargo run -p egui-demo` (needs `slangc` on PATH).

use kiln_app::{Example, FrameCtx};
use kiln_rhi::{CommandBuffer, Device, Format};

struct Demo {
    name: String,
    age: u32,
    checked: bool,
}

impl Example for Demo {
    fn new(_device: &Device, _color_format: Format) -> Self {
        Self {
            name: "Kiln".to_string(),
            age: 36,
            checked: true,
        }
    }

    // The harness's cleared swapchain pass is all the "render" this demo needs; the overlay (on
    // whenever the crate is built with the default `egui` feature) draws the UI on top.
    fn render(&mut self, _ctx: &FrameCtx, _cmd: &mut CommandBuffer) {}

    fn ui(&mut self, ui: &mut egui::Ui) {
        // `show_inside(ui)` (rather than `show(ctx)`) nests the panel inside the harness's pass
        // root, below its stats bar; it restores the central panel's background fill and margins.
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.heading("Kiln RHI · egui");
            ui.separator();
            ui.horizontal(|ui| {
                ui.label("Name:");
                ui.text_edit_singleline(&mut self.name);
            });
            ui.add(egui::Slider::new(&mut self.age, 0..=120).text("age"));
            if ui.button("Have a birthday").clicked() {
                self.age += 1;
            }
            ui.checkbox(&mut self.checked, "A checkbox");
            ui.label(format!("{} is {}", self.name, self.age));
        });

        egui::Window::new("Draggable window")
            .default_pos([420.0, 80.0])
            .show(ui.ctx(), |ui| {
                ui.label("Drag me around to test scissor clipping.");
                ui.spinner();
            });
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    kiln_app::run::<Demo>("Kiln · egui demo", [0.05, 0.05, 0.08, 1.0])
}
