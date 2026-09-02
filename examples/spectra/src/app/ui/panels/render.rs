//! Sampling and resolution controls for the progressive path tracer.

use super::super::theme;
use super::super::widgets;
use super::super::{Edit, Inspector};

pub fn show(inspector: &mut Inspector<'_>, ui: &mut egui::Ui) {
    let mut settings = inspector.view.settings;
    let mut changed = false;

    theme::section_title(ui, "sampling", None);
    theme::card(ui, |ui| {
        widgets::grid(ui, "render_sampling", |ui| {
            changed |= widgets::integer_row(
                ui,
                "Target",
                &mut settings.target_spp,
                1..=1_000_000,
                " spp",
            );
            changed |= widgets::integer_row(
                ui,
                "Per frame",
                &mut settings.passes_per_frame,
                1..=1024,
                " passes",
            );
        });
    });
    widgets::note(
        ui,
        "Passes per frame trades interactivity for convergence rate; the film keeps accumulating \
         until the target is reached.",
    );
    ui.add_space(8.0);

    theme::section_title(ui, "resolution", None);
    theme::card(ui, |ui| {
        widgets::grid(ui, "render_resolution", |ui| {
            changed |=
                widgets::integer_row(ui, "Render scale", &mut settings.render_scale, 1..=16, "");
            changed |=
                widgets::integer_row(ui, "Pixel stride", &mut settings.pixel_stride, 1..=16, "");
            let extent = inspector.view.viewport_extent;
            let scale = settings.render_scale.max(1);
            widgets::readonly_row(
                ui,
                "Trace extent",
                format!("{}\u{00d7}{}", extent.x / scale, extent.y / scale),
            );
        });
    });
    widgets::note(
        ui,
        "Render scale divides the traced resolution and the blit upscales it. Pixel stride \
         interleaves an NxN grid across frames at full film resolution.",
    );

    if changed {
        inspector.edit(Edit::Settings(settings));
    }
}
