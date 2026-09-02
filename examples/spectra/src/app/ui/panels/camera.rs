//! Viewpoint and projection.

use super::super::theme;
use super::super::widgets;
use super::super::{Edit, Inspector};

pub fn show(inspector: &mut Inspector<'_>, ui: &mut egui::Ui) {
    let mut position = inspector.view.camera_position;
    let projection = inspector.view.scene.camera.projection;
    let mut fov_degrees = projection.vertical_fov_rad.to_degrees();
    let mut moved = false;
    let mut zoomed = false;

    theme::section_title(ui, "viewpoint", None);
    theme::card(ui, |ui| {
        widgets::grid(ui, "camera_viewpoint", |ui| {
            widgets::label(ui, "Position");
            moved |= widgets::axis_fields(
                ui,
                "camera_position",
                [&mut position.x, &mut position.y, &mut position.z],
                0.05,
                2,
                None,
                "",
            );
            ui.end_row();

            widgets::label(ui, "Vertical FOV");
            zoomed |= ui
                .add(
                    egui::Slider::new(&mut fov_degrees, 10.0..=170.0)
                        .suffix("\u{00b0}")
                        .fixed_decimals(1),
                )
                .changed();
            ui.end_row();

            widgets::readonly_row(
                ui,
                "Clipping",
                format!(
                    "{:.3} \u{2192} {:.1}",
                    projection.clipping_range[0], projection.clipping_range[1]
                ),
            );
        });
        ui.add_space(4.0);
        if theme::button(ui, "reset to authored camera")
            .on_hover_text("Return to the camera the USD stage defines")
            .clicked()
        {
            inspector.edit(Edit::ResetCamera);
        }
    });

    theme::section_title(ui, "navigation", None);
    theme::card(ui, |ui| {
        for (keys, action) in [
            ("W A S D", "Move along the view plane"),
            ("Q / E", "Move down and up"),
            ("SHIFT", "Boost movement speed"),
            ("LMB DRAG", "Orbit"),
        ] {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(keys)
                        .monospace()
                        .size(11.0)
                        .color(theme::PALETTE.ink),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new(action)
                            .small()
                            .color(theme::PALETTE.ink_dim),
                    );
                });
            });
        }
    });
    widgets::note(
        ui,
        "The viewport takes keyboard and mouse only while it is focused; click it to give it \
         focus, and click the inspector to take it back.",
    );

    if moved {
        inspector.edit(Edit::CameraPosition(position));
    }
    if zoomed {
        inspector.edit(Edit::CameraFov(fov_degrees.to_radians()));
    }
}
