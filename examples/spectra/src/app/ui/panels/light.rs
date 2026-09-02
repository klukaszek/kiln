//! The analytic light editor.
//!
//! Shape and placement live here; everything radiometric is delegated to
//! [`super::illuminant`], the editor an emissive material uses too.

use glam::DVec3;

use spectra::base::scene::{LightKind, NodeId};

use super::super::theme;
use super::super::transform::TransformEditor;
use super::super::widgets;
use super::super::{Edit, Inspector};
use super::illuminant;

pub fn show(inspector: &mut Inspector<'_>, ui: &mut egui::Ui, node_id: NodeId, index: usize) {
    let Some(light) = inspector.view.scene.lights.get(index) else {
        return;
    };
    let mut edited = light.clone();
    let mut changed = false;

    theme::section_title(ui, "light", Some(edited.kind.label().to_owned()));
    theme::card(ui, |ui| {
        widgets::grid(ui, ("light_identity", index), |ui| {
            changed |= widgets::text_row(ui, "Name", &mut edited.name);
            changed |= widgets::combo_row(
                ui,
                ("light_kind", index),
                "Shape",
                &mut edited.kind,
                LightKind::ALL,
                |kind| kind.label().to_owned(),
            );
            changed |= shape_rows(ui, &mut edited.kind);
        });
    });

    theme::section_title(ui, "transform", None);
    let mut editor = TransformEditor::resolve(
        inspector.state.transform,
        node_id,
        edited.transform,
        DVec3::ZERO,
    );
    let mut moved = false;
    theme::card(ui, |ui| {
        widgets::grid(ui, ("light_transform", index), |ui| {
            moved = editor.rows(ui, ("light", index));
        });
    });
    if moved {
        edited.transform = editor.matrix();
        editor.adopt(edited.transform);
        changed = true;
    }
    inspector.state.transform = Some(editor);

    // The extent is recomputed from the edited shape and transform, so the unit readout follows a
    // resize in the same frame the resize happens.
    let extent = edited.extent();
    changed |= illuminant::show(
        ui,
        ("light_illuminant", index),
        &mut edited.illuminant,
        extent,
        inspector.view.default_spectrum,
    );

    if changed {
        inspector.edit(Edit::Light {
            index,
            light: Box::new(edited),
        });
    }
}

/// The dimensions the selected shape needs.
fn shape_rows(ui: &mut egui::Ui, kind: &mut LightKind) -> bool {
    match kind {
        LightKind::Directional { angle_deg } => {
            widgets::scalar_row(ui, "Angular size", angle_deg, 0.0..=180.0, 0.01, "\u{00b0}")
        }
        LightKind::Rect { width, height } => {
            let mut changed =
                widgets::scalar_row(ui, "Width", width, 0.001..=100_000.0, 0.01, " m");
            changed |= widgets::scalar_row(ui, "Height", height, 0.001..=100_000.0, 0.01, " m");
            changed
        }
        LightKind::Disk { radius } | LightKind::Sphere { radius } => {
            widgets::scalar_row(ui, "Radius", radius, 0.001..=100_000.0, 0.01, " m")
        }
        LightKind::Point => {
            widgets::readonly_row(ui, "Extent", "Punctual");
            false
        }
        LightKind::Dome => {
            widgets::readonly_row(ui, "Extent", "Surrounds the scene");
            false
        }
    }
}
