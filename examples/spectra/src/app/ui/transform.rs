//! The numeric transform editor.
//!
//! A matrix has no single decomposition, so editing one through position/rotation/scale fields
//! needs state: the editor remembers the Euler branch it last showed, otherwise a rotation dragged
//! past ±180° snaps to the equivalent angle and the field jumps under the cursor.

use glam::{DMat4, DQuat, DVec3, EulerRot};

use spectra::scene::NodeId;

use super::widgets;

#[derive(Clone, Copy)]
pub struct TransformEditor {
    pub node: NodeId,
    /// The point the object rotates and scales about, in object space.
    pivot: DVec3,
    /// The matrix this editor was decomposed from, used to detect external changes.
    source: DMat4,
    translation: DVec3,
    rotation_degrees: DVec3,
    scale: DVec3,
}

impl TransformEditor {
    pub fn from_matrix(node: NodeId, source: DMat4, pivot: DVec3, previous: Option<Self>) -> Self {
        let (scale, rotation, _) = source.to_scale_rotation_translation();
        let translation = source.transform_point3(pivot);
        let raw = rotation.to_euler(EulerRot::XYZ);
        let raw = DVec3::new(raw.0.to_degrees(), raw.1.to_degrees(), raw.2.to_degrees());
        let rotation_degrees = previous
            .filter(|previous| previous.node == node)
            .map(|previous| unwrap_rotation(raw, previous.rotation_degrees))
            .unwrap_or(raw);
        Self {
            node,
            pivot,
            source,
            translation,
            rotation_degrees,
            scale,
        }
    }

    /// Reuse the existing decomposition when it still describes `source`, so dragging a field does
    /// not re-derive the Euler angles from the matrix it just produced.
    pub fn resolve(previous: Option<Self>, node: NodeId, source: DMat4, pivot: DVec3) -> Self {
        previous
            .filter(|previous| {
                previous.node == node
                    && previous.pivot == pivot
                    && matrices_nearly_equal(previous.source, source)
            })
            .unwrap_or_else(|| Self::from_matrix(node, source, pivot, previous))
    }

    pub fn matrix(self) -> DMat4 {
        DMat4::from_scale_rotation_translation(
            self.scale,
            DQuat::from_euler(
                EulerRot::XYZ,
                self.rotation_degrees.x.to_radians(),
                self.rotation_degrees.y.to_radians(),
                self.rotation_degrees.z.to_radians(),
            ),
            self.translation,
        ) * DMat4::from_translation(-self.pivot)
    }

    /// Adopt a matrix this editor just produced, so the next frame reuses the decomposition rather
    /// than recovering a different one from the same rotation.
    pub fn adopt(&mut self, source: DMat4) {
        self.source = source;
    }

    /// The position, rotation, and scale rows. Returns whether any field changed.
    pub fn rows(&mut self, ui: &mut egui::Ui, id: impl std::hash::Hash + Copy) -> bool {
        let mut changed = false;

        widgets::label(ui, "Position");
        changed |= widgets::axis_fields(
            ui,
            (id, "position"),
            [
                &mut self.translation.x,
                &mut self.translation.y,
                &mut self.translation.z,
            ],
            0.05,
            2,
            None,
            "",
        );
        ui.end_row();

        widgets::label(ui, "Rotation");
        changed |= widgets::axis_fields(
            ui,
            (id, "rotation"),
            [
                &mut self.rotation_degrees.x,
                &mut self.rotation_degrees.y,
                &mut self.rotation_degrees.z,
            ],
            0.5,
            1,
            None,
            "\u{00b0}",
        );
        ui.end_row();

        widgets::label(ui, "Scale");
        changed |= widgets::axis_fields(
            ui,
            (id, "scale"),
            [&mut self.scale.x, &mut self.scale.y, &mut self.scale.z],
            0.05,
            2,
            Some(0.001..=100_000.0),
            "",
        );
        ui.end_row();

        changed
    }
}

/// Keep an angle on the branch the editor last displayed, so dragging through ±180° is continuous.
fn unwrap_rotation(raw: DVec3, previous: DVec3) -> DVec3 {
    DVec3::new(
        unwrap_angle(raw.x, previous.x),
        unwrap_angle(raw.y, previous.y),
        unwrap_angle(raw.z, previous.z),
    )
}

fn unwrap_angle(raw: f64, previous: f64) -> f64 {
    raw + 360.0 * ((previous - raw) / 360.0).round()
}

fn matrices_nearly_equal(a: DMat4, b: DMat4) -> bool {
    a.to_cols_array()
        .into_iter()
        .zip(b.to_cols_array())
        .all(|(a, b)| (a - b).abs() <= 1e-10)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_stays_on_the_previous_euler_branch() {
        assert!((unwrap_angle(-179.0, 179.0) - 181.0).abs() < f64::EPSILON);
        assert!((unwrap_angle(179.0, -179.0) + 181.0).abs() < f64::EPSILON);
    }

    #[test]
    fn a_transform_round_trips_around_its_pivot() {
        let node = NodeId(1);
        let pivot = DVec3::new(1.0, 2.0, 0.0);
        let source = DMat4::from_translation(DVec3::new(5.0, -3.0, 2.0));
        let editor = TransformEditor::from_matrix(node, source, pivot, None);
        assert!(matrices_nearly_equal(editor.matrix(), source));

        let before = source.transform_point3(pivot);
        let mut rotated = editor;
        rotated.rotation_degrees.z = 90.0;
        let after = rotated.matrix().transform_point3(pivot);
        assert!((before - after).length() < 1e-10);
    }
}
