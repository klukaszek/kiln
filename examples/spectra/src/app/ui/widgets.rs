//! Property-editor primitives shared by every panel.
//!
//! Panels are written entirely in terms of these rows, so field layout, label casing, and drag
//! behaviour stay identical across the render, camera, object, light, and material editors. Each
//! row reports whether it changed, which is how panels decide what to emit as an [`super::Edit`].

use std::ops::RangeInclusive;

use egui::{Align, Layout, Ui, vec2};
use glam::Vec3;

use super::theme::{self, PALETTE};

/// Gap between the three fields of an axis triple.
const AXIS_GAP: f32 = 4.0;

/// The colour chip that closes a colour row.
const SWATCH_SIZE: egui::Vec2 = egui::vec2(22.0, 16.0);

/// Width of the label column.
///
/// Fixed rather than sized to the widest label in each grid, so the value column falls on the same
/// x in every section of the panel instead of stepping in and out as sections change. Wide enough
/// for the longest label in use; anything longer is truncated rather than allowed to push the
/// column.
const LABEL_WIDTH: f32 = 112.0;

/// A two-column grid of property rows: dim label on the left, control on the right.
///
/// Column widths come from the cells the rows allocate, not from their content, so `min_col_width`
/// is left at zero and the two [`label`] / [`value`] cells govern the geometry.
pub fn grid(ui: &mut Ui, id: impl std::hash::Hash, add_contents: impl FnOnce(&mut Ui)) {
    egui::Grid::new(id)
        .num_columns(2)
        .striped(true)
        .min_col_width(0.0)
        .spacing([8.0, 4.0])
        .show(ui, add_contents);
}

/// Height of one property row: exactly the height of a drag field.
fn row_height(ui: &Ui) -> f32 {
    ui.spacing().interact_size.y
}

/// The label cell that opens every property row.
///
/// Allocated at an exact size with a centring layout, which is what puts its text on the same
/// centre line as the field beside it. Centring inside a content-sized cell does nothing: the cell
/// is only as tall as the text, so there is no slack to centre within, and the label ends up a
/// pixel or two above a field whose own text is centred in a taller control.
pub fn label(ui: &mut Ui, text: &str) {
    ui.allocate_ui_with_layout(
        vec2(LABEL_WIDTH, row_height(ui)),
        Layout::left_to_right(Align::Center),
        |ui| {
            ui.add(
                egui::Label::new(
                    egui::RichText::new(text.to_uppercase())
                        .small()
                        .monospace()
                        .color(PALETTE.ink_dim),
                )
                .truncate(),
            );
        },
    );
}

/// The value cell that closes every property row.
///
/// It claims the whole rest of the row rather than just its content's width, so every row is the
/// same width and the grid's stripes run edge to edge instead of stopping at whichever value
/// happened to be widest.
fn value<R>(ui: &mut Ui, add_contents: impl FnOnce(&mut Ui) -> R) -> R {
    let size = vec2(ui.available_width(), row_height(ui));
    ui.allocate_ui_with_layout(size, Layout::left_to_right(Align::Center), add_contents)
        .inner
}

/// A read-only row: the value is rendered as monospace ink so figures stay column-aligned.
pub fn readonly_row(ui: &mut Ui, name: &str, text: impl ToString) {
    label(ui, name);
    value(ui, |ui| {
        ui.label(
            egui::RichText::new(text.to_string())
                .monospace()
                .color(PALETTE.ink),
        );
    });
    ui.end_row();
}

/// A row whose value is worth calling out, such as an emitter's derived luminance.
pub fn highlight_row(ui: &mut Ui, name: &str, text: impl ToString, color: egui::Color32) {
    label(ui, name);
    value(ui, |ui| {
        ui.label(
            egui::RichText::new(text.to_string())
                .monospace()
                .strong()
                .color(color),
        );
    });
    ui.end_row();
}

pub fn text_row(ui: &mut Ui, name: &str, text: &mut String) -> bool {
    label(ui, name);
    let changed = value(ui, |ui| {
        ui.add(egui::TextEdit::singleline(text).desired_width(f32::INFINITY))
            .changed()
    });
    ui.end_row();
    changed
}

pub fn checkbox_row(ui: &mut Ui, name: &str, flag: &mut bool) -> bool {
    label(ui, name);
    let changed = value(ui, |ui| ui.checkbox(flag, "").changed());
    ui.end_row();
    changed
}

/// A scalar drag field. `range` clamps dragging but not typed entry, so a scene can still hold the
/// out-of-range value an importer produced.
pub fn scalar_row(
    ui: &mut Ui,
    name: &str,
    number: &mut f32,
    range: RangeInclusive<f32>,
    speed: f64,
    suffix: &str,
) -> bool {
    label(ui, name);
    let changed = field(
        ui,
        egui::DragValue::new(number)
            .range(range)
            .clamp_existing_to_range(false)
            .speed(speed)
            .suffix(suffix),
    );
    ui.end_row();
    changed
}

pub fn integer_row(
    ui: &mut Ui,
    name: &str,
    number: &mut u32,
    range: RangeInclusive<u32>,
    suffix: &str,
) -> bool {
    label(ui, name);
    let changed = field(
        ui,
        egui::DragValue::new(number)
            .range(range)
            .speed(1)
            .suffix(suffix),
    );
    ui.end_row();
    changed
}

/// A three-component drag field. `id` disambiguates the axes when several vector rows share a grid.
pub fn vec3_row(
    ui: &mut Ui,
    id: impl std::hash::Hash + Copy,
    name: &str,
    value: &mut Vec3,
    speed: f64,
    decimals: usize,
) -> bool {
    label(ui, name);
    let changed = axis_fields(
        ui,
        id,
        [&mut value.x, &mut value.y, &mut value.z],
        speed,
        decimals,
        None,
        "",
    );
    ui.end_row();
    changed
}

/// The axis triple shared by vector and transform rows. Generic over the component type because
/// colours and sizes are `f32` while transforms are kept in `f64` for large-scene precision.
///
/// The three fields are given one width derived from the row rather than each sizing to its own
/// text, so a triple does not reflow as its digits change and the columns line up down the panel.
pub fn axis_fields<T: egui::emath::Numeric>(
    ui: &mut Ui,
    id: impl std::hash::Hash + Copy,
    components: [&mut T; 3],
    speed: f64,
    decimals: usize,
    range: Option<RangeInclusive<T>>,
    suffix: &str,
) -> bool {
    let mut changed = false;
    value(ui, |ui| {
        let available = ui.available_width();
        ui.spacing_mut().item_spacing.x = AXIS_GAP;
        pin_field_width(ui, available, 3);
        for (axis, component) in ["X", "Y", "Z"].into_iter().zip(components) {
            let mut drag = egui::DragValue::new(component)
                .prefix(format!("{axis} "))
                .clamp_existing_to_range(false)
                .speed(speed)
                .fixed_decimals(decimals)
                .suffix(suffix);
            if let Some(range) = range.clone() {
                drag = drag.range(range);
            }
            changed |= ui.push_id((id, axis), |ui| ui.add(drag).changed()).inner;
        }
    });
    changed
}

/// Pin every drag field built from `ui` to one width.
///
/// `DragValue` takes `spacing.interact_size` as its minimum size, and lays itself out as
/// `prefix | growing spacer | value`. Widening it there therefore pins the axis letter to the
/// field's left edge and the digits to its right edge, which is what makes a column of triples line
/// up: identical boxes, letters on one column, numbers ending on another. `add_sized` cannot do
/// this, because it centres a text-sized button inside the cell rather than widening the button.
fn pin_field_width(ui: &mut Ui, available: f32, count: usize) {
    ui.spacing_mut().interact_size.x = pinned_width(available, count);
}

/// The width one of `count` fields gets when they share a row.
fn pinned_width(available: f32, count: usize) -> f32 {
    let gaps = AXIS_GAP * count.saturating_sub(1) as f32;
    ((available - gaps) / count as f32).max(40.0)
}

/// A lone field, sized as if it were the first of a triple so a scalar row's control starts and
/// ends on the same columns as the X field of a vector row.
fn field<W: egui::Widget>(ui: &mut Ui, widget: W) -> bool {
    value(ui, |ui| {
        let available = ui.available_width();
        pin_field_width(ui, available, 3);
        ui.add(widget).changed()
    })
}

/// A drag field paired with an exposure slider in stops. Emitter intensities span many orders of
/// magnitude, so the slider covers the range coarsely while the field stays exact.
pub fn stops_row(ui: &mut Ui, name: &str, stops: &mut f32) -> bool {
    label(ui, name);
    let mut changed = false;
    value(ui, |ui| {
        // The field takes an axis field's width and the slider fills what is left, so the readout
        // ends on the same column as every other value in the panel.
        let field_width = pinned_width(ui.available_width(), 3);
        ui.spacing_mut().item_spacing.x = AXIS_GAP;
        ui.spacing_mut().slider_width =
            (ui.available_width() - field_width - AXIS_GAP * 2.0).max(40.0);
        changed |= ui
            .add(
                egui::Slider::new(stops, -12.0..=20.0)
                    .clamping(egui::SliderClamping::Edits)
                    .show_value(false),
            )
            .changed();
        ui.spacing_mut().interact_size.x = field_width;
        changed |= ui
            .add(
                egui::DragValue::new(stops)
                    .speed(0.05)
                    .fixed_decimals(2)
                    .suffix(" EV"),
            )
            .changed();
    });
    ui.end_row();
    changed
}

/// A dropdown row. `options` are rendered with `label_of` and compared for the current selection.
pub fn combo_row<T: Clone + PartialEq>(
    ui: &mut Ui,
    id: impl std::hash::Hash,
    name: &str,
    selected: &mut T,
    options: impl IntoIterator<Item = T>,
    label_of: impl Fn(&T) -> String,
) -> bool {
    label(ui, name);
    let mut changed = false;
    value(ui, |ui| {
        // The combo fills the value cell, so its box ends on the same column as a drag field's.
        let width = ui.available_width() - ui.spacing().item_spacing.x;
        egui::ComboBox::from_id_salt(id)
            .selected_text(
                egui::RichText::new(label_of(selected))
                    .monospace()
                    .size(11.0)
                    .color(PALETTE.ink),
            )
            .width(width.max(60.0))
            .show_ui(ui, |ui| {
                for option in options {
                    let text = label_of(&option);
                    changed |= ui
                        .selectable_value(
                            selected,
                            option,
                            egui::RichText::new(text).monospace().size(11.0),
                        )
                        .changed();
                }
            });
    });
    ui.end_row();
    changed
}

/// A linear-sRGB colour row: three drag fields and a swatch showing the clamped result.
pub fn color_row(
    ui: &mut Ui,
    id: impl std::hash::Hash + Copy,
    name: &str,
    color: &mut Vec3,
) -> bool {
    label(ui, name);
    let mut changed = false;
    let mut edited = *color;
    value(ui, |ui| {
        // The swatch's share is taken out first so the three fields divide what is left, keeping
        // their width in step with the plain axis rows above and below.
        let available = ui.available_width() - SWATCH_SIZE.x - AXIS_GAP;
        ui.spacing_mut().item_spacing.x = AXIS_GAP;
        pin_field_width(ui, available, 3);
        for (channel, component) in
            ["R", "G", "B"]
                .into_iter()
                .zip([&mut edited.x, &mut edited.y, &mut edited.z])
        {
            changed |= ui
                .push_id((id, channel), |ui| {
                    ui.add(
                        egui::DragValue::new(component)
                            .prefix(format!("{channel} "))
                            .range(0.0..=1_000_000.0)
                            .clamp_existing_to_range(false)
                            .speed(0.01)
                            .fixed_decimals(3),
                    )
                    .changed()
                })
                .inner;
        }
        swatch(ui, edited);
    });
    *color = edited;
    ui.end_row();
    changed
}

/// A bracketed colour chip.
pub fn swatch(ui: &mut Ui, color: Vec3) {
    let (rect, _) = ui.allocate_exact_size(SWATCH_SIZE, egui::Sense::hover());
    let channel = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
    ui.painter().rect_filled(
        rect,
        0,
        egui::Color32::from_rgb(channel(color.x), channel(color.y), channel(color.z)),
    );
    ui.painter().rect_stroke(
        rect,
        0,
        egui::Stroke::new(theme::HAIRLINE, PALETTE.ink),
        egui::StrokeKind::Inside,
    );
}

/// A note in dim ink, for the explanations that keep a physical control honest.
pub fn note(ui: &mut Ui, text: &str) {
    ui.add_space(2.0);
    ui.label(
        egui::RichText::new(text)
            .small()
            .italics()
            .color(PALETTE.ink_dim),
    );
}

/// A message in the alert colour.
pub fn alert(ui: &mut Ui, text: &str) {
    ui.label(
        egui::RichText::new(text)
            .small()
            .monospace()
            .color(PALETTE.alert),
    );
}

/// The empty state a panel shows when nothing is selected.
pub fn placeholder(ui: &mut Ui, text: &str) {
    theme::card(ui, |ui| {
        ui.label(
            egui::RichText::new(text.to_uppercase())
                .monospace()
                .size(11.0)
                .color(PALETTE.ink_dim),
        );
    });
}
