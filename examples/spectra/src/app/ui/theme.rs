//! The inspector's design language.
//!
//! Everything visual lives here: the palette, the egui style derived from it, and the small set of
//! painters the panels draw with. Panels describe *what* they show; this module decides how it
//! looks, so a change of look never means editing a panel.
//!
//! The language is the flat, warm-paper, high-contrast one of NieR:Automata's menus: square
//! corners, hairline rules, uppercase labels, filled square bullets, selection by inversion rather
//! than by highlight, and faint horizontal scanlines over the panel ground.

use egui::{Color32, Rect, Response, Stroke, Ui, Vec2, pos2, vec2};

/// Warm paper ground and dark ink, the two poles the whole interface is built from.
pub struct Palette {
    /// Panel ground.
    pub surface: Color32,
    /// Recessed ground: striped rows, wells, the tab rail.
    pub sunken: Color32,
    /// Raised ground: cards, grouped property blocks, and controls at rest.
    pub raised: Color32,
    /// A control under the pointer.
    pub hover: Color32,
    /// A control being pressed or dragged.
    ///
    /// Engaging a control lightens it rather than inverting it. Inversion is reserved for row
    /// selection, and a dark fill under a value being scrubbed is the one place it would hide the
    /// number the user is watching.
    pub engaged: Color32,
    /// Primary text and rules.
    pub ink: Color32,
    /// Secondary text: labels, units, counts.
    pub ink_dim: Color32,
    /// Disabled text and inactive rules.
    pub ink_faint: Color32,
    /// Selection fill. Selection inverts a row rather than tinting it.
    pub selected: Color32,
    /// Text drawn on [`Self::selected`].
    pub on_selected: Color32,
    /// Errors and destructive actions.
    pub alert: Color32,
    /// Emitters: the marker on an emissive material or a light.
    pub emissive: Color32,
}

pub const PALETTE: Palette = Palette {
    surface: Color32::from_rgb(202, 197, 173),
    sunken: Color32::from_rgb(186, 181, 158),
    raised: Color32::from_rgb(213, 208, 186),
    hover: Color32::from_rgb(226, 222, 203),
    engaged: Color32::from_rgb(240, 237, 222),
    ink: Color32::from_rgb(69, 65, 56),
    ink_dim: Color32::from_rgb(112, 106, 90),
    ink_faint: Color32::from_rgb(148, 142, 124),
    selected: Color32::from_rgb(69, 65, 56),
    on_selected: Color32::from_rgb(213, 208, 186),
    alert: Color32::from_rgb(156, 58, 40),
    emissive: Color32::from_rgb(166, 118, 46),
};

/// Hairline width. Every rule, border, and bracket in the interface is drawn at this weight.
pub const HAIRLINE: f32 = 1.0;

/// Height of one row in the scene tree and the list panels.
pub const ROW_HEIGHT: f32 = 24.0;

/// Horizontal breathing room between the dock's edge and its content. Selection bars and rules run
/// full width for the inverted-row look, so the gutter is applied to content rather than to the
/// panel frame.
pub const GUTTER: f32 = 10.0;

/// Install the palette as egui's global style. Called once, when the first frame is built.
pub fn install(ctx: &egui::Context) {
    let mut style = (*ctx.global_style()).clone();

    style.spacing.item_spacing = vec2(8.0, 5.0);
    style.spacing.button_padding = vec2(8.0, 3.0);
    style.spacing.window_margin = egui::Margin::same(0);
    style.spacing.slider_width = 120.0;
    style.spacing.interact_size.y = 20.0;

    let visuals = &mut style.visuals;
    visuals.dark_mode = false;
    visuals.override_text_color = Some(PALETTE.ink);
    visuals.panel_fill = PALETTE.surface;
    visuals.window_fill = PALETTE.surface;
    visuals.extreme_bg_color = PALETTE.sunken;
    visuals.faint_bg_color = PALETTE.sunken;
    visuals.selection.bg_fill = PALETTE.selected;
    visuals.selection.stroke = Stroke::new(HAIRLINE, PALETTE.on_selected);
    visuals.window_stroke = Stroke::new(HAIRLINE, PALETTE.ink);
    visuals.window_shadow = egui::Shadow::NONE;
    visuals.popup_shadow = egui::Shadow::NONE;
    visuals.window_corner_radius = egui::CornerRadius::ZERO;
    visuals.menu_corner_radius = egui::CornerRadius::ZERO;

    // Controls read as flat paper chips that lighten as they are engaged, keeping the ink text
    // legible at every step. Inversion belongs to row selection, not to widgets.
    let widgets = &mut visuals.widgets;
    for widget in [
        &mut widgets.noninteractive,
        &mut widgets.inactive,
        &mut widgets.hovered,
        &mut widgets.active,
        &mut widgets.open,
    ] {
        widget.corner_radius = egui::CornerRadius::ZERO;
        widget.expansion = 0.0;
        widget.bg_stroke = Stroke::new(HAIRLINE, PALETTE.ink_faint);
        widget.fg_stroke = Stroke::new(HAIRLINE, PALETTE.ink);
        widget.weak_bg_fill = PALETTE.raised;
        widget.bg_fill = PALETTE.raised;
    }
    widgets.noninteractive.bg_stroke = Stroke::new(HAIRLINE, PALETTE.ink_faint);
    widgets.noninteractive.fg_stroke = Stroke::new(HAIRLINE, PALETTE.ink_dim);
    widgets.inactive.bg_fill = PALETTE.raised;
    widgets.inactive.weak_bg_fill = PALETTE.raised;
    widgets.hovered.bg_fill = PALETTE.hover;
    widgets.hovered.weak_bg_fill = PALETTE.hover;
    widgets.hovered.bg_stroke = Stroke::new(HAIRLINE, PALETTE.ink_dim);
    widgets.active.bg_fill = PALETTE.engaged;
    widgets.active.weak_bg_fill = PALETTE.engaged;
    widgets.active.fg_stroke = Stroke::new(HAIRLINE, PALETTE.ink);
    widgets.active.bg_stroke = Stroke::new(HAIRLINE, PALETTE.ink);
    widgets.open.bg_fill = PALETTE.engaged;
    widgets.open.weak_bg_fill = PALETTE.engaged;
    widgets.open.bg_stroke = Stroke::new(HAIRLINE, PALETTE.ink);

    use egui::{FontFamily, FontId, TextStyle};
    style.text_styles = [
        (TextStyle::Heading, FontId::new(15.0, FontFamily::Monospace)),
        (TextStyle::Body, FontId::new(13.0, FontFamily::Proportional)),
        (TextStyle::Button, FontId::new(12.0, FontFamily::Monospace)),
        (
            TextStyle::Small,
            FontId::new(11.0, FontFamily::Proportional),
        ),
        (
            TextStyle::Monospace,
            FontId::new(12.0, FontFamily::Monospace),
        ),
    ]
    .into();

    ctx.set_global_style(style);
}

/// The panel ground: flat fill, no border, no rounding.
pub fn panel_frame() -> egui::Frame {
    egui::Frame::new().fill(PALETTE.surface)
}

/// The top strip above the viewport.
pub fn toolbar_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(PALETTE.sunken)
        .inner_margin(egui::Margin::symmetric(10, 5))
}

/// A raised block of related properties. Bracketed rather than boxed: the corner marks read as
/// grouping without drawing a full border around every field.
pub fn card(ui: &mut Ui, add_contents: impl FnOnce(&mut Ui)) {
    let response = egui::Frame::new()
        .fill(PALETTE.raised)
        .inner_margin(egui::Margin::symmetric(10, 8))
        .show(ui, add_contents);
    brackets(ui, response.response.rect, 7.0, PALETTE.ink_faint);
    ui.add_space(6.0);
}

/// Corner brackets: four L-shaped marks that frame a rect without enclosing it.
pub fn brackets(ui: &Ui, rect: Rect, arm: f32, color: Color32) {
    let painter = ui.painter();
    let stroke = Stroke::new(HAIRLINE, color);
    for (corner, dx, dy) in [
        (rect.left_top(), 1.0, 1.0),
        (rect.right_top(), -1.0, 1.0),
        (rect.left_bottom(), 1.0, -1.0),
        (rect.right_bottom(), -1.0, -1.0),
    ] {
        painter.line_segment([corner, corner + vec2(arm * dx, 0.0)], stroke);
        painter.line_segment([corner, corner + vec2(0.0, arm * dy)], stroke);
    }
}

/// A full-width hairline rule.
pub fn rule(ui: &mut Ui) {
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(vec2(width, HAIRLINE), egui::Sense::hover());
    ui.painter().line_segment(
        [
            pos2(rect.left(), rect.center().y),
            pos2(rect.right(), rect.center().y),
        ],
        Stroke::new(HAIRLINE, PALETTE.ink_faint),
    );
}

/// A section title: uppercase ink over a rule, with the count or unit trailing in dim ink.
pub fn section_title(ui: &mut Ui, title: &str, trailing: Option<String>) {
    ui.horizontal(|ui| {
        bullet(ui, PALETTE.ink);
        ui.label(
            egui::RichText::new(title.to_uppercase())
                .monospace()
                .strong()
                .color(PALETTE.ink),
        );
        if let Some(trailing) = trailing {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    egui::RichText::new(trailing.to_uppercase())
                        .small()
                        .color(PALETTE.ink_dim),
                );
            });
        }
    });
    ui.add_space(2.0);
    rule(ui);
    ui.add_space(5.0);
}

/// The filled square that marks a heading or a list entry.
pub fn bullet(ui: &mut Ui, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(7.0), egui::Sense::hover());
    ui.painter().rect_filled(
        Rect::from_center_size(rect.center(), Vec2::splat(6.0)),
        0,
        color,
    );
}

/// Paint a selectable row's ground. Selection inverts: ink fill, paper text, with a solid marker
/// on the leading edge. Hover only tints, so the two states never read the same.
pub fn row_background(ui: &Ui, rect: Rect, selected: bool, hovered: bool) -> Color32 {
    let painter = ui.painter();
    if selected {
        painter.rect_filled(rect, 0, PALETTE.selected);
        painter.rect_filled(
            Rect::from_min_size(rect.left_top(), vec2(3.0, rect.height())),
            0,
            PALETTE.emissive,
        );
        PALETTE.on_selected
    } else {
        if hovered {
            painter.rect_filled(rect, 0, PALETTE.sunken);
        }
        PALETTE.ink
    }
}

/// Faint horizontal scanlines over a rect. The texture that keeps large flat areas from reading as
/// dead space; drawn at low contrast so it never competes with text.
///
/// A `Ui`'s available rect can be far taller than what is on screen, so the band is clipped to the
/// visible clip rect first. Stepping over an unclipped rect would emit lines by the hundred
/// thousand, all of them off screen and all of them tessellated every frame.
pub fn scanlines(ui: &Ui, rect: Rect) {
    let painter = ui.painter();
    let rect = rect.intersect(painter.clip_rect());
    if !rect.is_positive() {
        return;
    }
    const STEP: f32 = 3.0;
    let color = PALETTE.ink.gamma_multiply(0.05);
    let stroke = Stroke::new(HAIRLINE, color);
    let count = (rect.height() / STEP) as usize;
    let lines = (0..count).map(|index| {
        let y = rect.top() + index as f32 * STEP;
        egui::Shape::line_segment([pos2(rect.left(), y), pos2(rect.right(), y)], stroke)
    });
    painter.extend(lines);
}

/// A flat bracketed button. Returns its response so callers can attach hover text.
pub fn button(ui: &mut Ui, label: &str) -> Response {
    ui.add(
        egui::Button::new(
            egui::RichText::new(label.to_uppercase())
                .monospace()
                .size(11.0),
        )
        .fill(PALETTE.raised)
        .stroke(Stroke::new(HAIRLINE, PALETTE.ink_faint)),
    )
}

/// A button that reads as destructive.
pub fn danger_button(ui: &mut Ui, label: &str, enabled: bool) -> Response {
    ui.add_enabled(
        enabled,
        egui::Button::new(
            egui::RichText::new(label.to_uppercase())
                .monospace()
                .size(11.0)
                .color(if enabled {
                    PALETTE.alert
                } else {
                    PALETTE.ink_faint
                }),
        )
        .fill(PALETTE.raised)
        .stroke(Stroke::new(
            HAIRLINE,
            if enabled {
                PALETTE.alert
            } else {
                PALETTE.ink_faint
            },
        )),
    )
}
