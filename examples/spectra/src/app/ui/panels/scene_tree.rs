//! The scene hierarchy.
//!
//! Rows are drawn from the flattened, cached row list in [`super::super::state::TreeState`], so a
//! large hierarchy costs only the rows actually on screen.

use egui::{Rect, Sense, Stroke, pos2, vec2};

use spectra::base::scene::NodeObject;

use super::super::theme::{self, HAIRLINE, PALETTE, ROW_HEIGHT};
use super::super::{Edit, Inspector, Tab};

const INDENT: f32 = 14.0;
const EXPANDER: f32 = 18.0;
const ICON: f32 = 18.0;

pub fn show(inspector: &mut Inspector<'_>, ui: &mut egui::Ui) {
    header(inspector, ui);
    rows(inspector, ui);
}

fn header(inspector: &mut Inspector<'_>, ui: &mut egui::Ui) {
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        ui.add_space(theme::GUTTER);
        theme::bullet(ui, PALETTE.ink);
        ui.label(
            egui::RichText::new("SCENE")
                .monospace()
                .strong()
                .color(PALETTE.ink),
        );
        ui.label(
            egui::RichText::new(format!(
                "{} OBJ \u{2502} {} MAT \u{2502} {} LIGHT",
                inspector.view.object_count,
                inspector.view.scene.materials.len(),
                inspector.view.scene.lights.len(),
            ))
            .small()
            .monospace()
            .color(PALETTE.ink_dim),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_space(theme::GUTTER);
            add_menu(inspector, ui);
        });
    });
    ui.add_space(3.0);
    theme::rule(ui);
}

/// The scene's add menu, for objects that live in the hierarchy. Materials are library entries
/// rather than scene objects, so they are created from the material pane instead.
fn add_menu(inspector: &mut Inspector<'_>, ui: &mut egui::Ui) {
    let mut edit = None;
    ui.menu_button(egui::RichText::new(" + ").monospace().strong(), |ui| {
        ui.set_min_width(150.0);
        if menu_entry(
            ui,
            "POINT LIGHT",
            "Add a point light in front of the camera",
        ) {
            edit = Some(Edit::AddLight);
            ui.close();
        }
    })
    .response
    .on_hover_text("Add an object to the scene");
    if let Some(edit) = edit {
        inspector.edit(edit);
    }
}

fn menu_entry(ui: &mut egui::Ui, label: &str, hint: &str) -> bool {
    ui.add(
        egui::Button::new(egui::RichText::new(label).monospace().size(11.0))
            .fill(egui::Color32::TRANSPARENT)
            .stroke(Stroke::NONE),
    )
    .on_hover_text(hint)
    .clicked()
}

fn rows(inspector: &mut Inspector<'_>, ui: &mut egui::Ui) {
    inspector.state.tree.rebuild_if_dirty(inspector.view.scene);
    let mut toggle = None;
    let mut select = None;

    ui.scope(|ui| {
        ui.spacing_mut().item_spacing.y = 0.0;
        egui::ScrollArea::vertical()
            .id_salt("scene_tree")
            .auto_shrink([false, false])
            .show_rows(
                ui,
                ROW_HEIGHT,
                inspector.state.tree.visible.len(),
                |ui, range| {
                    for index in range {
                        let row = inspector.state.tree.visible[index];
                        let Some(node) = inspector.view.scene.nodes.get(row.node.0) else {
                            continue;
                        };
                        let (rect, _) = ui.allocate_exact_size(
                            vec2(ui.available_width(), ROW_HEIGHT),
                            Sense::hover(),
                        );
                        let selected = inspector.state.selected_node == Some(row.node);
                        let response =
                            ui.interact(rect, ui.id().with(("row", row.node)), Sense::click());
                        let ink = theme::row_background(ui, rect, selected, response.hovered());

                        let indent = rect.left() + 6.0 + f32::from(row.depth) * INDENT;
                        let expander = Rect::from_center_size(
                            pos2(indent + EXPANDER * 0.5, rect.center().y),
                            vec2(EXPANDER, EXPANDER),
                        );
                        let expandable = node.is_group() && !node.children.is_empty();
                        let expanded = inspector.state.tree.expanded.contains(&row.node);
                        let mut hit_expander = false;
                        if node.is_group() {
                            let expander_response = ui.interact(
                                expander,
                                ui.id().with(("expander", row.node)),
                                if expandable {
                                    Sense::click()
                                } else {
                                    Sense::hover()
                                },
                            );
                            draw_expander(ui, expander, expanded, expandable, ink);
                            if expandable && expander_response.clicked() {
                                toggle = Some(row.node);
                                hit_expander = true;
                            }
                        }

                        let icon = Rect::from_center_size(
                            pos2(expander.right() + 4.0 + ICON * 0.5, rect.center().y),
                            vec2(ICON, ICON),
                        );
                        draw_icon(ui, icon, node.object, ink);
                        ui.painter().text(
                            pos2(icon.right() + 6.0, rect.center().y),
                            egui::Align2::LEFT_CENTER,
                            &node.name,
                            egui::FontId::monospace(12.0),
                            ink,
                        );
                        if emits(inspector, node.object) {
                            emission_marker(ui, rect);
                        }
                        ui.painter().line_segment(
                            [rect.left_bottom(), rect.right_bottom()],
                            Stroke::new(HAIRLINE, PALETTE.ink.gamma_multiply(0.15)),
                        );

                        if response.clicked() && !hit_expander {
                            select = Some(row.node);
                        }
                        response.on_hover_text(&node.path);
                    }
                },
            );
    });

    if let Some(node) = toggle {
        inspector.state.tree.toggle(node);
    }
    if let Some(node) = select {
        inspector.state.selected_node = Some(node);
        inspector.state.tab = Tab::Object;
    }
}

/// Whether a node emits light, so the tree can mark the emitters in a scene at a glance. Emissive
/// meshes are as much a light as a lux prim, and this is the only place that shows both as one.
fn emits(inspector: &Inspector<'_>, object: NodeObject) -> bool {
    let scene = inspector.view.scene;
    match object {
        NodeObject::Group => false,
        NodeObject::Light { light } => scene
            .lights
            .get(light)
            .is_some_and(|light| light.illuminant.emits()),
        NodeObject::Mesh { instance } => scene
            .geometry
            .instances
            .get(instance)
            .and_then(|instance| scene.geometry.meshes.get(instance.mesh.0))
            .is_some_and(|mesh| !mesh.emissive_components.is_empty()),
    }
}

/// A filled wedge on the trailing edge of an emitter's row.
fn emission_marker(ui: &egui::Ui, rect: Rect) {
    let centre = pos2(rect.right() - 10.0, rect.center().y);
    ui.painter().add(egui::Shape::convex_polygon(
        vec![
            centre + vec2(0.0, -4.0),
            centre + vec2(4.0, 3.0),
            centre + vec2(-4.0, 3.0),
        ],
        PALETTE.emissive,
        Stroke::NONE,
    ));
}

fn draw_expander(ui: &egui::Ui, rect: Rect, expanded: bool, expandable: bool, ink: egui::Color32) {
    if !expandable {
        return;
    }
    let centre = rect.center();
    let points = if expanded {
        vec![
            centre + vec2(-3.5, -1.5),
            centre + vec2(3.5, -1.5),
            centre + vec2(0.0, 3.0),
        ]
    } else {
        vec![
            centre + vec2(-1.5, -3.5),
            centre + vec2(3.0, 0.0),
            centre + vec2(-1.5, 3.5),
        ]
    };
    ui.painter()
        .add(egui::Shape::convex_polygon(points, ink, Stroke::NONE));
}

/// Line-art glyphs for the three node kinds, drawn rather than shipped as a font.
fn draw_icon(ui: &egui::Ui, rect: Rect, object: NodeObject, ink: egui::Color32) {
    let painter = ui.painter();
    let stroke = Stroke::new(1.2, ink);
    let centre = rect.center();
    match object {
        NodeObject::Mesh { .. } => {
            let top = centre + vec2(0.0, -6.0);
            let bottom = centre + vec2(0.0, 6.0);
            let (left, right) = (centre + vec2(-6.0, -2.5), centre + vec2(6.0, -2.5));
            let (lower_left, lower_right) = (centre + vec2(-6.0, 2.5), centre + vec2(6.0, 2.5));
            for segment in [
                [top, left],
                [top, right],
                [left, lower_left],
                [right, lower_right],
                [lower_left, bottom],
                [lower_right, bottom],
                [left, right],
            ] {
                painter.line_segment(segment, stroke);
            }
        }
        NodeObject::Light { .. } => {
            painter.circle_stroke(centre, 3.0, stroke);
            for (dx, dy) in [
                (0.0, -7.0),
                (0.0, 7.0),
                (-7.0, 0.0),
                (7.0, 0.0),
                (-5.0, -5.0),
                (5.0, 5.0),
                (5.0, -5.0),
                (-5.0, 5.0),
            ] {
                let direction = vec2(dx, dy).normalized();
                painter.line_segment([centre + direction * 4.5, centre + direction * 6.5], stroke);
            }
        }
        NodeObject::Group => {
            painter.rect_stroke(rect.shrink(5.0), 0, stroke, egui::StrokeKind::Inside);
        }
    }
}
