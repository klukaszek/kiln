//! The scene inspector.
//!
//! The inspector reads a [`View`] of the frame and returns the [`Edit`]s it wants applied. It never
//! touches the scene, the renderer, or the camera controller, which is what keeps every panel
//! independently readable: a panel's whole contract is "given this view, emit these edits".
//!
//! Layout is a left dock over the viewport: the scene tree on top, a draggable splitter, then a tab
//! strip and the properties for the open tab. [`theme`] owns how all of it looks.

mod panels;
mod state;
pub mod theme;
mod transform;
mod widgets;

use glam::{DMat4, DVec3, UVec2};
use kiln_app::PerformanceStats;

use spectra::base::scene::{Light, Material, MaterialId, Scene};
use spectra::renderers::spectral::Settings;

pub use state::{InspectorState, Tab};

/// Everything the inspector may read while building a frame.
pub struct View<'a> {
    pub scene: &'a Scene,
    pub settings: Settings,
    pub stats: PerformanceStats,
    /// Smoothed time spent building this overlay. The harness's CPU figure excludes egui, so
    /// without this the inspector's own cost is the one part of the frame nothing reports.
    pub ui_ms: f64,
    /// `None` when the active renderer does not accumulate samples.
    pub samples_drawn: Option<u32>,
    pub viewport_extent: UVec2,
    pub renderer_error: Option<&'a str>,
    /// Camera position from the controller, which owns it between frames.
    pub camera_position: DVec3,
    /// Object-space centre of each mesh, used as the transform editor's pivot.
    pub mesh_pivots: &'a [DVec3],
    /// Triangles bound to each material, indexed by material id.
    pub material_usage: &'a [usize],
    /// World-space area of each material's emissive regions, indexed by material id. This is what a
    /// total-power intensity is spread over, and it mirrors what the spectral backend computes.
    pub emissive_areas: &'a [f32],
    /// Non-group nodes, shown as the scene's object count.
    pub object_count: usize,
    /// The renderer-wide spectrum name, which unresolved illuminant spectra fall back to.
    pub default_spectrum: &'a str,
}

/// A change the inspector asks the application to make.
///
/// Edits are values, not closures over the scene, so applying them is one place that knows how to
/// mark the matching renderer transaction.
pub enum Edit {
    Settings(Settings),
    CameraPosition(DVec3),
    CameraFov(f32),
    ResetCamera,
    AddLight,
    Light {
        index: usize,
        light: Box<Light>,
    },
    InstanceTransform {
        index: usize,
        transform: DMat4,
    },
    Material {
        index: usize,
        material: Box<Material>,
    },
    AddMaterial,
    DeleteMaterial(MaterialId),
}

/// Draws the inspector and collects its edits.
pub struct Inspector<'a> {
    pub(crate) view: View<'a>,
    pub(crate) state: &'a mut InspectorState,
    edits: Vec<Edit>,
}

impl<'a> Inspector<'a> {
    pub fn new(state: &'a mut InspectorState, view: View<'a>) -> Self {
        Self {
            view,
            state,
            edits: Vec::new(),
        }
    }

    pub(crate) fn edit(&mut self, edit: Edit) {
        self.edits.push(edit);
    }

    /// Build the whole overlay: dock, toolbar, and the viewport's input region. Returns the edits
    /// requested this frame.
    pub fn show(mut self, ui: &mut egui::Ui) -> Vec<Edit> {
        self.state.install_style_once(ui.ctx());
        if self.state.open {
            self.dock(ui);
        }
        self.toolbar(ui);
        self.viewport(ui);
        self.edits
    }

    /// The left dock: scene tree, splitter, tab strip, properties.
    fn dock(&mut self, ui: &mut egui::Ui) {
        egui::Panel::left("spectra_inspector")
            .default_size(360.0)
            .resizable(true)
            .frame(theme::panel_frame())
            .show_inside(ui, |ui| {
                // Pin the contents to the rect the panel allocated. Child scopes must not grow this
                // Ui from a long label or scroll content, or the resize handle drifts off the edge.
                let dock = ui.available_rect_before_wrap();
                ui.set_width(dock.width());
                ui.set_height(dock.height());
                theme::scanlines(ui, dock);

                let splitter_height = 7.0;
                let tree_height = self.split(dock, splitter_height);
                let tree_rect =
                    egui::Rect::from_min_size(dock.min, egui::vec2(dock.width(), tree_height));
                let splitter_rect = egui::Rect::from_min_size(
                    egui::pos2(dock.left(), tree_rect.bottom()),
                    egui::vec2(dock.width(), splitter_height),
                );
                let properties_rect =
                    egui::Rect::from_min_max(splitter_rect.left_bottom(), dock.max);

                region(ui, tree_rect, |ui| panels::scene_tree::show(self, ui));
                self.splitter(ui, splitter_rect, dock, splitter_height);
                region(ui, properties_rect, |ui| {
                    self.tab_strip(ui);
                    egui::ScrollArea::vertical()
                        .id_salt("inspector_properties")
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            // The gutter is applied inside the scroll area so the scrollbar still
                            // sits on the panel edge.
                            egui::Frame::new()
                                .inner_margin(egui::Margin {
                                    left: theme::GUTTER as i8,
                                    right: theme::GUTTER as i8,
                                    top: 8,
                                    bottom: 12,
                                })
                                .show(ui, |ui| {
                                    ui.set_width(
                                        properties_rect.width() - theme::GUTTER * 2.0 - 12.0,
                                    );
                                    self.properties(ui);
                                });
                        });
                });
            });
    }

    /// Clamp and store the scene tree's share of the dock.
    fn split(&mut self, dock: egui::Rect, splitter_height: f32) -> f32 {
        const MIN_TREE: f32 = 120.0;
        const MIN_PROPERTIES: f32 = 180.0;
        let max = (dock.height() - splitter_height - MIN_PROPERTIES).max(MIN_TREE);
        let height = self.state.tree_height.clamp(MIN_TREE, max);
        self.state.tree_height = height;
        height
    }

    fn splitter(
        &mut self,
        ui: &mut egui::Ui,
        rect: egui::Rect,
        dock: egui::Rect,
        splitter_height: f32,
    ) {
        let response = ui.interact(rect, ui.id().with("tree_splitter"), egui::Sense::drag());
        if response.dragged() {
            self.state.tree_height += response.drag_delta().y;
            self.split(dock, splitter_height);
        }
        if response.hovered() || response.dragged() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
        }
        let painter = ui.painter();
        painter.rect_filled(rect, 0, theme::PALETTE.sunken);
        let color = if response.hovered() || response.dragged() {
            theme::PALETTE.ink
        } else {
            theme::PALETTE.ink_faint
        };
        // A grip of three short ticks, so the splitter reads as draggable without a handle widget.
        let centre = rect.center();
        for offset in [-8.0, 0.0, 8.0] {
            painter.line_segment(
                [
                    egui::pos2(centre.x + offset - 2.0, centre.y),
                    egui::pos2(centre.x + offset + 2.0, centre.y),
                ],
                egui::Stroke::new(theme::HAIRLINE, color),
            );
        }
    }

    /// The horizontal tab strip above the properties.
    fn tab_strip(&mut self, ui: &mut egui::Ui) {
        let height = 26.0;
        let (strip, _) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), height),
            egui::Sense::hover(),
        );
        ui.painter().rect_filled(strip, 0, theme::PALETTE.sunken);
        let width = strip.width() / Tab::ALL.len() as f32;
        for (index, tab) in Tab::ALL.into_iter().enumerate() {
            let rect = egui::Rect::from_min_size(
                egui::pos2(strip.left() + width * index as f32, strip.top()),
                egui::vec2(width, height),
            );
            if self.tab_button(ui, rect, tab) {
                self.state.tab = tab;
            }
        }
        ui.painter().line_segment(
            [strip.left_bottom(), strip.right_bottom()],
            egui::Stroke::new(theme::HAIRLINE, theme::PALETTE.ink),
        );
    }

    fn tab_button(&self, ui: &mut egui::Ui, rect: egui::Rect, tab: Tab) -> bool {
        let selected = self.state.tab == tab;
        let response = ui.interact(
            rect,
            ui.id().with(("inspector_tab", tab.label())),
            egui::Sense::click(),
        );
        let painter = ui.painter();
        let color = if selected {
            painter.rect_filled(rect, 0, theme::PALETTE.selected);
            theme::PALETTE.on_selected
        } else {
            if response.hovered() {
                painter.rect_filled(rect, 0, theme::PALETTE.raised);
            }
            theme::PALETTE.ink_dim
        };
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            tab.label(),
            egui::FontId::monospace(11.0),
            color,
        );
        painter.line_segment(
            [rect.right_top(), rect.right_bottom()],
            egui::Stroke::new(theme::HAIRLINE, theme::PALETTE.ink_faint),
        );
        response.on_hover_text(tab.hint()).clicked()
    }

    fn properties(&mut self, ui: &mut egui::Ui) {
        if let Some(error) = self.view.renderer_error {
            theme::card(ui, |ui| {
                theme::section_title(ui, "renderer error", None);
                widgets::alert(ui, error);
            });
        }
        match self.state.tab {
            Tab::Render => panels::render::show(self, ui),
            Tab::Camera => panels::camera::show(self, ui),
            Tab::Object => panels::object::show(self, ui),
            Tab::Material => panels::material::show(self, ui),
        }
        ui.add_space(12.0);
    }

    /// The status strip over the top of the viewport.
    fn toolbar(&mut self, ui: &mut egui::Ui) {
        egui::Panel::top("spectra_toolbar")
            .frame(theme::toolbar_frame())
            .show_inside(ui, |ui| {
                ui.horizontal(|ui| {
                    let chevron = if self.state.open {
                        "\u{25c0}"
                    } else {
                        "\u{25b6}"
                    };
                    if theme::button(ui, chevron)
                        .on_hover_text("Toggle the inspector")
                        .clicked()
                    {
                        self.state.open = !self.state.open;
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let stats = self.view.stats;
                        self.stat(ui, "UI", format!("{:.2} ms", self.view.ui_ms));
                        self.stat(ui, "WAIT", format!("{:.2} ms", stats.wait_ms));
                        self.stat(ui, "CPU", format!("{:.2} ms", stats.cpu_ms));
                        self.stat(ui, "GPU", format!("{:.2} ms", stats.gpu_ms));
                        self.stat(
                            ui,
                            "SPP",
                            self.view.samples_drawn.map_or_else(
                                || format!("\u{2014}/{}", self.view.settings.target_spp),
                                |drawn| format!("{drawn}/{}", self.view.settings.target_spp),
                            ),
                        );
                        let extent = self.view.viewport_extent;
                        self.stat(
                            ui,
                            "RES",
                            if extent == UVec2::ZERO {
                                "\u{2014}".to_owned()
                            } else {
                                format!("{}\u{00d7}{}", extent.x, extent.y)
                            },
                        );
                    });
                });
            });
    }

    fn stat(&self, ui: &mut egui::Ui, name: &str, value: String) {
        ui.label(
            egui::RichText::new(value)
                .monospace()
                .size(11.0)
                .color(theme::PALETTE.ink),
        );
        ui.label(
            egui::RichText::new(name)
                .monospace()
                .size(11.0)
                .color(theme::PALETTE.ink_dim),
        );
        ui.label(
            egui::RichText::new("\u{2502}")
                .small()
                .color(theme::PALETTE.ink_faint),
        );
    }

    /// The region the renderer draws into. Nothing is painted here; the interaction exists so the
    /// camera controller knows whether the pointer and keyboard belong to it or to the inspector.
    fn viewport(&mut self, ui: &mut egui::Ui) {
        let rect = ui.available_rect_before_wrap();
        let response = ui.interact(
            rect,
            ui.id().with("viewport"),
            egui::Sense::click_and_drag(),
        );
        let pixels = response.rect.size() * ui.ctx().pixels_per_point();
        self.state.viewport_extent = UVec2::new(
            pixels.x.round().max(1.0) as u32,
            pixels.y.round().max(1.0) as u32,
        );
        self.state.viewport_hovered = response.contains_pointer();
        if response.clicked() || response.has_focus() {
            self.state.viewport_focused = true;
        } else if response.lost_focus() || ui.ctx().input(|input| input.pointer.any_click()) {
            self.state.viewport_focused = false;
        }
    }
}

/// Run `add_contents` inside a fixed sub-rectangle of the dock.
fn region(ui: &mut egui::Ui, rect: egui::Rect, add_contents: impl FnOnce(&mut egui::Ui)) {
    ui.scope_builder(
        egui::UiBuilder::new()
            .max_rect(rect)
            .layout(egui::Layout::top_down(egui::Align::Min)),
        |ui| {
            ui.shrink_clip_rect(rect);
            ui.set_width(rect.width());
            ui.set_height(rect.height());
            add_contents(ui);
        },
    );
}
