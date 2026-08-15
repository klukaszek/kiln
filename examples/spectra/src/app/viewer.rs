//! Windowed application mode.

use std::{collections::HashSet, sync::OnceLock};

use glam::{DMat4, DQuat, DVec3, EulerRot, UVec2, Vec3};
use kiln_app::{Example, FrameCtx, PerformanceStats};
use kiln_rhi::{CommandBuffer, Device, Format};
use winit::event::WindowEvent;

use spectra::base::renderer::{self, PresentRenderer, RenderFrame, Renderer};
use spectra::base::scene::{
    Light, LightKind, MaterialId, Mesh, NodeId, NodeObject, PrincipledBsdf, Scene, Surface,
    build_scene_nodes,
};
use spectra::importers::usd;
use spectra::renderers::raster::RasterRenderer;
use spectra::renderers::spectral::{PathTracer, SceneUpdate, Settings, spectrum, spectrum::Spd};

use super::Result;
use super::config::Config;
use super::controls::CameraController;

static CONFIG: OnceLock<Config> = OnceLock::new();
const INSPECTOR_TAB_ICON_SIZE: f32 = 28.0;
const INSPECTOR_TAB_TOP_GAP: f32 = 8.0;
const INSPECTOR_TAB_STEP: f32 = 32.0;
const SCENE_TREE_SLOT_SIZE: f32 = 20.0;
const SCENE_TREE_ROW_HEIGHT: f32 = 24.0;

pub fn run(config: Config) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let title = format!("Kiln \u{00b7} Spectral \u{2014} {}", config.scene_name());
    let harness = config.harness.clone();
    CONFIG
        .set(config)
        .expect("viewer configured more than once");
    kiln_app::run_with::<App>(&title, [0.02, 0.02, 0.03, 1.0], harness)
}

struct App {
    scene: Scene,
    renderer: ActiveRenderer,
    controls: CameraController,
    settings: Settings,
    viewport_extent: UVec2,
    light_spectrum: Spd,
    scene_object_count: usize,
    mesh_pivots: Vec<DVec3>,
    pending_scene_update: Option<SceneUpdate>,
    settings_dirty: bool,
    renderer_error: Option<String>,
    ui_state: UiState,
}

#[derive(Default)]
struct UiState {
    inspector_open: bool,
    selected_node: Option<NodeId>,
    selected_material: Option<MaterialId>,
    transform_editor: Option<TransformEditorState>,
    style_configured: bool,
    scene_tree: SceneTreeState,
    scene_tree_height: f32,
    active_tab: InspectorTab,
    render_panel_selected: bool,
    render_panel_hovered: bool,
    renderer_update_deferred: bool,
}

#[derive(Clone, Copy)]
struct SceneTreeRow {
    node: NodeId,
    depth: u16,
}

struct SceneTreeState {
    expanded: HashSet<NodeId>,
    visible: Vec<SceneTreeRow>,
    scene_node_count: usize,
    dirty: bool,
}

impl Default for SceneTreeState {
    fn default() -> Self {
        Self {
            expanded: HashSet::new(),
            visible: Vec::new(),
            scene_node_count: 0,
            dirty: true,
        }
    }
}

impl SceneTreeState {
    fn rebuild_if_dirty(&mut self, scene: &Scene) {
        if !self.dirty && self.scene_node_count == scene.nodes.len() {
            return;
        }
        let initialize_expansion = self.scene_node_count == 0 && self.visible.is_empty();
        self.scene_node_count = scene.nodes.len();
        self.expanded.retain(|node| {
            scene
                .nodes
                .get(node.0)
                .is_some_and(|node| node.is_group() && !node.children.is_empty())
        });
        if initialize_expansion {
            for node in &scene.nodes {
                if node.is_group() && !node.children.is_empty() {
                    self.expanded.insert(node.id);
                }
            }
        }
        self.visible.clear();
        if let Some(root) = scene.nodes.first() {
            for &child in &root.children {
                append_visible_scene_rows(scene, child, 0, &self.expanded, &mut self.visible);
            }
        }
        self.dirty = false;
    }
}

fn append_visible_scene_rows(
    scene: &Scene,
    node_id: NodeId,
    depth: u16,
    expanded: &HashSet<NodeId>,
    visible: &mut Vec<SceneTreeRow>,
) {
    let Some(node) = scene.nodes.get(node_id.0) else {
        return;
    };
    visible.push(SceneTreeRow {
        node: node_id,
        depth,
    });
    if expanded.contains(&node_id) {
        for &child in &node.children {
            append_visible_scene_rows(scene, child, depth.saturating_add(1), expanded, visible);
        }
    }
}

#[derive(Clone, Copy)]
struct TransformEditorState {
    node: NodeId,
    pivot: DVec3,
    source: DMat4,
    translation: DVec3,
    rotation_degrees: DVec3,
    scale: DVec3,
}

impl TransformEditorState {
    fn from_matrix(node: NodeId, source: DMat4, pivot: DVec3, previous: Option<Self>) -> Self {
        let (scale, rotation, _) = source.to_scale_rotation_translation();
        let translation = source.transform_point3(pivot);
        let raw_rotation = rotation.to_euler(EulerRot::XYZ);
        let raw_rotation = DVec3::new(
            raw_rotation.0.to_degrees(),
            raw_rotation.1.to_degrees(),
            raw_rotation.2.to_degrees(),
        );
        let rotation_degrees = previous
            .filter(|previous| previous.node == node)
            .map(|previous| unwrap_rotation(raw_rotation, previous.rotation_degrees))
            .unwrap_or(raw_rotation);
        Self {
            node,
            pivot,
            source,
            translation,
            rotation_degrees,
            scale,
        }
    }

    fn matrix(self) -> DMat4 {
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
}

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

fn matrix_nearly_equal(a: DMat4, b: DMat4) -> bool {
    a.to_cols_array()
        .into_iter()
        .zip(b.to_cols_array())
        .all(|(a, b)| (a - b).abs() <= 1e-10)
}

fn mesh_pivot(mesh: &Mesh) -> DVec3 {
    let Some(first) = mesh.vertices.first() else {
        return DVec3::ZERO;
    };
    let first = first.position.as_dvec3();
    let (min, max) = mesh
        .vertices
        .iter()
        .skip(1)
        .fold((first, first), |(min, max), vertex| {
            let position = vertex.position.as_dvec3();
            (min.min(position), max.max(position))
        });
    (min + max) * 0.5
}

enum ActiveRenderer {
    Spectral(Box<PathTracer>),
    Raster(Box<RasterRenderer>),
}

impl ActiveRenderer {
    fn update_scene(
        &mut self,
        device: &Device,
        scene: &Scene,
        light_spectrum: &Spd,
        update: &SceneUpdate,
        frame_slot: usize,
    ) -> renderer::Result<()> {
        match self {
            Self::Spectral(renderer) => {
                renderer.update_scene(device, scene, light_spectrum, update, frame_slot)
            }
            Self::Raster(renderer) => renderer.update_scene(device, scene),
        }
    }

    fn update_settings(&mut self, settings: Settings) -> renderer::Result<()> {
        match self {
            Self::Spectral(renderer) => renderer.update_settings(settings),
            Self::Raster(_) => Ok(()),
        }
    }

    fn samples_drawn(&self) -> Option<u32> {
        match self {
            Self::Spectral(renderer) => Some(renderer.sample_count()),
            Self::Raster(_) => None,
        }
    }

    fn is_complete(&self) -> bool {
        match self {
            Self::Spectral(renderer) => renderer.is_complete(),
            Self::Raster(_) => true,
        }
    }

    fn destroy(self, device: &Device) {
        match self {
            Self::Spectral(renderer) => Renderer::destroy(renderer, device),
            Self::Raster(renderer) => Renderer::destroy(renderer, device),
        }
    }
}

impl PresentRenderer for ActiveRenderer {
    fn depth_format(&self) -> Option<Format> {
        match self {
            Self::Spectral(renderer) => renderer.depth_format(),
            Self::Raster(renderer) => renderer.depth_format(),
        }
    }

    fn encode_present(&mut self, frame: &RenderFrame<'_>, commands: &mut CommandBuffer) {
        match self {
            Self::Spectral(renderer) => renderer.encode_present(frame, commands),
            Self::Raster(renderer) => renderer.encode_present(frame, commands),
        }
    }
}

impl Renderer for ActiveRenderer {
    fn encode(
        &mut self,
        frame: &RenderFrame<'_>,
        commands: &mut CommandBuffer,
        camera: &spectra::base::scene::Camera,
    ) -> renderer::Result<()> {
        match self {
            Self::Spectral(renderer) => renderer.encode(frame, commands, camera),
            Self::Raster(renderer) => renderer.encode(frame, commands, camera),
        }
    }

    fn destroy(self: Box<Self>, device: &Device) {
        (*self).destroy(device);
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum InspectorTab {
    Render,
    Camera,
    #[default]
    Object,
    Materials,
}

impl App {
    fn try_new(device: &Device, color_format: Format) -> Result<Self> {
        let config = CONFIG.get().expect("viewer config not installed");
        let asset = config.scene_path()?;
        let mut scene = usd::load(&asset)?;
        let default_light_spectrum = config.light_spectrum_name().to_owned();
        for light in &mut scene.lights {
            light.spectrum = default_light_spectrum.clone();
        }
        let light_spectrum = config.light_spectrum()?;
        let scene_object_count = scene.nodes.iter().filter(|node| !node.is_group()).count();
        let mesh_pivots = scene.geometry.meshes.iter().map(mesh_pivot).collect();
        let settings = Settings {
            target_spp: config.spp,
            passes_per_frame: config.passes_per_frame,
            render_scale: config.render_scale,
            pixel_stride: config.pixel_stride,
        };
        let renderer = create_renderer(device, color_format, &scene, &light_spectrum, settings)?;

        let controls = CameraController::new(scene.camera.world, scene.up);
        let selected_node = scene
            .nodes
            .iter()
            .find(|node| !node.is_group())
            .map(|node| node.id);
        Ok(Self {
            scene,
            renderer,
            controls,
            settings,
            viewport_extent: UVec2::ZERO,
            light_spectrum,
            scene_object_count,
            mesh_pivots,
            pending_scene_update: None,
            settings_dirty: false,
            renderer_error: None,
            ui_state: UiState {
                inspector_open: true,
                selected_node,
                selected_material: Some(MaterialId::DEFAULT),
                scene_tree_height: 240.0,
                ..Default::default()
            },
        })
    }
}

fn create_renderer(
    device: &Device,
    color_format: Format,
    scene: &Scene,
    light_spectrum: &Spd,
    settings: Settings,
) -> Result<ActiveRenderer> {
    match PathTracer::new(device, color_format, scene, light_spectrum, settings) {
        Ok(renderer) => Ok(ActiveRenderer::Spectral(Box::new(renderer))),
        Err(error) => {
            eprintln!("spectral path tracer unavailable; using raster renderer: {error:#}");
            Ok(ActiveRenderer::Raster(Box::new(RasterRenderer::new(
                device,
                color_format,
                scene,
            )?)))
        }
    }
}

impl App {
    fn draw_scene_inspector(&mut self, root: &mut egui::Ui) {
        if !self.ui_state.inspector_open {
            return;
        }
        egui::Panel::left("spectra_inspector")
            .default_size(340.0)
            .resizable(true)
            .frame(inspector_frame())
            .show_inside(root, |ui| {
                // Lock the frame content to the rectangle allocated by SidePanel. Child scopes
                // must not grow this Ui from long inspector labels or scroll content, otherwise
                // the panel resize handle and tab rail drift apart.
                let panel_rect = ui.available_rect_before_wrap();
                ui.set_width(panel_rect.width());
                ui.set_height(panel_rect.height());
                let rail_width = INSPECTOR_TAB_ICON_SIZE + ui.spacing().item_spacing.x;
                let rail_left = panel_rect.right() - rail_width;
                let properties_right =
                    (rail_left - ui.spacing().item_spacing.x).max(panel_rect.left());

                let splitter_height = 6.0;
                let min_scene_height = 120.0;
                let min_properties_height = 140.0;
                let max_scene_height =
                    (panel_rect.height() - splitter_height - min_properties_height)
                        .max(min_scene_height);
                let scene_height = self
                    .ui_state
                    .scene_tree_height
                    .max(min_scene_height)
                    .min(max_scene_height);
                self.ui_state.scene_tree_height = scene_height;

                let scene_rect = egui::Rect::from_min_size(
                    panel_rect.min,
                    egui::vec2(panel_rect.width(), scene_height),
                );
                let splitter_rect = egui::Rect::from_min_size(
                    egui::pos2(panel_rect.left(), scene_rect.bottom()),
                    egui::vec2(panel_rect.width(), splitter_height),
                );
                let properties_rect = egui::Rect::from_min_max(
                    egui::pos2(panel_rect.left(), splitter_rect.bottom()),
                    egui::pos2(properties_right, panel_rect.bottom()),
                );

                ui.scope_builder(
                    egui::UiBuilder::new()
                        .max_rect(scene_rect)
                        .layout(egui::Layout::top_down(egui::Align::Min)),
                    |ui| {
                        ui.shrink_clip_rect(scene_rect);
                        ui.set_height(scene_rect.height());
                        ui.set_width(scene_rect.width());
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("SCENE").strong());
                            ui.label(
                                egui::RichText::new(format!(
                                    "{} objects  |  {} materials",
                                    self.scene_object_count,
                                    self.scene.materials.len()
                                ))
                                .small()
                                .weak(),
                            );
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui.button("+").on_hover_text("Add point light").clicked() {
                                        self.add_point_light();
                                    }
                                },
                            );
                        });
                        if let Some(error) = &self.renderer_error {
                            ui.colored_label(
                                egui::Color32::from_rgb(255, 150, 100),
                                egui::RichText::new(format!("Renderer: {error}"))
                                    .small()
                                    .monospace(),
                            );
                        }
                        self.scene_tree_ui(ui);
                    },
                );

                let splitter_response = ui.interact(
                    splitter_rect,
                    ui.id().with("scene_tree_splitter"),
                    egui::Sense::drag(),
                );
                if splitter_response.dragged() {
                    self.ui_state.scene_tree_height = (scene_height
                        + splitter_response.drag_delta().y)
                        .clamp(min_scene_height, max_scene_height);
                }
                ui.painter()
                    .rect_filled(splitter_rect, 0.0, egui::Color32::from_rgb(36, 38, 44));
                ui.painter().line_segment(
                    [
                        egui::pos2(splitter_rect.left(), splitter_rect.center().y),
                        egui::pos2(splitter_rect.right(), splitter_rect.center().y),
                    ],
                    egui::Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color),
                );

                ui.scope_builder(
                    egui::UiBuilder::new()
                        .max_rect(properties_rect)
                        .layout(egui::Layout::top_down(egui::Align::Min)),
                    |ui| {
                        ui.shrink_clip_rect(properties_rect);
                        ui.set_height(properties_rect.height());
                        ui.set_width(properties_rect.width());
                        egui::ScrollArea::vertical()
                            .id_salt("inspector_properties_scroll")
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                ui.set_width(properties_rect.width());
                                self.inspector_properties_ui(ui);
                            });
                    },
                );

                let rail_rect = egui::Rect::from_min_max(
                    egui::pos2(rail_left, splitter_rect.bottom()),
                    egui::pos2(panel_rect.right(), panel_rect.bottom()),
                );
                ui.scope_builder(
                    egui::UiBuilder::new()
                        .max_rect(rail_rect)
                        .layout(egui::Layout::top_down(egui::Align::Min)),
                    |ui| {
                        ui.shrink_clip_rect(rail_rect);
                        ui.set_width(rail_rect.width());
                        ui.set_height(rail_rect.height());
                        self.inspector_tab_rail(ui, rail_rect);
                    },
                );
            });
    }

    fn inspector_properties_ui(&mut self, ui: &mut egui::Ui) {
        ui.add_space(8.0);
        match self.ui_state.active_tab {
            InspectorTab::Render => {
                inspector_tab_title(ui, "RENDER");
                self.render_settings_ui(ui);
            }
            InspectorTab::Camera => {
                inspector_tab_title(ui, "CAMERA");
                self.camera_editor_ui(ui);
            }
            InspectorTab::Object => {
                inspector_tab_title(ui, "OBJECT");
                self.selected_object_ui(ui);
            }
            InspectorTab::Materials => {
                inspector_tab_title(ui, "MATERIALS");
                self.materials_tab_ui(ui);
            }
        }
    }

    fn inspector_tab_rail(&mut self, ui: &mut egui::Ui, rail_rect: egui::Rect) {
        ui.painter()
            .rect_filled(rail_rect, 0.0, egui::Color32::from_rgb(24, 26, 31));
        ui.painter().line_segment(
            [
                egui::pos2(rail_rect.left(), rail_rect.top()),
                egui::pos2(rail_rect.left(), rail_rect.bottom()),
            ],
            egui::Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color),
        );

        let icon_size = egui::Vec2::splat(INSPECTOR_TAB_ICON_SIZE);
        let icon_x = rail_rect.right() - icon_size.x;
        let tab_rect = |index: f32| {
            egui::Rect::from_min_size(
                egui::pos2(
                    icon_x,
                    rail_rect.top() + INSPECTOR_TAB_TOP_GAP + index * INSPECTOR_TAB_STEP,
                ),
                icon_size,
            )
        };
        if inspector_tab_button(
            ui,
            tab_rect(0.0),
            InspectorTab::Render,
            self.ui_state.active_tab == InspectorTab::Render,
            "Render settings",
        ) {
            self.ui_state.active_tab = InspectorTab::Render;
        }
        if inspector_tab_button(
            ui,
            tab_rect(1.0),
            InspectorTab::Camera,
            self.ui_state.active_tab == InspectorTab::Camera,
            "Camera",
        ) {
            self.ui_state.active_tab = InspectorTab::Camera;
        }
        if inspector_tab_button(
            ui,
            tab_rect(2.0),
            InspectorTab::Object,
            self.ui_state.active_tab == InspectorTab::Object,
            "Selected object",
        ) {
            self.ui_state.active_tab = InspectorTab::Object;
        }
        if inspector_tab_button(
            ui,
            tab_rect(3.0),
            InspectorTab::Materials,
            self.ui_state.active_tab == InspectorTab::Materials,
            "Materials",
        ) {
            self.ui_state.active_tab = InspectorTab::Materials;
        }
    }

    fn materials_tab_ui(&mut self, ui: &mut egui::Ui) {
        let rows = self
            .scene
            .materials
            .iter()
            .enumerate()
            .map(|(index, material)| {
                let id = MaterialId(index);
                (
                    id,
                    material.name.clone(),
                    self.scene.material_usage(id),
                    material.is_emissive(),
                )
            })
            .collect::<Vec<_>>();

        let mut selected = self.ui_state.selected_material;
        let mut clicked = None;
        let mut create = false;

        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(format!(
                    "{} material{}",
                    rows.len(),
                    if rows.len() == 1 { "" } else { "s" }
                ))
                .weak(),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                create = ui.button("+ New").clicked();
            });
        });

        inspector_card(ui, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("material_browser_rows")
                .auto_shrink([false, false])
                .max_height(220.0)
                .show(ui, |ui| {
                    for (material_id, name, usage, emissive) in &rows {
                        let selected_row = selected == Some(*material_id);
                        let (row_rect, response) = ui.allocate_exact_size(
                            egui::vec2(ui.available_width(), 32.0),
                            egui::Sense::click(),
                        );
                        let visuals = ui.style().interact_selectable(&response, selected_row);
                        if response.hovered() || selected_row {
                            ui.painter().rect_filled(row_rect, 3.0, visuals.bg_fill);
                        }
                        let accent = if *emissive {
                            egui::Color32::from_rgb(255, 186, 72)
                        } else {
                            visuals.fg_stroke.color
                        };
                        ui.painter().circle_filled(
                            egui::pos2(row_rect.left() + 9.0, row_rect.center().y),
                            3.0,
                            accent,
                        );
                        let title = ui.painter().layout_no_wrap(
                            name.clone(),
                            egui::TextStyle::Body.resolve(ui.style()),
                            visuals.fg_stroke.color,
                        );
                        let subtitle = ui.painter().layout_no_wrap(
                            format!(
                                "{}  ·  {} triangle{}",
                                if material_id.0 == MaterialId::DEFAULT.0 {
                                    "DEFAULT"
                                } else if *emissive {
                                    "EMISSIVE"
                                } else {
                                    "SURFACE"
                                },
                                usage,
                                if *usage == 1 { "" } else { "s" }
                            ),
                            egui::TextStyle::Small.resolve(ui.style()),
                            ui.visuals().weak_text_color(),
                        );
                        ui.painter().galley(
                            egui::pos2(row_rect.left() + 18.0, row_rect.top() + 3.0),
                            title,
                            visuals.fg_stroke.color,
                        );
                        ui.painter().galley(
                            egui::pos2(row_rect.left() + 18.0, row_rect.top() + 17.0),
                            subtitle,
                            ui.visuals().weak_text_color(),
                        );
                        if response.clicked() {
                            clicked = Some(*material_id);
                        }
                    }
                    if rows.is_empty() {
                        ui.label(egui::RichText::new("No materials in this scene.").weak());
                    }
                });
        });

        if let Some(material_id) = clicked {
            selected = Some(material_id);
        }
        if create {
            let material_id = self.scene.add_material();
            selected = Some(material_id);
            self.mark_scene_update(SceneUpdate::material(material_id.0));
        }

        let selected_material = selected.and_then(|material_id| {
            self.scene
                .materials
                .get(material_id.0)
                .map(|material| (material_id, material.clone()))
        });
        let Some((material_id, material)) = selected_material else {
            self.ui_state.selected_material = selected;
            return;
        };
        self.ui_state.selected_material = Some(material_id);

        ui.add_space(4.0);
        let mut name = material.name.clone();
        let mut name_changed = false;
        let mut delete = false;
        inspector_card(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("IDENTITY").small().strong());
                if material_id != MaterialId::DEFAULT {
                    delete = ui
                        .small_button("Delete")
                        .on_hover_text("Delete this material and reset its geometry bindings")
                        .clicked();
                } else {
                    ui.add_enabled(false, egui::Button::new("Delete"));
                }
            });
            property_grid(ui, ("material_identity", material_id), |ui| {
                ui.label(egui::RichText::new("Name").weak());
                name_changed |= ui.text_edit_singleline(&mut name).changed();
                ui.end_row();
                property_row(ui, "Triangles", self.scene.material_usage(material_id));
                property_row(
                    ui,
                    "Binding",
                    if material_id == MaterialId::DEFAULT {
                        "Protected default"
                    } else {
                        "Editable"
                    },
                );
            });
        });

        if name_changed && !name.trim().is_empty() {
            self.scene.rename_material(material_id, name);
        }
        if delete {
            self.delete_material(material_id);
            return;
        }

        self.material_editor_ui(ui, material_id.0, self.scene.material_usage(material_id));
    }

    fn delete_material(&mut self, material: MaterialId) {
        if self.scene.remove_material(material).is_some() {
            self.ui_state.selected_material = Some(MaterialId::DEFAULT);
            self.ui_state.active_tab = InspectorTab::Materials;
            self.mark_scene_update(SceneUpdate::material(MaterialId::DEFAULT.0));
        }
    }

    fn transform_editor_state(
        &mut self,
        node: NodeId,
        source: DMat4,
        pivot: DVec3,
    ) -> TransformEditorState {
        let previous = self.ui_state.transform_editor;
        let state = previous
            .filter(|previous| {
                previous.node == node
                    && previous.pivot == pivot
                    && matrix_nearly_equal(previous.source, source)
            })
            .unwrap_or_else(|| TransformEditorState::from_matrix(node, source, pivot, previous));
        self.ui_state.transform_editor = Some(state);
        state
    }

    fn scene_tree_ui(&mut self, ui: &mut egui::Ui) {
        self.ui_state.scene_tree.rebuild_if_dirty(&self.scene);
        let row_height = SCENE_TREE_ROW_HEIGHT;
        let visible = &self.ui_state.scene_tree.visible;
        let expanded = &self.ui_state.scene_tree.expanded;
        let mut selected = self.ui_state.selected_node;
        let mut active_tab = self.ui_state.active_tab;
        let mut toggle = None;
        ui.scope(|ui| {
            ui.spacing_mut().item_spacing.y = 0.0;
            egui::ScrollArea::vertical()
                .id_salt("scene_tree_rows")
                .auto_shrink([false, false])
                .show_rows(ui, row_height, visible.len(), |ui, rows| {
                    for row_index in rows {
                        let row = visible[row_index];
                        let Some(node) = self.scene.nodes.get(row.node.0) else {
                            continue;
                        };
                        let (row_rect, _) = ui.allocate_exact_size(
                            egui::vec2(ui.available_width(), row_height),
                            egui::Sense::hover(),
                        );
                        let selected_row = selected == Some(row.node);
                        ui.push_id(row.node, |ui| {
                            let row_response = ui.interact(
                                row_rect,
                                ui.id().with("scene_row"),
                                egui::Sense::click(),
                            );
                            let row_visuals =
                                ui.style().interact_selectable(&row_response, selected_row);
                            let center_y = row_rect.center().y;
                            let indent = f32::from(row.depth) * 14.0;
                            let expander_rect = egui::Rect::from_center_size(
                                egui::pos2(
                                    row_rect.left() + indent + SCENE_TREE_SLOT_SIZE * 0.5,
                                    center_y,
                                ),
                                egui::vec2(SCENE_TREE_SLOT_SIZE, SCENE_TREE_SLOT_SIZE),
                            );
                            let has_children = !node.children.is_empty();
                            let expander_clicked = if node.is_group() {
                                let response = scene_tree_expander(
                                    ui,
                                    expander_rect,
                                    expanded.contains(&row.node),
                                    has_children,
                                );
                                if has_children && response.clicked() {
                                    toggle = Some(row.node);
                                }
                                response.clicked()
                            } else {
                                false
                            };
                            let icon_rect = egui::Rect::from_center_size(
                                egui::pos2(
                                    expander_rect.right() + ui.spacing().item_spacing.x + 9.0,
                                    center_y,
                                ),
                                egui::vec2(18.0, 18.0),
                            );
                            scene_object_icon(ui, icon_rect, node.object, selected_row);

                            let painter = ui.painter();
                            let font = egui::TextStyle::Monospace.resolve(ui.style());
                            let text_color = row_visuals.fg_stroke.color;
                            let galley =
                                painter.layout_no_wrap(node.name.clone(), font, text_color);
                            let text_pos = egui::pos2(
                                icon_rect.right() + ui.spacing().item_spacing.x,
                                center_y - galley.size().y * 0.5,
                            );
                            if selected_row {
                                painter.rect_filled(
                                    egui::Rect::from_min_size(
                                        text_pos - egui::vec2(4.0, 2.0),
                                        galley.size() + egui::vec2(8.0, 4.0),
                                    ),
                                    2.0,
                                    row_visuals.bg_fill,
                                );
                            }
                            painter.galley(text_pos, galley, text_color);

                            if row_response.clicked() && !expander_clicked {
                                selected = Some(row.node);
                                active_tab = InspectorTab::Object;
                            }
                            row_response.on_hover_text(&node.path);
                        });
                        let divider_y = row_rect.bottom() + ui.spacing().item_spacing.y * 0.5;
                        ui.painter().line_segment(
                            [
                                egui::pos2(row_rect.left(), divider_y),
                                egui::pos2(row_rect.right(), divider_y),
                            ],
                            egui::Stroke::new(
                                1.0,
                                ui.visuals().widgets.noninteractive.bg_stroke.color,
                            ),
                        );
                    }
                });
        });
        if let Some(node) = toggle {
            if !self.ui_state.scene_tree.expanded.remove(&node) {
                self.ui_state.scene_tree.expanded.insert(node);
            }
            self.ui_state.scene_tree.dirty = true;
        }
        self.ui_state.selected_node = selected;
        self.ui_state.active_tab = active_tab;
    }

    fn selected_object_ui(&mut self, ui: &mut egui::Ui) {
        let Some(node_id) = self.ui_state.selected_node else {
            inspector_card(ui, |ui| {
                ui.label(egui::RichText::new("Select an object in the scene tree.").weak());
            });
            return;
        };
        let Some(node) = self.scene.nodes.get(node_id.0) else {
            self.ui_state.selected_node = None;
            return;
        };

        ui.label(egui::RichText::new(&node.name).strong().size(14.0));
        ui.horizontal(|ui| {
            let kind = match node.object {
                NodeObject::Group => "GROUP",
                NodeObject::Mesh { .. } => "MESH",
                NodeObject::Light { .. } => "LIGHT",
            };
            ui.label(egui::RichText::new(kind).small().weak().monospace());
            ui.label(egui::RichText::new("·").small().weak());
            ui.label(egui::RichText::new(&node.path).small().weak().monospace());
        });
        ui.add_space(6.0);
        match node.object {
            NodeObject::Group => {
                inspector_card(ui, |ui| {
                    property_grid(ui, ("group", node_id), |ui| {
                        property_row(ui, "Children", node.children.len());
                        property_row(ui, "Object", "Group");
                    });
                });
            }
            NodeObject::Mesh { instance } => self.mesh_object_ui(ui, node.id, instance),
            NodeObject::Light { light } => self.light_editor_ui(ui, node.id, light),
        }
    }

    fn mesh_object_ui(&mut self, ui: &mut egui::Ui, node_id: NodeId, instance_index: usize) {
        let Some(instance) = self.scene.geometry.instances.get(instance_index) else {
            return;
        };
        let mesh_index = instance.mesh.0;
        let material_usages = self
            .scene
            .geometry
            .meshes
            .get(mesh_index)
            .map(|mesh| {
                let mut usages = Vec::new();
                for primitive in &mesh.primitives {
                    let triangles = primitive.index_count / 3;
                    if let Some((_, usage)) = usages
                        .iter_mut()
                        .find(|(material, _)| *material == primitive.material.0)
                    {
                        *usage += triangles;
                    } else {
                        usages.push((primitive.material.0, triangles));
                    }
                }
                usages
            })
            .unwrap_or_default();
        let pivot = self
            .mesh_pivots
            .get(mesh_index)
            .copied()
            .unwrap_or(DVec3::ZERO);
        let mut transform_state = self.transform_editor_state(node_id, instance.transform, pivot);
        let mut transform_changed = false;

        inspector_section(
            ui,
            ("mesh_transform", instance_index),
            "TRANSFORM",
            true,
            |ui| {
                property_grid(ui, ("mesh_transform_grid", instance_index), |ui| {
                    transform_changed |= transform_editor_rows(ui, &mut transform_state);
                });
            },
        );

        if transform_changed {
            let transform = transform_state.matrix();
            if let Some(instance) = self.scene.geometry.instances.get_mut(instance_index) {
                instance.transform = transform;
            }
            transform_state.source = transform;
            self.ui_state.transform_editor = Some(transform_state);
            self.mark_scene_update(SceneUpdate::instance_transform(instance_index));
            self.defer_renderer_update(ui);
        }

        let Some(mesh) = self.scene.geometry.meshes.get(mesh_index) else {
            return;
        };
        inspector_section(ui, ("mesh_data", instance_index), "MESH", true, |ui| {
            property_grid(ui, ("mesh_summary", instance_index), |ui| {
                property_row(ui, "Vertices", mesh.vertices.len());
                property_row(ui, "Triangles", mesh.indices.len() / 3);
                property_row(ui, "Primitives", mesh.primitives.len());
                property_row(ui, "Emissive regions", mesh.emissive_components.len());
            });
            for (component_index, component) in mesh.emissive_components.iter().enumerate() {
                let material_name = self
                    .scene
                    .materials
                    .get(component.material.0)
                    .map(|material| material.name.clone())
                    .unwrap_or_else(|| format!("Invalid material {:02}", component.material.0));
                egui::CollapsingHeader::new(format!(
                    "EMISSIVE REGION {:02}  |  {} triangles  |  {}",
                    component_index,
                    component.triangles.len(),
                    material_name
                ))
                .id_salt(("emissive_region", instance_index, component_index))
                .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new(
                            "Emission is a property of this mesh material, not a separate light.",
                        )
                        .small()
                        .weak(),
                    );
                });
            }
        });

        inspector_section(
            ui,
            ("mesh_materials", instance_index),
            "MATERIALS",
            true,
            |ui| {
                if material_usages.is_empty() {
                    ui.label(egui::RichText::new("No material bindings.").weak());
                }
                for (material_index, primitive_count) in &material_usages {
                    self.material_editor_ui(ui, *material_index, *primitive_count);
                }
            },
        );
    }

    fn material_editor_ui(
        &mut self,
        ui: &mut egui::Ui,
        material_index: usize,
        primitive_count: usize,
    ) {
        let Some(material) = self.scene.materials.get(material_index).cloned() else {
            inspector_card(ui, |ui| {
                ui.label(format!("Invalid material #{}", material_index));
            });
            return;
        };
        let material_name = material.name.clone();
        let mut edited = material;
        let mut changed = false;
        let mut inspect = false;

        egui::Frame::group(ui.style())
            .fill(surface_fill())
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(&material_name).strong());
                    ui.label(
                        egui::RichText::new(format!("{} triangle(s)", primitive_count))
                            .small()
                            .weak(),
                    );
                    if ui
                        .small_button("Open")
                        .on_hover_text("Open this material in the Materials tab")
                        .clicked()
                    {
                        inspect = true;
                    }
                });
                ui.add_space(4.0);
                property_grid(ui, ("material_editor", material_index), |ui| {
                    match &mut edited.surface {
                        Surface::Principled(bsdf) => {
                            changed |=
                                property_vec3_row(ui, "Base color", &mut bsdf.base_color, 0.01);
                            changed |= property_scalar_row(
                                ui,
                                "Roughness",
                                &mut bsdf.roughness,
                                0.0..=1.0,
                                0.01,
                            );
                            changed |= property_scalar_row(
                                ui,
                                "Metallic",
                                &mut bsdf.metallic,
                                0.0..=1.0,
                                0.01,
                            );
                            changed |=
                                property_scalar_row(ui, "IOR", &mut bsdf.ior, 1.0..=4.0, 0.01);
                        }
                        Surface::Diffuse { albedo } => {
                            changed |= property_vec3_row(ui, "Albedo", albedo, 0.01);
                        }
                        Surface::Dielectric { ior, roughness } => {
                            changed |= property_scalar_row(ui, "IOR", ior, 1.0..=4.0, 0.01);
                            changed |=
                                property_scalar_row(ui, "Roughness", roughness, 0.0..=1.0, 0.01);
                        }
                        Surface::Conductor { eta, k, roughness } => {
                            changed |= property_vec3_row(ui, "Eta", eta, 0.01);
                            changed |= property_vec3_row(ui, "K", k, 0.01);
                            changed |=
                                property_scalar_row(ui, "Roughness", roughness, 0.0..=1.0, 0.01);
                        }
                    }
                    changed |= property_vec3_row(ui, "Emission", &mut edited.emission.color, 0.05);
                });

                if let Surface::Principled(PrincipledBsdf {
                    base_color_texture: Some(texture_id),
                    ..
                }) = edited.surface
                {
                    ui.add_space(6.0);
                    show_texture(ui, &self.scene, texture_id);
                }
            });

        // The imported surface models are currently immutable variants in the scene IR; the
        // controls above edit their scalar/vector payloads. Queue one renderer transaction for the
        // whole material so the progressive film resets once, after a drag completes.
        if changed {
            if let Some(target) = self.scene.materials.get_mut(material_index) {
                *target = edited;
                self.scene.refresh_emissive_components();
                self.mark_scene_update(SceneUpdate::material(material_index));
                self.defer_renderer_update(ui);
            }
        }
        if inspect {
            self.ui_state.selected_material = Some(MaterialId(material_index));
            self.ui_state.active_tab = InspectorTab::Materials;
        }
    }
}

fn scene_tree_expander(
    ui: &mut egui::Ui,
    rect: egui::Rect,
    expanded: bool,
    has_children: bool,
) -> egui::Response {
    let response = ui.interact(
        rect,
        ui.id().with("scene_expander"),
        if has_children {
            egui::Sense::click()
        } else {
            egui::Sense::hover()
        },
    );
    let color = if has_children {
        ui.visuals().text_color()
    } else {
        ui.visuals().weak_text_color()
    };
    let center = rect.center();
    let points = if expanded {
        vec![
            center + egui::vec2(-3.5, -1.5),
            center + egui::vec2(3.5, -1.5),
            center + egui::vec2(0.0, 3.0),
        ]
    } else {
        vec![
            center + egui::vec2(-1.5, -3.5),
            center + egui::vec2(3.0, 0.0),
            center + egui::vec2(-1.5, 3.5),
        ]
    };
    ui.painter().add(egui::Shape::convex_polygon(
        points,
        color,
        egui::Stroke::NONE,
    ));
    response
}

fn scene_object_icon(ui: &egui::Ui, rect: egui::Rect, object: NodeObject, selected: bool) {
    let color = if selected {
        ui.visuals().selection.stroke.color
    } else {
        ui.visuals().weak_text_color()
    };
    let stroke = egui::Stroke::new(1.2, color);
    let center = rect.center();
    let painter = ui.painter();
    match object {
        NodeObject::Mesh { .. } => {
            let top = center + egui::vec2(0.0, -6.0);
            let left = center + egui::vec2(-6.0, -2.5);
            let right = center + egui::vec2(6.0, -2.5);
            let bottom = center + egui::vec2(0.0, 6.0);
            let lower_left = center + egui::vec2(-6.0, 2.5);
            let lower_right = center + egui::vec2(6.0, 2.5);
            painter.line_segment([top, left], stroke);
            painter.line_segment([top, right], stroke);
            painter.line_segment([left, lower_left], stroke);
            painter.line_segment([right, lower_right], stroke);
            painter.line_segment([lower_left, bottom], stroke);
            painter.line_segment([lower_right, bottom], stroke);
            painter.line_segment([left, right], stroke);
        }
        NodeObject::Light { .. } => {
            painter.circle_stroke(center, 3.5, stroke);
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
                let direction = egui::vec2(dx, dy).normalized();
                let start = center + direction * 5.0;
                let end = center + direction * 7.0;
                painter.line_segment([start, end], stroke);
            }
        }
        NodeObject::Group => {
            painter.rect_stroke(rect.shrink(4.0), 2.0, stroke, egui::StrokeKind::Inside);
        }
    }
}

fn inspector_tab_button(
    ui: &mut egui::Ui,
    rect: egui::Rect,
    tab: InspectorTab,
    selected: bool,
    tooltip: &str,
) -> bool {
    let response = ui.interact(
        rect,
        ui.id().with(("inspector_tab", tooltip)),
        egui::Sense::click(),
    );
    let visuals = ui.style().interact_selectable(&response, selected);
    let painter = ui.painter();
    let icon_rect = rect.shrink(3.0);

    if response.hovered() || selected {
        painter.rect_filled(icon_rect, 4.0, visuals.bg_fill);
    }
    if selected {
        painter.rect_filled(
            egui::Rect::from_min_max(
                egui::pos2(rect.left(), rect.top() + 6.0),
                egui::pos2(rect.left() + 2.0, rect.bottom() - 6.0),
            ),
            1.0,
            visuals.fg_stroke.color,
        );
    }

    let stroke = egui::Stroke::new(1.4, visuals.fg_stroke.color);
    let center = icon_rect.center();
    match tab {
        InspectorTab::Render => {
            for (offset, knob) in [(-5.0, 3.0), (0.0, -3.0), (5.0, 1.0)] {
                let y = center.y + offset;
                painter.line_segment(
                    [egui::pos2(center.x - 7.0, y), egui::pos2(center.x + 7.0, y)],
                    stroke,
                );
                painter.circle_filled(egui::pos2(center.x + knob, y), 2.0, stroke.color);
            }
        }
        InspectorTab::Camera => {
            let body =
                egui::Rect::from_center_size(center + egui::vec2(1.0, 1.0), egui::vec2(15.0, 10.0));
            painter.rect_stroke(body, 2.0, stroke, egui::StrokeKind::Inside);
            painter.circle_stroke(center + egui::vec2(2.0, 1.0), 3.0, stroke);
            painter.line_segment(
                [
                    egui::pos2(body.left() + 2.0, body.top()),
                    egui::pos2(body.left() + 5.0, body.top() - 3.0),
                ],
                stroke,
            );
        }
        InspectorTab::Object => {
            let top = egui::pos2(center.x, center.y - 7.0);
            let left = egui::pos2(center.x - 7.0, center.y - 3.0);
            let right = egui::pos2(center.x + 7.0, center.y - 3.0);
            let bottom = egui::pos2(center.x, center.y + 8.0);
            let lower_left = egui::pos2(center.x - 7.0, center.y + 4.0);
            let lower_right = egui::pos2(center.x + 7.0, center.y + 4.0);
            painter.line_segment([top, left], stroke);
            painter.line_segment([top, right], stroke);
            painter.line_segment([left, lower_left], stroke);
            painter.line_segment([right, lower_right], stroke);
            painter.line_segment([lower_left, bottom], stroke);
            painter.line_segment([lower_right, bottom], stroke);
            painter.line_segment([left, right], stroke);
        }
        InspectorTab::Materials => {
            for (offset, width) in [(-5.0, 12.0), (0.0, 16.0), (5.0, 10.0)] {
                let bar = egui::Rect::from_center_size(
                    center + egui::vec2(0.0, offset),
                    egui::vec2(width, 3.0),
                );
                painter.rect_stroke(bar, 1.0, stroke, egui::StrokeKind::Inside);
            }
        }
    }

    response.on_hover_text(tooltip).clicked()
}

impl App {
    fn render_settings_ui(&mut self, ui: &mut egui::Ui) {
        property_grid(ui, "render_settings_editor", |ui| {
            ui.label(egui::RichText::new("Samples / pixel (SPP)").weak());
            let mut value = self.settings.target_spp;
            if ui
                .add(
                    egui::DragValue::new(&mut value)
                        .range(1..=1_000_000)
                        .speed(1),
                )
                .changed()
            {
                self.settings.target_spp = value;
                self.settings_dirty = true;
                self.defer_renderer_update(ui);
            }
            ui.end_row();

            ui.label(egui::RichText::new("Samples / frame (SPF)").weak());
            let mut value = self.settings.passes_per_frame;
            if ui
                .add(egui::DragValue::new(&mut value).range(1..=1024).speed(1))
                .changed()
            {
                self.settings.passes_per_frame = value;
                self.settings_dirty = true;
                self.defer_renderer_update(ui);
            }
            ui.end_row();

            ui.label(egui::RichText::new("Render scale (1 / N)").weak());
            let mut value = self.settings.render_scale;
            if ui
                .add(egui::DragValue::new(&mut value).range(1..=16).speed(1))
                .changed()
            {
                self.settings.render_scale = value;
                self.settings_dirty = true;
                self.defer_renderer_update(ui);
            }
            ui.end_row();

            ui.label(egui::RichText::new("Pixel stride").weak());
            let mut value = self.settings.pixel_stride;
            if ui
                .add(egui::DragValue::new(&mut value).range(1..=16).speed(1))
                .changed()
            {
                self.settings.pixel_stride = value;
                self.settings_dirty = true;
                self.defer_renderer_update(ui);
            }
            ui.end_row();
        });
        ui.add_space(4.0);
        ui.label(
            egui::RichText::new("Changing render settings rebuilds the active renderer.")
                .small()
                .weak(),
        );
    }

    fn add_point_light(&mut self) {
        let light_index = self.scene.lights.len();
        let position = self
            .scene
            .camera
            .world
            .transform_point3(DVec3::new(0.0, 0.0, -2.0));
        self.scene.lights.push(Light::point(
            format!("Light {light_index:02}"),
            DMat4::from_translation(position),
        ));
        self.scene_object_count += 1;
        self.scene.nodes = build_scene_nodes(&self.scene.geometry.instances, &self.scene.lights);
        self.ui_state.scene_tree.dirty = true;
        self.ui_state.selected_node = self
            .scene
            .nodes
            .iter()
            .find(|node| matches!(node.object, NodeObject::Light { light } if light == light_index))
            .map(|node| node.id);
        self.ui_state.active_tab = InspectorTab::Object;
        self.mark_scene_update(SceneUpdate::light(light_index));
    }

    fn light_editor_ui(&mut self, ui: &mut egui::Ui, node_id: NodeId, index: usize) {
        let Some(source) = self.scene.lights.get(index).map(|light| light.transform) else {
            return;
        };
        let mut transform_state = self.transform_editor_state(node_id, source, DVec3::ZERO);
        let mut changed = false;
        let mut light_changed = false;
        let mut edited_transform = None;
        let mut renamed_light = None;
        {
            let light = &mut self.scene.lights[index];
            let mut name = light.name.clone();
            let mut enabled = light.enabled;
            let original_intensity = light.intensity;
            let mut intensity = original_intensity;
            let mut color = light.color.to_array();
            let mut spectrum_name = light.spectrum.clone();
            let mut intensity_exposure = intensity.max(0.001).log2();
            let mut kind = light.kind.clone();
            let mut transform_changed = false;

            property_grid(ui, ("light_properties", index), |ui| {
                ui.label(egui::RichText::new("Name").weak());
                changed |= ui.text_edit_singleline(&mut name).changed();
                ui.end_row();

                ui.label(egui::RichText::new("Type").weak());
                egui::ComboBox::from_id_salt(("light_kind", index))
                    .selected_text(kind.label())
                    .show_ui(ui, |ui| {
                        for candidate in [
                            LightKind::Point,
                            LightKind::Directional { angle_deg: 0.53 },
                            LightKind::Rect {
                                width: 1.0,
                                height: 1.0,
                            },
                            LightKind::Disk { radius: 0.5 },
                            LightKind::Sphere { radius: 0.5 },
                            LightKind::Dome,
                        ] {
                            let label = candidate.label();
                            changed |= ui.selectable_value(&mut kind, candidate, label).changed();
                        }
                    });
                ui.end_row();

                ui.label(egui::RichText::new("Enabled").weak());
                changed |= ui.checkbox(&mut enabled, "").changed();
                ui.end_row();

                ui.label(egui::RichText::new("Spectrum").weak());
                egui::ComboBox::from_id_salt(("light_spectrum", index))
                    .selected_text(&spectrum_name)
                    .show_ui(ui, |ui| {
                        for &candidate in spectrum::BUILTIN_NAMES {
                            changed |= ui
                                .selectable_value(
                                    &mut spectrum_name,
                                    candidate.to_owned(),
                                    candidate,
                                )
                                .changed();
                        }
                        if !spectrum::BUILTIN_NAMES.contains(&spectrum_name.as_str()) {
                            let current = spectrum_name.clone();
                            ui.separator();
                            ui.selectable_value(&mut spectrum_name, current.clone(), current);
                        }
                    });
                ui.end_row();

                ui.label(egui::RichText::new("Color (linear)").weak());
                ui.horizontal(|ui| {
                    for (channel, value) in ["R", "G", "B"].into_iter().zip(color.iter_mut()) {
                        changed |= ui
                            .add(
                                egui::DragValue::new(value)
                                    .prefix(format!("{channel} "))
                                    .range(0.0..=1_000_000.0)
                                    .clamp_existing_to_range(false)
                                    .speed(0.01)
                                    .fixed_decimals(3),
                            )
                            .changed();
                    }
                    ui.colored_label(color32(Vec3::from_array(color)), "   ");
                });
                ui.end_row();

                ui.label(egui::RichText::new("Intensity").weak());
                ui.horizontal(|ui| {
                    let slider_changed = ui
                        .add(
                            egui::Slider::new(&mut intensity_exposure, -12.0..=20.0)
                                .clamping(egui::SliderClamping::Edits)
                                .text("stops")
                                .show_value(false),
                        )
                        .changed();
                    if slider_changed {
                        intensity = 2.0_f32.powf(intensity_exposure).clamp(0.0, 1_000_000.0);
                        changed = true;
                    }
                    if ui
                        .add(
                            egui::DragValue::new(&mut intensity)
                                .range(0.0..=1_000_000.0)
                                .clamp_existing_to_range(false)
                                .speed(1.0)
                                .fixed_decimals(2),
                        )
                        .changed()
                    {
                        intensity_exposure = intensity.max(0.001).log2();
                        changed = true;
                    }
                });
                ui.end_row();

                transform_changed |= transform_editor_rows(ui, &mut transform_state);

                match &mut kind {
                    LightKind::Directional { angle_deg } => {
                        changed |=
                            property_scalar_row(ui, "Angular size", angle_deg, 0.0..=180.0, 0.01);
                    }
                    LightKind::Rect { width, height } => {
                        changed |= property_scalar_row(ui, "Width", width, 0.001..=100_000.0, 0.01);
                        changed |=
                            property_scalar_row(ui, "Height", height, 0.001..=100_000.0, 0.01);
                    }
                    LightKind::Disk { radius } | LightKind::Sphere { radius } => {
                        changed |=
                            property_scalar_row(ui, "Radius", radius, 0.001..=100_000.0, 0.01);
                    }
                    _ => {}
                }
            });

            let transform = transform_state.matrix();
            let color = Vec3::from_array(color);
            let edited = changed
                || transform_changed
                || name != light.name
                || kind != light.kind
                || enabled != light.enabled
                || intensity != original_intensity
                || spectrum_name != light.spectrum
                || color != light.color;
            if edited {
                light.name = name;
                light.kind = kind;
                light.enabled = enabled;
                light.intensity = intensity;
                light.spectrum = spectrum_name;
                light.color = color;
                if transform_changed {
                    light.transform = transform;
                    edited_transform = Some(transform);
                }
                renamed_light = Some(light.name.clone());
                light_changed = true;
            }
        }

        if let Some(transform) = edited_transform {
            transform_state.source = transform;
            self.ui_state.transform_editor = Some(transform_state);
        }
        if let Some(name) = renamed_light {
            if let Some(node) =
                self.scene.nodes.iter_mut().find(
                    |node| matches!(node.object, NodeObject::Light { light } if light == index),
                )
            {
                node.name = name;
            }
        }
        if light_changed {
            self.mark_scene_update(SceneUpdate::light(index));
            self.defer_renderer_update(ui);
        }
    }

    fn defer_renderer_update(&mut self, ui: &egui::Ui) {
        self.ui_state.renderer_update_deferred |= ui.ctx().input(|input| input.pointer.any_down());
    }

    fn mark_scene_update(&mut self, update: SceneUpdate) {
        self.pending_scene_update
            .get_or_insert_with(SceneUpdate::default)
            .merge(update);
    }

    fn draw_viewport_toolbar(&mut self, root: &mut egui::Ui, stats: PerformanceStats) {
        egui::Panel::top("spectra_viewport_toolbar")
            .frame(toolbar_frame())
            .show_inside(root, |ui| {
                ui.horizontal(|ui| {
                    let chevron = if self.ui_state.inspector_open {
                        "‹"
                    } else {
                        "›"
                    };
                    if ui
                        .add(
                            egui::Button::new(egui::RichText::new(chevron).size(18.0)).frame(false),
                        )
                        .clicked()
                    {
                        self.ui_state.inspector_open = !self.ui_state.inspector_open;
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(egui::RichText::new("RESOLUTION").small().strong());
                        ui.label(
                            egui::RichText::new(if self.viewport_extent == UVec2::ZERO {
                                "—".to_owned()
                            } else {
                                format!("{} × {}", self.viewport_extent.x, self.viewport_extent.y)
                            })
                            .small(),
                        );
                        ui.separator();
                        ui.label(egui::RichText::new("SAMPLES").small().strong());
                        ui.label(
                            egui::RichText::new(self.renderer.samples_drawn().map_or_else(
                                || format!("—/{}", self.settings.target_spp),
                                |samples| format!("{samples}/{}", self.settings.target_spp),
                            ))
                            .small(),
                        );
                        ui.separator();
                        ui.label(egui::RichText::new("GPU").small().strong());
                        ui.label(egui::RichText::new(format!("{:.2} ms", stats.gpu_ms)).small());
                        ui.separator();
                        ui.label(egui::RichText::new("CPU").small().strong());
                        ui.label(egui::RichText::new(format!("{:.2} ms", stats.cpu_ms)).small());
                        ui.separator();
                        ui.label(egui::RichText::new("WAIT").small().strong());
                        ui.label(egui::RichText::new(format!("{:.2} ms", stats.wait_ms)).small());
                    });
                });
            });
    }

    fn draw_viewport(&mut self, root: &mut egui::Ui) {
        let viewport_rect = root.available_rect_before_wrap();
        let response = root.interact(
            viewport_rect,
            root.id().with("render_panel_capture"),
            egui::Sense::click_and_drag(),
        );
        let viewport_size = response.rect.size() * root.ctx().pixels_per_point();
        self.viewport_extent = UVec2::new(
            viewport_size.x.round().max(1.0) as u32,
            viewport_size.y.round().max(1.0) as u32,
        );
        self.ui_state.render_panel_hovered = response.contains_pointer();
        if response.clicked() || response.has_focus() {
            self.ui_state.render_panel_selected = true;
        } else if response.lost_focus() || root.ctx().input(|input| input.pointer.any_click()) {
            self.ui_state.render_panel_selected = false;
        }
    }
}

impl Example for App {
    fn depth_format(&self) -> Option<Format> {
        self.renderer.depth_format()
    }

    fn new(device: &Device, color_format: Format) -> Self {
        Self::try_new(device, color_format).unwrap_or_else(|error| {
            eprintln!("{error:#}");
            std::process::exit(1);
        })
    }

    fn window_event(&mut self, event: &WindowEvent) {
        if matches!(event, WindowEvent::Focused(false)) {
            self.ui_state.render_panel_selected = false;
        }
        if matches!(
            event,
            WindowEvent::MouseInput {
                state: winit::event::ElementState::Pressed,
                ..
            }
        ) && self.ui_state.render_panel_hovered
        {
            // The press itself selects the viewport, so the first drag after clicking it is
            // immediately usable even though egui builds the next frame after this event.
            self.ui_state.render_panel_selected = true;
        }
        let keyboard_release = matches!(
            event,
            WindowEvent::KeyboardInput {
                event,
                ..
            } if event.state == winit::event::ElementState::Released
        );
        let camera_accepts_event = match event {
            WindowEvent::Focused(false) => true,
            WindowEvent::KeyboardInput { .. } => self.ui_state.render_panel_selected,
            WindowEvent::CursorMoved { .. } | WindowEvent::MouseInput { .. } => {
                (self.ui_state.render_panel_selected && self.ui_state.render_panel_hovered)
                    || self.controls.is_dragging()
            }
            _ => false,
        };
        if camera_accepts_event || keyboard_release {
            self.controls.window_event(event);
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, stats: PerformanceStats) {
        self.ui_state.renderer_update_deferred = false;
        if !self.ui_state.style_configured {
            configure_engine_style(ui.ctx());
            self.ui_state.style_configured = true;
        }
        self.draw_scene_inspector(ui);
        self.draw_viewport_toolbar(ui, stats);
        self.draw_viewport(ui);
    }

    fn pre_render(&mut self, ctx: &FrameCtx, cmd: &mut CommandBuffer) {
        if !self.ui_state.renderer_update_deferred {
            if self.settings_dirty {
                match self.renderer.update_settings(self.settings) {
                    Ok(()) => {
                        self.settings_dirty = false;
                        self.renderer_error = None;
                    }
                    Err(error) => {
                        let message = format!("failed to apply renderer settings: {error:#}");
                        eprintln!("{message}");
                        self.renderer_error = Some(message);
                        self.settings_dirty = true;
                    }
                }
            }
            if let Some(update) = self.pending_scene_update.take() {
                match self.renderer.update_scene(
                    ctx.device,
                    &self.scene,
                    &self.light_spectrum,
                    &update,
                    ctx.slot,
                ) {
                    Ok(()) => self.renderer_error = None,
                    Err(error) => {
                        let message = format!("failed to apply scene update: {error:#}");
                        eprintln!("{message}");
                        self.renderer_error = Some(message);
                        self.pending_scene_update = Some(update);
                    }
                }
            }
        }
        if let Some(world) = self.controls.update() {
            self.scene.camera.world = world;
        }
        let frame = RenderFrame {
            device: ctx.device,
            extent: ctx.extent,
            slot: ctx.slot,
        };
        if let Err(error) = self.renderer.encode(&frame, cmd, &self.scene.camera) {
            let message = format!("renderer pre-render failed: {error:#}");
            eprintln!("{message}");
            self.renderer_error = Some(message);
        }
    }

    fn render(&mut self, ctx: &FrameCtx, cmd: &mut CommandBuffer) {
        let frame = RenderFrame {
            device: ctx.device,
            extent: ctx.extent,
            slot: ctx.slot,
        };
        self.renderer.encode_present(&frame, cmd);
    }

    fn wants_continuous_redraw(&self) -> bool {
        !self.renderer.is_complete()
            || self.controls.needs_continuous_update()
            || self.ui_state.renderer_update_deferred
    }

    fn destroy(self, device: &Device) {
        self.renderer.destroy(device);
    }
}

impl App {
    fn camera_editor_ui(&mut self, ui: &mut egui::Ui) {
        let mut position = self.controls.position().to_array();
        let mut position_changed = false;
        let mut fov_degrees = self.scene.camera.projection.vertical_fov_rad.to_degrees();

        property_grid(ui, "camera_properties", |ui| {
            ui.label(egui::RichText::new("Position").weak());
            ui.horizontal_wrapped(|ui| {
                for (axis, value) in ["X", "Y", "Z"].into_iter().zip(&mut position) {
                    position_changed |= ui
                        .add(
                            egui::DragValue::new(value)
                                .prefix(format!("{axis} "))
                                .speed(0.05)
                                .fixed_decimals(2),
                        )
                        .changed();
                }
            });
            ui.end_row();

            ui.label(egui::RichText::new("Vertical FOV").weak());
            ui.add(
                egui::Slider::new(&mut fov_degrees, 10.0..=170.0)
                    .suffix("\u{00b0}")
                    .show_value(true),
            );
            ui.end_row();

            ui.label("");
            if ui.button("Reset authored camera").clicked() {
                self.controls.request_reset();
            }
            ui.end_row();
        });

        if position_changed {
            self.controls.set_position(DVec3::from_array(position));
        }
        if (fov_degrees - self.scene.camera.projection.vertical_fov_rad.to_degrees()).abs()
            > f32::EPSILON
        {
            self.scene.camera.projection.vertical_fov_rad = fov_degrees.to_radians();
        }

        ui.add_space(4.0);
        ui.label(
            egui::RichText::new("WASD move  |  Q/E vertical  |  Shift boost  |  LMB orbit")
                .small()
                .weak(),
        );
    }
}

fn property_scalar_row(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    speed: f64,
) -> bool {
    ui.label(egui::RichText::new(label).weak());
    let changed = ui
        .add(
            egui::DragValue::new(value)
                .range(range)
                .clamp_existing_to_range(false)
                .speed(speed),
        )
        .changed();
    ui.end_row();
    changed
}

fn property_vec3_row(ui: &mut egui::Ui, label: &str, value: &mut Vec3, speed: f64) -> bool {
    ui.label(egui::RichText::new(label).weak());
    let mut changed = false;
    ui.horizontal_wrapped(|ui| {
        for (axis, component) in [
            ("X", &mut value.x),
            ("Y", &mut value.y),
            ("Z", &mut value.z),
        ] {
            changed |= ui
                .push_id(("property_vec3", label, axis), |ui| {
                    ui.add(
                        egui::DragValue::new(component)
                            .clamp_existing_to_range(false)
                            .speed(speed)
                            .fixed_decimals(3),
                    )
                    .changed()
                })
                .inner;
        }
    });
    ui.end_row();
    changed
}

fn transform_editor_rows(ui: &mut egui::Ui, transform: &mut TransformEditorState) -> bool {
    let mut changed = false;
    ui.label(egui::RichText::new("Position").weak());
    ui.horizontal_wrapped(|ui| {
        for (axis, value) in [
            ("X", &mut transform.translation.x),
            ("Y", &mut transform.translation.y),
            ("Z", &mut transform.translation.z),
        ] {
            changed |= ui
                .push_id(("transform_position", axis), |ui| {
                    ui.add(
                        egui::DragValue::new(value)
                            .prefix(format!("{axis} "))
                            .clamp_existing_to_range(false)
                            .speed(0.05)
                            .fixed_decimals(2),
                    )
                    .changed()
                })
                .inner;
        }
    });
    ui.end_row();

    ui.label(egui::RichText::new("Rotation").weak());
    ui.horizontal_wrapped(|ui| {
        for (axis, value) in [
            ("X", &mut transform.rotation_degrees.x),
            ("Y", &mut transform.rotation_degrees.y),
            ("Z", &mut transform.rotation_degrees.z),
        ] {
            changed |= ui
                .push_id(("transform_rotation", axis), |ui| {
                    ui.add(
                        egui::DragValue::new(value)
                            .prefix(format!("{axis} "))
                            .suffix("\u{00b0}")
                            .clamp_existing_to_range(false)
                            .speed(0.5)
                            .fixed_decimals(1),
                    )
                    .changed()
                })
                .inner;
        }
    });
    ui.end_row();

    ui.label(egui::RichText::new("Scale").weak());
    ui.horizontal_wrapped(|ui| {
        for (axis, value) in [
            ("X", &mut transform.scale.x),
            ("Y", &mut transform.scale.y),
            ("Z", &mut transform.scale.z),
        ] {
            changed |= ui
                .push_id(("transform_scale", axis), |ui| {
                    ui.add(
                        egui::DragValue::new(value)
                            .prefix(format!("{axis} "))
                            .range(0.001..=100_000.0)
                            .clamp_existing_to_range(false)
                            .speed(0.05)
                            .fixed_decimals(2),
                    )
                    .changed()
                })
                .inner;
        }
    });
    ui.end_row();
    changed
}

fn configure_engine_style(ctx: &egui::Context) {
    let mut style = (*ctx.global_style()).clone();
    style.spacing.item_spacing = egui::vec2(6.0, 4.0);
    style.spacing.button_padding = egui::vec2(6.0, 3.0);
    style.spacing.window_margin = egui::Margin::same(8);
    ctx.set_global_style(style);
}

fn inspector_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(egui::Color32::from_rgb(24, 26, 31))
        .inner_margin(egui::Margin::same(8))
}

fn toolbar_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(egui::Color32::from_rgb(30, 32, 38))
        .inner_margin(egui::Margin::symmetric(8, 4))
}

fn surface_fill() -> egui::Color32 {
    egui::Color32::from_rgb(31, 34, 40)
}

fn inspector_card(ui: &mut egui::Ui, add_contents: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::group(ui.style())
        .fill(surface_fill())
        .inner_margin(egui::Margin::same(8))
        .show(ui, add_contents);
    ui.add_space(6.0);
}

fn inspector_tab_title(ui: &mut egui::Ui, title: &str) {
    ui.label(egui::RichText::new(title).strong().size(12.0));
    ui.add_space(2.0);
    ui.separator();
    ui.add_space(4.0);
}

fn inspector_section(
    ui: &mut egui::Ui,
    id: impl std::hash::Hash,
    title: &str,
    default_open: bool,
    add_contents: impl FnOnce(&mut egui::Ui),
) {
    egui::CollapsingHeader::new(egui::RichText::new(title).strong())
        .id_salt(id)
        .default_open(default_open)
        .show(ui, |ui| {
            ui.add_space(2.0);
            add_contents(ui);
        });
    ui.add_space(4.0);
}

fn property_grid(
    ui: &mut egui::Ui,
    id: impl std::hash::Hash,
    add_contents: impl FnOnce(&mut egui::Ui),
) {
    egui::Grid::new(id)
        .num_columns(2)
        .striped(true)
        .spacing([8.0, 4.0])
        .show(ui, add_contents);
}

fn property_row(ui: &mut egui::Ui, label: &str, value: impl ToString) {
    ui.label(egui::RichText::new(label).weak());
    ui.label(egui::RichText::new(value.to_string()).monospace());
    ui.end_row();
}

fn show_texture(ui: &mut egui::Ui, scene: &Scene, texture_id: spectra::base::scene::TextureId) {
    ui.label(egui::RichText::new("BASE COLOR TEXTURE").small().strong());
    property_grid(ui, ("texture", texture_id.0), |ui| {
        let Some(texture) = scene.textures.get(texture_id.0) else {
            property_row(ui, "Texture", format!("Invalid #{}", texture_id.0));
            return;
        };
        let Some(image) = scene.images.get(texture.image.0) else {
            property_row(
                ui,
                "Texture",
                format!("{} - invalid image #{}", texture_id.0, texture.image.0),
            );
            return;
        };
        property_row(
            ui,
            "Image",
            format!("{}  ({} x {})", image.name, image.width, image.height),
        );
        property_row(ui, "Color space", format!("{:?}", image.color_space));
        property_row(ui, "Wrap U", format!("{:?}", texture.wrap_u));
        property_row(ui, "Wrap V", format!("{:?}", texture.wrap_v));
    });
}

fn color32(value: Vec3) -> egui::Color32 {
    let channel = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
    egui::Color32::from_rgb(channel(value.x), channel(value.y), channel(value.z))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn editor_rotation_stays_on_the_previous_euler_branch() {
        assert!((unwrap_angle(-179.0, 179.0) - 181.0).abs() < f64::EPSILON);
        assert!((unwrap_angle(179.0, -179.0) + 181.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mesh_transform_round_trips_around_its_pivot() {
        let node = NodeId(1);
        let pivot = DVec3::new(1.0, 2.0, 0.0);
        let source = DMat4::from_translation(DVec3::new(5.0, -3.0, 2.0));
        let state = TransformEditorState::from_matrix(node, source, pivot, None);
        assert!(matrix_nearly_equal(state.matrix(), source));

        let before = source.transform_point3(pivot);
        let mut rotated = state;
        rotated.rotation_degrees.z = 90.0;
        let after = rotated.matrix().transform_point3(pivot);
        assert!((before - after).length() < 1e-10);
    }

    #[test]
    fn scene_tree_keeps_groups_collapsed_after_rebuilding_rows() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/cornell-box.usda");
        let scene = usd::load(&path).unwrap();
        let group = scene
            .nodes
            .iter()
            .find(|node| node.id != NodeId(0) && node.is_group() && !node.children.is_empty())
            .unwrap();
        let child = group.children[0];

        let mut state = SceneTreeState::default();
        state.rebuild_if_dirty(&scene);
        assert!(state.expanded.contains(&group.id));
        assert!(state.visible.iter().any(|row| row.node == child));

        state.expanded.remove(&group.id);
        state.dirty = true;
        state.rebuild_if_dirty(&scene);
        assert!(!state.expanded.contains(&group.id));
        assert!(!state.visible.iter().any(|row| row.node == child));
    }
}
