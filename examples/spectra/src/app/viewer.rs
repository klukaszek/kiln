//! Windowed application mode.
//!
//! This module owns the live scene, the active renderer, and the harness lifecycle. The inspector
//! is built in [`super::ui`] and returns [`ui::Edit`]s; applying one here is the only place a scene
//! change turns into a renderer transaction, so no UI code has to know how the renderer is
//! invalidated.

use std::sync::OnceLock;

use glam::{DMat4, DVec3, UVec2};
use kiln_app::{Example, FrameCtx, PerformanceStats};
use kiln_rhi::{CommandBuffer, Device, Format};
use winit::event::WindowEvent;

use spectra::base::renderer::{self, PresentRenderer, RenderFrame, Renderer};
use spectra::base::scene::{
    Light, MaterialId, Mesh, NodeObject, Scene, SpectrumSource, build_scene_nodes,
};
use spectra::importers::usd;
use spectra::renderers::raster::RasterRenderer;
use spectra::renderers::spectral::{PathTracer, SceneUpdate, Settings, spectrum::Spd};

use super::Result;
use super::config::Config;
use super::ui::{self, Edit, InspectorState};

static CONFIG: OnceLock<Config> = OnceLock::new();

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
    controls: super::controls::CameraController,
    settings: Settings,
    viewport_extent: UVec2,
    /// The renderer-wide spectrum, which every illuminant's own spectrum is expressed relative to.
    light_spectrum: Spd,
    object_count: usize,
    /// Object-space centre of each mesh, used as the transform editor's pivot.
    mesh_pivots: Vec<DVec3>,
    /// Triangles bound to each material, and the world-space area of its emissive regions. Both are
    /// whole-scene scans, so they are derived once per scene change rather than once per frame.
    material_usage: Vec<usize>,
    emissive_areas: Vec<f32>,
    /// Smoothed inspector build time, which the harness's own CPU figure excludes.
    ui_ms: f64,
    pending_scene_update: Option<SceneUpdate>,
    settings_dirty: bool,
    /// Set while an edit is still being dragged, so the film resets once on release instead of on
    /// every frame of the drag.
    update_deferred: bool,
    renderer_error: Option<String>,
    inspector: InspectorState,
}

impl App {
    fn try_new(device: &Device, color_format: Format) -> Result<Self> {
        let config = CONFIG.get().expect("viewer config not installed");
        let asset = config.scene_path()?;
        let mut scene = usd::load(&asset)?;
        // Imported lights adopt the renderer-wide spectrum, which is the one the CLI selected and
        // the one an unresolved name would fall back to anyway.
        let default_spectrum = SpectrumSource::Named(config.light_spectrum_name().to_owned());
        for light in &mut scene.lights {
            light.illuminant.spectrum = default_spectrum.clone();
        }
        let light_spectrum = config.light_spectrum()?;
        let object_count = scene.nodes.iter().filter(|node| !node.is_group()).count();
        let mesh_pivots = scene.geometry.meshes.iter().map(mesh_pivot).collect();
        let settings = Settings {
            target_spp: config.spp,
            passes_per_frame: config.passes_per_frame,
            render_scale: config.render_scale,
            pixel_stride: config.pixel_stride,
            spectral_capture: false,
        };
        let renderer = create_renderer(device, color_format, &scene, &light_spectrum, settings)?;
        let controls = super::controls::CameraController::new(scene.camera.world, scene.up);
        let selected = scene
            .nodes
            .iter()
            .find(|node| !node.is_group())
            .map(|node| node.id);

        let mut app = Self {
            material_usage: Vec::new(),
            emissive_areas: Vec::new(),
            scene,
            renderer,
            controls,
            settings,
            viewport_extent: UVec2::ZERO,
            light_spectrum,
            object_count,
            mesh_pivots,
            pending_scene_update: None,
            settings_dirty: false,
            update_deferred: false,
            renderer_error: None,
            ui_ms: 0.0,
            inspector: InspectorState::new(selected),
        };
        app.resummarize_materials();
        Ok(app)
    }

    /// Recompute the per-material summaries the inspector displays. Both are whole-scene scans, so
    /// they run when the scene changes instead of on every frame of every drag.
    fn resummarize_materials(&mut self) {
        let scene = &self.scene;
        self.material_usage = vec![0; scene.materials.len()];
        self.emissive_areas = vec![0.0; scene.materials.len()];
        for mesh in &scene.geometry.meshes {
            for primitive in &mesh.primitives {
                if let Some(usage) = self.material_usage.get_mut(primitive.material.0) {
                    *usage += primitive.index_count / 3;
                }
            }
        }
        for instance in &scene.geometry.instances {
            let Some(mesh) = scene.geometry.meshes.get(instance.mesh.0) else {
                continue;
            };
            for component in &mesh.emissive_components {
                let Some(area) = self.emissive_areas.get_mut(component.material.0) else {
                    continue;
                };
                for corners in &component.triangles {
                    let points = corners.map(|index| {
                        instance
                            .transform
                            .transform_point3(mesh.vertices[index as usize].position.as_dvec3())
                    });
                    let cross = (points[1] - points[0]).cross(points[2] - points[0]);
                    *area += (0.5 * cross.length()) as f32;
                }
            }
        }
    }

    /// Apply one inspector edit and queue the renderer transaction it implies.
    fn apply(&mut self, edit: Edit) {
        match edit {
            Edit::Settings(settings) => {
                self.settings = settings;
                self.settings_dirty = true;
            }
            Edit::CameraPosition(position) => self.controls.set_position(position),
            Edit::CameraFov(radians) => self.scene.camera.projection.vertical_fov_rad = radians,
            Edit::ResetCamera => self.controls.request_reset(),
            Edit::AddLight => self.add_point_light(),
            Edit::Light { index, light } => {
                let Some(target) = self.scene.lights.get_mut(index) else {
                    return;
                };
                let renamed = target.name != light.name;
                *target = *light;
                if renamed {
                    self.rename_light_node(index);
                }
                self.mark(SceneUpdate::light(index));
            }
            Edit::InstanceTransform { index, transform } => {
                let Some(instance) = self.scene.geometry.instances.get_mut(index) else {
                    return;
                };
                instance.transform = transform;
                self.resummarize_materials();
                self.mark(SceneUpdate::instance_transform(index));
            }
            Edit::Material { index, material } => {
                let Some(target) = self.scene.materials.get_mut(index) else {
                    return;
                };
                let surface_changed = target.surface != material.surface;
                let emission_changed = target.emission != material.emission;
                *target = *material;
                if surface_changed {
                    self.mark(SceneUpdate::material_surface(index));
                }
                if emission_changed {
                    self.scene.refresh_emissive_components();
                    self.resummarize_materials();
                    self.mark(SceneUpdate::material_emission(index));
                }
            }
            Edit::AddMaterial => {
                let id = self.scene.add_material();
                self.inspector.selected_material = Some(id);
                self.inspector.tab = ui::Tab::Material;
                self.resummarize_materials();
                self.mark(SceneUpdate::material(id.0));
            }
            Edit::DeleteMaterial(id) => {
                if self.scene.remove_material(id).is_some() {
                    self.inspector.selected_material = Some(MaterialId::DEFAULT);
                    self.resummarize_materials();
                    self.mark(SceneUpdate::material(MaterialId::DEFAULT.0));
                }
            }
        }
    }

    fn add_point_light(&mut self) {
        let index = self.scene.lights.len();
        let position = self
            .scene
            .camera
            .world
            .transform_point3(DVec3::new(0.0, 0.0, -2.0));
        self.scene.lights.push(Light::point(
            format!("Light {index:02}"),
            DMat4::from_translation(position),
        ));
        self.object_count += 1;
        self.scene.nodes = build_scene_nodes(&self.scene.geometry.instances, &self.scene.lights);
        self.inspector.tree.invalidate();
        self.inspector.selected_node = self
            .scene
            .nodes
            .iter()
            .find(|node| matches!(node.object, NodeObject::Light { light } if light == index))
            .map(|node| node.id);
        self.inspector.tab = ui::Tab::Object;
        self.mark(SceneUpdate::light(index));
    }

    /// Keep the tree's label in step with a renamed light. Nodes mirror the light array rather than
    /// owning it, so a rename has to be pushed across.
    fn rename_light_node(&mut self, index: usize) {
        let Some(name) = self.scene.lights.get(index).map(|light| light.name.clone()) else {
            return;
        };
        if let Some(node) = self
            .scene
            .nodes
            .iter_mut()
            .find(|node| matches!(node.object, NodeObject::Light { light } if light == index))
        {
            node.name = name;
        }
    }

    fn mark(&mut self, update: SceneUpdate) {
        self.pending_scene_update
            .get_or_insert_with(SceneUpdate::default)
            .merge(update);
    }

    /// Push queued settings and scene edits into the renderer.
    fn flush_updates(&mut self, ctx: &FrameCtx) {
        if self.update_deferred {
            return;
        }
        if self.settings_dirty {
            match self.renderer.update_settings(self.settings) {
                Ok(()) => {
                    self.settings_dirty = false;
                    self.renderer_error = None;
                }
                Err(error) => self.report(format!("failed to apply renderer settings: {error:#}")),
            }
        }
        let Some(update) = self.pending_scene_update.take() else {
            return;
        };
        match self.renderer.update_scene(
            ctx.device,
            &self.scene,
            &self.light_spectrum,
            &update,
            ctx.slot,
        ) {
            Ok(()) => self.renderer_error = None,
            Err(error) => {
                self.report(format!("failed to apply scene update: {error:#}"));
                self.pending_scene_update = Some(update);
            }
        }
    }

    fn report(&mut self, message: String) {
        eprintln!("{message}");
        self.renderer_error = Some(message);
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
            self.inspector.viewport_focused = false;
        }
        if matches!(
            event,
            WindowEvent::MouseInput {
                state: winit::event::ElementState::Pressed,
                ..
            }
        ) && self.inspector.viewport_hovered
        {
            // The press itself focuses the viewport, so the first drag after clicking it is
            // immediately usable even though egui builds the next frame after this event.
            self.inspector.viewport_focused = true;
        }
        let keyboard_release = matches!(
            event,
            WindowEvent::KeyboardInput { event, .. }
                if event.state == winit::event::ElementState::Released
        );
        let camera_accepts = match event {
            WindowEvent::Focused(false) => true,
            WindowEvent::KeyboardInput { .. } => self.inspector.viewport_focused,
            WindowEvent::CursorMoved { .. } | WindowEvent::MouseInput { .. } => {
                (self.inspector.viewport_focused && self.inspector.viewport_hovered)
                    || self.controls.is_dragging()
            }
            _ => false,
        };
        if camera_accepts || keyboard_release {
            self.controls.window_event(event);
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, stats: PerformanceStats) {
        let started = std::time::Instant::now();
        // Built field by field rather than through a `&self` method so the inspector's own state
        // can be borrowed mutably alongside the read-only view of everything else.
        let view = ui::View {
            scene: &self.scene,
            settings: self.settings,
            stats,
            ui_ms: self.ui_ms,
            samples_drawn: self.renderer.samples_drawn(),
            viewport_extent: self.viewport_extent,
            renderer_error: self.renderer_error.as_deref(),
            camera_position: self.controls.position(),
            mesh_pivots: &self.mesh_pivots,
            material_usage: &self.material_usage,
            emissive_areas: &self.emissive_areas,
            object_count: self.object_count,
            default_spectrum: &self.light_spectrum.name,
        };
        let edits = ui::Inspector::new(&mut self.inspector, view).show(ui);
        let elapsed = started.elapsed().as_secs_f64() * 1000.0;
        self.ui_ms = if self.ui_ms == 0.0 {
            elapsed
        } else {
            self.ui_ms * 0.9 + elapsed * 0.1
        };
        // An edit mid-drag is provisional: hold the renderer transaction until the pointer is
        // released so a slider drag resets the progressive film once, not every frame.
        self.update_deferred =
            !edits.is_empty() && ui.ctx().input(|input| input.pointer.any_down());
        for edit in edits {
            self.apply(edit);
        }
        self.viewport_extent = self.inspector.viewport_extent;
    }

    fn pre_render(&mut self, ctx: &FrameCtx, cmd: &mut CommandBuffer) {
        self.flush_updates(ctx);
        if let Some(world) = self.controls.update() {
            self.scene.camera.world = world;
        }
        let frame = RenderFrame {
            device: ctx.device,
            extent: ctx.extent,
            slot: ctx.slot,
        };
        if let Err(error) = self.renderer.encode(&frame, cmd, &self.scene.camera) {
            self.report(format!("renderer pre-render failed: {error:#}"));
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
            || self.update_deferred
    }

    fn destroy(self, device: &Device) {
        self.renderer.destroy(device);
    }
}

/// The centre of a mesh's bounding box, in object space.
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

/// The renderer in use, with the raster backend as the fallback when the path tracer cannot build.
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
