//! Inspector state that outlives a frame: which tab is open, what is selected, and the cached
//! scene-tree rows.
//!
//! This is view state only. Nothing here is authoritative about the scene; it holds selections and
//! layout, and every scene change leaves as an [`super::Edit`].

use std::collections::HashSet;

use spectra::base::scene::{MaterialId, NodeId, Scene};

use super::transform::TransformEditor;

/// The property tabs, in the order they appear on the strip.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum Tab {
    Render,
    Camera,
    #[default]
    Object,
    Material,
}

impl Tab {
    pub const ALL: [Self; 4] = [Self::Render, Self::Camera, Self::Object, Self::Material];

    pub fn label(self) -> &'static str {
        match self {
            Self::Render => "RENDER",
            Self::Camera => "CAMERA",
            Self::Object => "OBJECT",
            Self::Material => "MATERIAL",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Self::Render => "Sampling and resolution",
            Self::Camera => "Viewpoint and projection",
            Self::Object => "The object selected in the scene tree",
            Self::Material => "Surface and emission library",
        }
    }
}

#[derive(Default)]
pub struct InspectorState {
    pub open: bool,
    pub tab: Tab,
    pub selected_node: Option<NodeId>,
    pub selected_material: Option<MaterialId>,
    pub tree: TreeState,
    /// Height of the scene tree above the splitter.
    pub tree_height: f32,
    pub transform: Option<TransformEditor>,
    /// Whether the viewport, rather than the inspector, owns keyboard and mouse input.
    pub viewport_focused: bool,
    pub viewport_hovered: bool,
    /// The region left for the renderer after the docked panels, in physical pixels.
    pub viewport_extent: glam::UVec2,
    style_installed: bool,
}

impl InspectorState {
    pub fn new(selected_node: Option<NodeId>) -> Self {
        Self {
            open: true,
            selected_node,
            selected_material: Some(MaterialId::DEFAULT),
            tree_height: 260.0,
            ..Self::default()
        }
    }

    /// Install the theme on the first frame. egui styles are global, so this happens once against
    /// the live context rather than at construction, where no context exists yet.
    pub fn install_style_once(&mut self, ctx: &egui::Context) {
        if !self.style_installed {
            super::theme::install(ctx);
            self.style_installed = true;
        }
    }
}

/// One visible line of the scene tree.
#[derive(Clone, Copy)]
pub struct TreeRow {
    pub node: NodeId,
    pub depth: u16,
}

/// The flattened scene tree. Rebuilt only when the hierarchy or its expansion changes, so
/// scrolling a large scene costs nothing beyond drawing the visible rows.
pub struct TreeState {
    pub expanded: HashSet<NodeId>,
    pub visible: Vec<TreeRow>,
    node_count: usize,
    dirty: bool,
}

impl Default for TreeState {
    fn default() -> Self {
        Self {
            expanded: HashSet::new(),
            visible: Vec::new(),
            node_count: 0,
            dirty: true,
        }
    }
}

impl TreeState {
    pub fn invalidate(&mut self) {
        self.dirty = true;
    }

    pub fn toggle(&mut self, node: NodeId) {
        if !self.expanded.remove(&node) {
            self.expanded.insert(node);
        }
        self.dirty = true;
    }

    pub fn rebuild_if_dirty(&mut self, scene: &Scene) {
        if !self.dirty && self.node_count == scene.nodes.len() {
            return;
        }
        let first_build = self.node_count == 0 && self.visible.is_empty();
        self.node_count = scene.nodes.len();
        self.expanded.retain(|node| {
            scene
                .nodes
                .get(node.0)
                .is_some_and(|node| node.is_group() && !node.children.is_empty())
        });
        if first_build {
            for node in &scene.nodes {
                if node.is_group() && !node.children.is_empty() {
                    self.expanded.insert(node.id);
                }
            }
        }
        self.visible.clear();
        if let Some(root) = scene.nodes.first() {
            for &child in &root.children {
                append_rows(scene, child, 0, &self.expanded, &mut self.visible);
            }
        }
        self.dirty = false;
    }
}

fn append_rows(
    scene: &Scene,
    node_id: NodeId,
    depth: u16,
    expanded: &HashSet<NodeId>,
    visible: &mut Vec<TreeRow>,
) {
    let Some(node) = scene.nodes.get(node_id.0) else {
        return;
    };
    visible.push(TreeRow {
        node: node_id,
        depth,
    });
    if expanded.contains(&node_id) {
        for &child in &node.children {
            append_rows(scene, child, depth.saturating_add(1), expanded, visible);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spectra::importers::usd;

    #[test]
    fn collapsing_a_group_hides_its_children_after_a_rebuild() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/cornell-box.usda");
        let scene = usd::load(&path).unwrap();
        let group = scene
            .nodes
            .iter()
            .find(|node| node.id != NodeId(0) && node.is_group() && !node.children.is_empty())
            .unwrap();
        let child = group.children[0];

        let mut state = TreeState::default();
        state.rebuild_if_dirty(&scene);
        assert!(state.expanded.contains(&group.id));
        assert!(state.visible.iter().any(|row| row.node == child));

        state.toggle(group.id);
        state.rebuild_if_dirty(&scene);
        assert!(!state.expanded.contains(&group.id));
        assert!(!state.visible.iter().any(|row| row.node == child));
    }
}
