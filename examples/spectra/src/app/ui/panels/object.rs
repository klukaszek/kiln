//! The object selected in the scene tree.
//!
//! This panel owns identity and mesh data, and hands lights and materials to their own editors, so
//! the same light controls appear here and nowhere else in duplicate.

use glam::DVec3;

use spectra::scene::{NodeId, NodeObject};

use super::super::theme::{self, PALETTE};
use super::super::transform::TransformEditor;
use super::super::widgets;
use super::super::{Edit, Inspector};
use super::{light, material};

pub fn show(inspector: &mut Inspector<'_>, ui: &mut egui::Ui) {
    let Some(node_id) = inspector.state.selected_node else {
        widgets::placeholder(ui, "Select an object in the scene tree");
        return;
    };
    let Some(node) = inspector.view.scene.nodes.get(node_id.0) else {
        inspector.state.selected_node = None;
        return;
    };
    let (name, path, object, child_count) = (
        node.name.clone(),
        node.path.clone(),
        node.object,
        node.children.len(),
    );

    identity(ui, &name, &path, object);
    match object {
        NodeObject::Group => group(ui, child_count),
        NodeObject::Mesh { instance } => mesh(inspector, ui, node_id, instance),
        NodeObject::Light { light } => light::show(inspector, ui, node_id, light),
    }
}

fn identity(ui: &mut egui::Ui, name: &str, path: &str, object: NodeObject) {
    let kind = match object {
        NodeObject::Group => "GROUP",
        NodeObject::Mesh { .. } => "MESH",
        NodeObject::Light { .. } => "LIGHT",
    };
    ui.horizontal(|ui| {
        theme::bullet(ui, PALETTE.ink);
        ui.label(
            egui::RichText::new(name)
                .monospace()
                .strong()
                .size(14.0)
                .color(PALETTE.ink),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                egui::RichText::new(kind)
                    .monospace()
                    .size(10.0)
                    .color(PALETTE.ink_dim),
            );
        });
    });
    ui.label(
        egui::RichText::new(path)
            .small()
            .monospace()
            .color(PALETTE.ink_faint),
    );
    ui.add_space(6.0);
}

fn group(ui: &mut egui::Ui, child_count: usize) {
    theme::section_title(ui, "group", None);
    theme::card(ui, |ui| {
        widgets::grid(ui, "group_summary", |ui| {
            widgets::readonly_row(ui, "Children", child_count);
        });
    });
    widgets::note(
        ui,
        "Groups come from the USD path hierarchy. Transforms belong to the mesh and light objects \
         beneath them.",
    );
}

fn mesh(inspector: &mut Inspector<'_>, ui: &mut egui::Ui, node_id: NodeId, instance_index: usize) {
    let scene = inspector.view.scene;
    let Some(instance) = scene.geometry.instances.get(instance_index) else {
        return;
    };
    let mesh_index = instance.mesh.0;
    let source = instance.transform;
    let pivot = inspector
        .view
        .mesh_pivots
        .get(mesh_index)
        .copied()
        .unwrap_or(DVec3::ZERO);

    theme::section_title(ui, "transform", None);
    let mut editor = TransformEditor::resolve(inspector.state.transform, node_id, source, pivot);
    let mut changed = false;
    theme::card(ui, |ui| {
        widgets::grid(ui, ("mesh_transform", instance_index), |ui| {
            changed = editor.rows(ui, ("mesh", instance_index));
        });
    });
    if changed {
        let transform = editor.matrix();
        editor.adopt(transform);
        inspector.edit(Edit::InstanceTransform {
            index: instance_index,
            transform,
        });
    }
    inspector.state.transform = Some(editor);

    let Some(mesh) = inspector.view.scene.geometry.meshes.get(mesh_index) else {
        return;
    };
    theme::section_title(ui, "geometry", None);
    theme::card(ui, |ui| {
        widgets::grid(ui, ("mesh_summary", instance_index), |ui| {
            widgets::readonly_row(ui, "Vertices", mesh.vertices.len());
            widgets::readonly_row(ui, "Triangles", mesh.indices.len() / 3);
            widgets::readonly_row(ui, "Primitives", mesh.primitives.len());
            widgets::highlight_row(
                ui,
                "Emissive regions",
                mesh.emissive_components.len(),
                if mesh.emissive_components.is_empty() {
                    PALETTE.ink
                } else {
                    PALETTE.emissive
                },
            );
        });
    });

    bindings(inspector, ui, mesh_index);
}

/// The materials this mesh binds, with the triangle count each covers. Selecting one opens it in
/// the material tab rather than duplicating the editor here.
fn bindings(inspector: &mut Inspector<'_>, ui: &mut egui::Ui, mesh_index: usize) {
    let Some(mesh) = inspector.view.scene.geometry.meshes.get(mesh_index) else {
        return;
    };
    let mut usage: Vec<(usize, usize)> = Vec::new();
    for primitive in &mesh.primitives {
        let triangles = primitive.index_count / 3;
        match usage
            .iter_mut()
            .find(|(material, _)| *material == primitive.material.0)
        {
            Some((_, count)) => *count += triangles,
            None => usage.push((primitive.material.0, triangles)),
        }
    }

    theme::section_title(ui, "material bindings", Some(format!("{}", usage.len())));
    if usage.is_empty() {
        widgets::placeholder(ui, "No material bindings");
        return;
    }
    for (material_index, triangles) in usage {
        material::binding_row(inspector, ui, material_index, triangles);
    }
}
