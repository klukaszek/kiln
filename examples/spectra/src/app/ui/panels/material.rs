//! The material library: a browser over every material in the scene, then the surface and emission
//! editors for the selected one.

use egui::{Rect, Sense, Stroke, pos2, vec2};

use spectra::base::scene::{
    EmitterExtent, Illuminant, Material, MaterialId, PrincipledBsdf, Scene, Surface, TextureId,
};

use super::super::theme::{self, HAIRLINE, PALETTE, ROW_HEIGHT};
use super::super::widgets;
use super::super::{Edit, Inspector};
use super::illuminant;

/// Browser rows carry two lines of text, so they are taller than a tree row by enough to keep the
/// name's descenders clear of the classification beneath it.
const BROWSER_ROW: f32 = ROW_HEIGHT + 11.0;
/// Vertical offset of each text line from the row's centre.
const BROWSER_LINE_OFFSET: f32 = 8.0;

pub fn show(inspector: &mut Inspector<'_>, ui: &mut egui::Ui) {
    browser(inspector, ui);
    let Some(selected) = inspector.state.selected_material else {
        widgets::placeholder(ui, "Select a material");
        return;
    };
    let Some(material) = inspector.view.scene.materials.get(selected.0).cloned() else {
        inspector.state.selected_material = Some(MaterialId::DEFAULT);
        return;
    };
    editor(inspector, ui, selected, material);
}

fn browser(inspector: &mut Inspector<'_>, ui: &mut egui::Ui) {
    let count = inspector.view.scene.materials.len();
    theme::section_title(ui, "library", Some(format!("{count}")));
    ui.horizontal(|ui| {
        if theme::button(ui, "+ new material")
            .on_hover_text("Add an unbound material to the library")
            .clicked()
        {
            inspector.edit(Edit::AddMaterial);
        }
    });
    ui.add_space(4.0);

    let mut clicked = None;
    egui::ScrollArea::vertical()
        .id_salt("material_browser")
        .max_height(BROWSER_ROW * 5.0)
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 0.0;
            for (index, material) in inspector.view.scene.materials.iter().enumerate() {
                let id = MaterialId(index);
                let selected = inspector.state.selected_material == Some(id);
                let (rect, _) =
                    ui.allocate_exact_size(vec2(ui.available_width(), BROWSER_ROW), Sense::hover());
                let response = ui.interact(rect, ui.id().with(("material", index)), Sense::click());
                let ink = theme::row_background(ui, rect, selected, response.hovered());
                let emissive = material.is_emissive();

                let painter = ui.painter();
                painter.rect_filled(
                    Rect::from_center_size(
                        pos2(rect.left() + 14.0, rect.center().y),
                        vec2(7.0, 7.0),
                    ),
                    0,
                    if emissive { PALETTE.emissive } else { ink },
                );
                painter.text(
                    pos2(rect.left() + 26.0, rect.center().y - BROWSER_LINE_OFFSET),
                    egui::Align2::LEFT_CENTER,
                    &material.name,
                    egui::FontId::monospace(12.0),
                    ink,
                );
                painter.text(
                    pos2(rect.left() + 26.0, rect.center().y + BROWSER_LINE_OFFSET),
                    egui::Align2::LEFT_CENTER,
                    format!(
                        "{}  \u{2502}  {} tri",
                        classify(index, emissive),
                        inspector
                            .view
                            .material_usage
                            .get(index)
                            .copied()
                            .unwrap_or(0)
                    ),
                    egui::FontId::monospace(10.0),
                    if selected { ink } else { PALETTE.ink_dim },
                );
                painter.line_segment(
                    [rect.left_bottom(), rect.right_bottom()],
                    Stroke::new(HAIRLINE, PALETTE.ink.gamma_multiply(0.15)),
                );
                if response.clicked() {
                    clicked = Some(id);
                }
            }
            if count == 0 {
                widgets::placeholder(ui, "No materials in this scene");
            }
        });
    ui.add_space(8.0);

    if let Some(id) = clicked {
        inspector.state.selected_material = Some(id);
    }
}

fn classify(index: usize, emissive: bool) -> &'static str {
    if index == MaterialId::DEFAULT.0 {
        "DEFAULT"
    } else if emissive {
        "EMISSIVE"
    } else {
        "SURFACE"
    }
}

fn editor(inspector: &mut Inspector<'_>, ui: &mut egui::Ui, id: MaterialId, material: Material) {
    let mut edited = material;
    let mut changed = false;
    let protected = id == MaterialId::DEFAULT;

    theme::section_title(
        ui,
        "identity",
        Some(classify(id.0, edited.is_emissive()).to_owned()),
    );
    theme::card(ui, |ui| {
        widgets::grid(ui, ("material_identity", id.0), |ui| {
            changed |= widgets::text_row(ui, "Name", &mut edited.name);
            widgets::readonly_row(
                ui,
                "Triangles",
                inspector
                    .view
                    .material_usage
                    .get(id.0)
                    .copied()
                    .unwrap_or(0),
            );
        });
        ui.add_space(4.0);
        if theme::danger_button(ui, "delete material", !protected)
            .on_hover_text(if protected {
                "The default material is permanent so no binding can be left invalid"
            } else {
                "Delete this material and rebind its geometry to the default"
            })
            .clicked()
        {
            inspector.edit(Edit::DeleteMaterial(id));
        }
    });

    changed |= surface(ui, id, &mut edited.surface);
    if let Surface::Principled(PrincipledBsdf {
        base_color_texture: Some(texture),
        ..
    }) = edited.surface
    {
        base_color_texture(ui, inspector.view.scene, texture);
    }

    changed |= emission(inspector, ui, id, &mut edited.emission);

    if changed {
        inspector.edit(Edit::Material {
            index: id.0,
            material: Box::new(edited),
        });
    }
}

fn surface(ui: &mut egui::Ui, id: MaterialId, surface: &mut Surface) -> bool {
    let model = match surface {
        Surface::Principled(_) => "PRINCIPLED",
        Surface::Diffuse { .. } => "DIFFUSE",
        Surface::Dielectric { .. } => "DIELECTRIC",
        Surface::Conductor { .. } => "CONDUCTOR",
    };
    theme::section_title(ui, "surface", Some(model.to_owned()));
    let mut changed = false;
    theme::card(ui, |ui| {
        widgets::grid(ui, ("material_surface", id.0), |ui| match surface {
            Surface::Principled(bsdf) => {
                changed |=
                    widgets::color_row(ui, (id.0, "base"), "Base color", &mut bsdf.base_color);
                changed |=
                    widgets::scalar_row(ui, "Roughness", &mut bsdf.roughness, 0.0..=1.0, 0.01, "");
                changed |=
                    widgets::scalar_row(ui, "Metallic", &mut bsdf.metallic, 0.0..=1.0, 0.01, "");
                changed |= widgets::scalar_row(ui, "IOR", &mut bsdf.ior, 1.0..=4.0, 0.01, "");
            }
            Surface::Diffuse { albedo } => {
                changed |= widgets::color_row(ui, (id.0, "albedo"), "Albedo", albedo);
            }
            Surface::Dielectric { ior, roughness } => {
                changed |= widgets::scalar_row(ui, "IOR", ior, 1.0..=4.0, 0.01, "");
                changed |= widgets::scalar_row(ui, "Roughness", roughness, 0.0..=1.0, 0.01, "");
            }
            Surface::Conductor { eta, k, roughness } => {
                changed |= widgets::vec3_row(ui, (id.0, "eta"), "Eta", eta, 0.01, 3);
                changed |= widgets::vec3_row(ui, (id.0, "k"), "K", k, 0.01, 3);
                changed |= widgets::scalar_row(ui, "Roughness", roughness, 0.0..=1.0, 0.01, "");
            }
        });
    });
    changed
}

/// The emission slot. A material that does not emit shows one control to make it an emitter; once
/// it does, it gets the same physical editor an analytic light gets.
fn emission(
    inspector: &mut Inspector<'_>,
    ui: &mut egui::Ui,
    id: MaterialId,
    emission: &mut Illuminant,
) -> bool {
    if !emission.enabled {
        theme::section_title(ui, "emission", Some("not an emitter".into()));
        let mut enable = false;
        theme::card(ui, |ui| {
            enable = theme::button(ui, "make this material emit")
                .on_hover_text("Bind an illuminant to every surface using this material")
                .clicked();
        });
        widgets::note(
            ui,
            "Emissive geometry is how most exported USD scenes light themselves. An emitting \
             material gets the same spectrum and photometric units an analytic light does.",
        );
        if enable {
            *emission = Illuminant::default();
        }
        return enable;
    }

    let area = inspector
        .view
        .emissive_areas
        .get(id.0)
        .copied()
        .unwrap_or(0.0);
    let extent = EmitterExtent::Surface(area);
    let changed = illuminant::show(
        ui,
        ("material_emission", id.0),
        emission,
        extent,
        inspector.view.default_spectrum,
    );
    widgets::note(
        ui,
        "The area covers every emissive region bound to this material, so all of them emit the \
         same luminance and next-event estimation agrees with a BSDF-sampled hit.",
    );
    changed
}

fn base_color_texture(ui: &mut egui::Ui, scene: &Scene, texture: TextureId) {
    theme::section_title(ui, "base color texture", None);
    theme::card(ui, |ui| {
        widgets::grid(ui, ("texture", texture.0), |ui| {
            let Some(binding) = scene.textures.get(texture.0) else {
                widgets::readonly_row(ui, "Texture", format!("invalid #{}", texture.0));
                return;
            };
            let Some(image) = scene.images.get(binding.image.0) else {
                widgets::readonly_row(ui, "Image", format!("invalid #{}", binding.image.0));
                return;
            };
            widgets::readonly_row(ui, "Image", &image.name);
            widgets::readonly_row(
                ui,
                "Size",
                format!("{}\u{00d7}{}", image.width, image.height),
            );
            widgets::readonly_row(ui, "Color space", format!("{:?}", image.color_space));
            widgets::readonly_row(
                ui,
                "Wrap",
                format!("{:?} / {:?}", binding.wrap_u, binding.wrap_v),
            );
        });
    });
}

/// A compact material row shown under a mesh's bindings. Opens the material in this tab.
pub fn binding_row(
    inspector: &mut Inspector<'_>,
    ui: &mut egui::Ui,
    material_index: usize,
    triangles: usize,
) {
    let id = MaterialId(material_index);
    let Some(material) = inspector.view.scene.materials.get(material_index) else {
        widgets::placeholder(ui, &format!("Invalid material #{material_index}"));
        return;
    };
    let (name, emissive) = (material.name.clone(), material.is_emissive());
    theme::card(ui, |ui| {
        ui.horizontal(|ui| {
            theme::bullet(
                ui,
                if emissive {
                    PALETTE.emissive
                } else {
                    PALETTE.ink
                },
            );
            ui.label(
                egui::RichText::new(&name)
                    .monospace()
                    .strong()
                    .color(PALETTE.ink),
            );
            ui.label(
                egui::RichText::new(format!("{triangles} tri"))
                    .small()
                    .monospace()
                    .color(PALETTE.ink_dim),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if theme::button(ui, "open").clicked() {
                    inspector.state.selected_material = Some(id);
                    inspector.state.tab = super::super::Tab::Material;
                }
            });
        });
    });
}
