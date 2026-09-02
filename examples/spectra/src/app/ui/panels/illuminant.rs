//! The physical emitter editor, shared by analytic lights and emissive materials.
//!
//! Both kinds of emitter carry the same [`Illuminant`], so both are authored here: pick a spectrum,
//! state an intensity in a real photometric unit, and read back the luminance the trace will
//! actually use. A USD scene lit by emissive geometry gets exactly the controls one lit by lux
//! prims gets, which is the whole point of the shared type.

use spectra::base::scene::{EmitterExtent, Illuminant, IntensityUnit, SpectrumSource};
use spectra::renderers::spectral::spectrum;

use super::super::theme::{self, PALETTE};
use super::super::widgets;

/// Which spectrum family is selected. The picker compares families, not values, so changing a
/// blackbody's temperature does not read as picking a different entry.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Family {
    Named(usize),
    Custom,
    Blackbody,
    Line,
}

/// Draw the emitter controls for `illuminant`, reporting whether anything changed.
///
/// `extent` is the geometry the emitter's total power spreads over; it decides which units are
/// offered and what the derived readout means. `default_spectrum` names the renderer-wide spectrum
/// an unresolved name falls back to.
pub fn show(
    ui: &mut egui::Ui,
    id: impl std::hash::Hash + Copy,
    illuminant: &mut Illuminant,
    extent: EmitterExtent,
    default_spectrum: &str,
) -> bool {
    let mut changed = false;

    theme::section_title(ui, "emission", Some(extent_label(extent)));
    widgets::grid(ui, (id, "emission"), |ui| {
        changed |= widgets::checkbox_row(ui, "Enabled", &mut illuminant.enabled);
        changed |= spectrum_rows(ui, id, illuminant, default_spectrum);
        changed |= widgets::color_row(ui, (id, "tint"), "Tint", &mut illuminant.color);
        changed |= unit_rows(ui, id, illuminant, extent);
    });

    widgets::note(
        ui,
        "Chromaticity comes from the spectrum. The tint contributes its luminance, so a \
         saturated tint dims the emitter rather than colouring it.",
    );
    ui.add_space(6.0);

    derived(ui, id, illuminant, extent);
    changed
}

/// The spectrum family picker and the parameter its family needs.
fn spectrum_rows(
    ui: &mut egui::Ui,
    id: impl std::hash::Hash + Copy,
    illuminant: &mut Illuminant,
    default_spectrum: &str,
) -> bool {
    let mut changed = false;
    let mut family = family_of(&illuminant.spectrum);
    let mut options = spectrum::BUILTIN_NAMES
        .iter()
        .enumerate()
        .map(|(index, _)| Family::Named(index))
        .collect::<Vec<_>>();
    // A spectrum loaded from a file is not in the built-in list, so keep it selectable.
    if family == Family::Custom {
        options.push(Family::Custom);
    }
    options.extend([Family::Blackbody, Family::Line]);

    let current = illuminant.spectrum.clone();
    if widgets::combo_row(
        ui,
        (id, "spectrum"),
        "Spectrum",
        &mut family,
        options,
        |family| family_label(*family, &current),
    ) {
        illuminant.spectrum = source_for(family, &current);
        changed = true;
    }

    match &mut illuminant.spectrum {
        SpectrumSource::Blackbody { kelvin } => {
            changed |=
                widgets::scalar_row(ui, "Temperature", kelvin, 1000.0..=25_000.0, 10.0, " K");
        }
        SpectrumSource::Monochromatic { nanometres } => {
            changed |= widgets::scalar_row(
                ui,
                "Wavelength",
                nanometres,
                spectrum::LAMBDA_MIN..=spectrum::LAMBDA_MAX,
                1.0,
                " nm",
            );
        }
        // Only a name outside the built-in list can fail to resolve, and that check is already the
        // one the family picker made; re-resolving here would rebuild the spectrum every frame.
        SpectrumSource::Named(_) if family == Family::Custom => {
            widgets::readonly_row(ui, "Resolves to", default_spectrum);
        }
        SpectrumSource::Named(_) => {}
    }
    changed
}

/// The unit picker, the intensity in that unit, and its exposure stops.
fn unit_rows(
    ui: &mut egui::Ui,
    id: impl std::hash::Hash + Copy,
    illuminant: &mut Illuminant,
    extent: EmitterExtent,
) -> bool {
    let mut changed = false;
    let units = IntensityUnit::ALL
        .into_iter()
        .filter(|unit| unit.applies_to(extent))
        .collect::<Vec<_>>();
    changed |= widgets::combo_row(
        ui,
        (id, "unit"),
        "Unit",
        &mut illuminant.unit,
        units,
        |unit| format!("{}  ({})", unit.label(), unit.symbol(extent)),
    );
    changed |= widgets::scalar_row(
        ui,
        "Intensity",
        &mut illuminant.intensity,
        0.0..=1.0e9,
        intensity_speed(illuminant.unit),
        &format!(" {}", illuminant.unit.symbol(extent)),
    );
    changed |= widgets::stops_row(ui, "Exposure", &mut illuminant.exposure);
    changed
}

/// The read-back: what the trace will emit, and the geometry that number was derived from.
fn derived(
    ui: &mut egui::Ui,
    id: impl std::hash::Hash + Copy,
    illuminant: &Illuminant,
    extent: EmitterExtent,
) {
    let efficacy = luminous_efficacy(illuminant);
    let emitted = illuminant.emitted_luminance(extent, efficacy);
    theme::card(ui, |ui| {
        widgets::grid(ui, (id, "derived"), |ui| {
            let symbol = IntensityUnit::Luminance.symbol(extent);
            widgets::highlight_row(
                ui,
                "Emitted",
                format!("{emitted:.3} {symbol}"),
                if emitted > 0.0 {
                    PALETTE.emissive
                } else {
                    PALETTE.ink_dim
                },
            );
            if let EmitterExtent::Surface(area) = extent {
                widgets::readonly_row(ui, "Emitting area", format!("{area:.4} m\u{b2}"));
            }
            if illuminant.unit == IntensityUnit::Power {
                widgets::readonly_row(ui, "Efficacy", format!("{efficacy:.1} lm/W"));
            }
        });
    });
}

/// The lm/W of this illuminant's spectrum, needed to restate radiant power as luminous flux. Only
/// resolved for the unit that consumes it, since resolving a spectrum allocates its sample table.
fn luminous_efficacy(illuminant: &Illuminant) -> f32 {
    if illuminant.unit != IntensityUnit::Power {
        return 0.0;
    }
    spectrum::named(&illuminant.spectrum.key())
        .map(|spd| spd.luminous_efficacy())
        .unwrap_or_default()
}

fn extent_label(extent: EmitterExtent) -> String {
    match extent {
        EmitterExtent::Surface(_) => "surface emitter".into(),
        EmitterExtent::Punctual => "punctual emitter".into(),
        EmitterExtent::Infinite => "emitter at infinity".into(),
    }
}

/// Intensity spans very different magnitudes per unit, so each drags at its own rate.
fn intensity_speed(unit: IntensityUnit) -> f64 {
    match unit {
        IntensityUnit::Luminance => 1.0,
        IntensityUnit::Flux => 10.0,
        IntensityUnit::Power => 0.1,
    }
}

fn family_of(source: &SpectrumSource) -> Family {
    match source {
        SpectrumSource::Blackbody { .. } => Family::Blackbody,
        SpectrumSource::Monochromatic { .. } => Family::Line,
        SpectrumSource::Named(name) => spectrum::BUILTIN_NAMES
            .iter()
            .position(|builtin| builtin == name)
            .map_or(Family::Custom, Family::Named),
    }
}

fn family_label(family: Family, current: &SpectrumSource) -> String {
    match family {
        Family::Named(index) => spectrum::BUILTIN_NAMES[index].to_owned(),
        Family::Custom => current.label(),
        Family::Blackbody => "BLACKBODY".into(),
        Family::Line => "SPECTRAL LINE".into(),
    }
}

/// Build the source for a newly picked family, carrying over the previous parameter when the
/// family is unchanged so re-picking it does not reset a tuned temperature.
fn source_for(family: Family, current: &SpectrumSource) -> SpectrumSource {
    match family {
        Family::Named(index) => SpectrumSource::Named(spectrum::BUILTIN_NAMES[index].to_owned()),
        Family::Custom => current.clone(),
        Family::Blackbody => SpectrumSource::Blackbody {
            kelvin: match current {
                SpectrumSource::Blackbody { kelvin } => *kelvin,
                _ => 3200.0,
            },
        },
        Family::Line => SpectrumSource::Monochromatic {
            nanometres: match current {
                SpectrumSource::Monochromatic { nanometres } => *nanometres,
                _ => 550.0,
            },
        },
    }
}
