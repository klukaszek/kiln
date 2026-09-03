//! The physical emitter description shared by analytic lights and emissive materials.
//!
//! USD scenes light themselves both ways: some author `UsdLux` prims, many more just bind an
//! emissive material to geometry. Both reach the renderer as the same [`Illuminant`], so a mesh
//! emitter is editable with the same spectrum, intensity, and photometric units as a lux prim.

use glam::Vec3;

use super::material::luminance;

/// How an illuminant's spectral power distribution is chosen.
///
/// The variants name a spectrum rather than carrying one: sampling tables are expensive to bake and
/// belong to the spectral backend, which resolves these through `spectrum::named`.
#[derive(Clone, Debug, PartialEq)]
pub enum SpectrumSource {
    /// A built-in illuminant (`"A"`, `"D50"`, `"D65"`, `"E"`, `"FL2"`, `"FL7"`, `"FL11"`) or the
    /// stem of a measured lamp spectrum dropped into the LSPDD asset folder.
    Named(String),
    /// A Planckian radiator at a colour temperature.
    Blackbody { kelvin: f32 },
    /// A single narrow spectral line.
    Monochromatic { nanometres: f32 },
}

impl Default for SpectrumSource {
    fn default() -> Self {
        Self::Named("D65".into())
    }
}

impl SpectrumSource {
    /// The lookup key understood by the backend's spectrum resolver.
    pub fn key(&self) -> String {
        match self {
            Self::Named(name) => name.clone(),
            Self::Blackbody { kelvin } => format!("{kelvin:.0}K"),
            Self::Monochromatic { nanometres } => format!("{nanometres:.0}nm"),
        }
    }

    /// A short label for the inspector's spectrum picker.
    pub fn label(&self) -> String {
        match self {
            Self::Named(name) => name.clone(),
            Self::Blackbody { kelvin } => format!("BLACKBODY {kelvin:.0}K"),
            Self::Monochromatic { nanometres } => format!("LINE {nanometres:.0}NM"),
        }
    }
}

/// The physical quantity an illuminant's `intensity` measures.
///
/// [`Self::Luminance`] is the renderer's native quantity; the other two are total-power units that
/// need the emitter's extent, and [`Self::Power`] additionally needs the luminous efficacy of the
/// illuminant's own spectrum. Authoring in lumens or watts is what makes two emitters with
/// different spectra comparable: an equal-wattage tungsten and fluorescent emitter differ in
/// brightness exactly as the real lamps do.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum IntensityUnit {
    /// Emitted luminance in cd/m² for a surface emitter, luminous intensity in cd for a punctual
    /// one. Independent of the emitter's size.
    #[default]
    Luminance,
    /// Luminous flux leaving the whole emitter, in lumens.
    Flux,
    /// Radiant power leaving the whole emitter, in watts.
    Power,
}

impl IntensityUnit {
    pub const ALL: [Self; 3] = [Self::Luminance, Self::Flux, Self::Power];

    /// The unit symbol, which depends on whether the emitter has area.
    pub fn symbol(self, extent: EmitterExtent) -> &'static str {
        match (self, extent) {
            (Self::Luminance, EmitterExtent::Surface(_)) => "cd/m²",
            (Self::Luminance, _) => "cd",
            (Self::Flux, _) => "lm",
            (Self::Power, _) => "W",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Luminance => "LUMINANCE",
            Self::Flux => "LUMINOUS FLUX",
            Self::Power => "RADIANT POWER",
        }
    }

    /// Whether this unit is meaningful for `extent`. Total power is undefined for an emitter at
    /// infinity, which has no finite extent to spread it over.
    pub fn applies_to(self, extent: EmitterExtent) -> bool {
        self == Self::Luminance || extent != EmitterExtent::Infinite
    }
}

/// The geometry a total-power unit is spread over.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EmitterExtent {
    /// A Lambertian surface of the given world-space area, radiating `Φ = π · A · L`.
    Surface(f32),
    /// A punctual emitter radiating into the full sphere, `Φ = 4π · I`.
    Punctual,
    /// An emitter at infinity, whose total emitted power is not defined.
    Infinite,
}

/// A physical emitter: a spectrum, a tint, and an intensity in a stated unit.
#[derive(Clone, Debug, PartialEq)]
pub struct Illuminant {
    pub spectrum: SpectrumSource,
    /// Linear-sRGB tint applied on top of the spectrum. The spectral backend draws chromaticity
    /// from the spectrum itself, so this contributes the tint's luminance and not its hue; a
    /// saturated tint dims the emitter rather than colouring it.
    pub color: Vec3,
    pub intensity: f32,
    /// Stops applied to `intensity`, matching `UsdLux` `inputs:exposure`.
    pub exposure: f32,
    pub unit: IntensityUnit,
    pub enabled: bool,
}

impl Default for Illuminant {
    fn default() -> Self {
        Self {
            spectrum: SpectrumSource::default(),
            color: Vec3::ONE,
            intensity: 1.0,
            exposure: 0.0,
            unit: IntensityUnit::default(),
            enabled: true,
        }
    }
}

impl Illuminant {
    /// A non-emitting illuminant: the resting state of a material's emission slot.
    pub fn dark() -> Self {
        Self {
            color: Vec3::ZERO,
            intensity: 0.0,
            enabled: false,
            ..Self::default()
        }
    }

    /// An emitter authored directly in the renderer's native quantity, the form importers produce
    /// when a source format carries no unit of its own.
    pub fn luminance(color: Vec3, intensity: f32) -> Self {
        Self {
            color,
            intensity,
            ..Self::default()
        }
    }

    pub fn emits(&self) -> bool {
        self.enabled && self.intensity > 0.0 && self.color.max_element() > 0.0
    }

    /// `intensity` with its exposure stops folded in and its tint's luminance applied, in whatever
    /// unit [`Self::unit`] names. Zero when the illuminant is switched off.
    pub fn stated_intensity(&self) -> f32 {
        if !self.enabled {
            return 0.0;
        }
        (self.intensity * self.exposure.exp2() * luminance(self.color)).max(0.0)
    }

    /// The emitted luminance the spectral film expects: cd/m² for a surface emitter, cd for a
    /// punctual one. `luminous_efficacy` is the lm/W of this illuminant's own resolved spectrum,
    /// and is only consulted for [`IntensityUnit::Power`].
    pub fn emitted_luminance(&self, extent: EmitterExtent, luminous_efficacy: f32) -> f32 {
        let stated = self.stated_intensity();
        let flux = match self.unit {
            IntensityUnit::Luminance => return stated,
            IntensityUnit::Flux => stated,
            IntensityUnit::Power => stated * luminous_efficacy.max(0.0),
        };
        match extent {
            EmitterExtent::Surface(area) if area > 1e-12 => flux / (std::f32::consts::PI * area),
            EmitterExtent::Surface(_) => 0.0,
            EmitterExtent::Punctual => flux / (4.0 * std::f32::consts::PI),
            // Total power cannot be spread over an infinite emitter; authoring it as luminance is
            // the only well-posed reading, so fall back to that rather than silently emitting zero.
            EmitterExtent::Infinite => stated,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::EmitterExtent::Surface;
    use super::*;

    #[test]
    fn luminance_units_ignore_the_emitter_extent() {
        let illuminant = Illuminant::luminance(Vec3::ONE, 40.0);
        for extent in [
            Surface(2.0),
            EmitterExtent::Punctual,
            EmitterExtent::Infinite,
        ] {
            assert!((illuminant.emitted_luminance(extent, 300.0) - 40.0).abs() < 1e-4);
        }
    }

    #[test]
    fn exposure_stops_scale_the_stated_intensity() {
        let mut illuminant = Illuminant::luminance(Vec3::ONE, 10.0);
        illuminant.exposure = 2.0;
        assert!((illuminant.stated_intensity() - 40.0).abs() < 1e-4);
        illuminant.enabled = false;
        assert_eq!(illuminant.stated_intensity(), 0.0);
    }

    /// A Lambertian emitter of area A and luminance L radiates π·A·L lumens, so authoring that
    /// many lumens must reproduce L.
    #[test]
    fn luminous_flux_round_trips_through_a_lambertian_surface() {
        let area = 0.25;
        let expected = 120.0;
        let mut illuminant =
            Illuminant::luminance(Vec3::ONE, std::f32::consts::PI * area * expected);
        illuminant.unit = IntensityUnit::Flux;
        let emitted = illuminant.emitted_luminance(Surface(area), 0.0);
        assert!((emitted - expected).abs() < 1e-2, "{emitted}");
    }

    /// Radiant power is luminous flux scaled by the spectrum's efficacy, so a spectrum with half
    /// the lm/W must come out half as bright for the same wattage.
    #[test]
    fn radiant_power_follows_the_spectrum_efficacy() {
        let mut illuminant = Illuminant::luminance(Vec3::ONE, 10.0);
        illuminant.unit = IntensityUnit::Power;
        let bright = illuminant.emitted_luminance(EmitterExtent::Punctual, 400.0);
        let dim = illuminant.emitted_luminance(EmitterExtent::Punctual, 200.0);
        assert!((bright - 2.0 * dim).abs() < 1e-3);
    }

    #[test]
    fn a_dark_illuminant_does_not_emit() {
        assert!(!Illuminant::dark().emits());
        assert!(Illuminant::luminance(Vec3::ONE, 1.0).emits());
        assert!(!Illuminant::luminance(Vec3::ZERO, 1.0).emits());
    }

    #[test]
    fn total_power_units_are_undefined_at_infinity() {
        assert!(IntensityUnit::Luminance.applies_to(EmitterExtent::Infinite));
        assert!(!IntensityUnit::Flux.applies_to(EmitterExtent::Infinite));
        assert!(IntensityUnit::Flux.applies_to(Surface(1.0)));
    }

    #[test]
    fn spectrum_sources_render_backend_lookup_keys() {
        assert_eq!(SpectrumSource::Named("D65".into()).key(), "D65");
        assert_eq!(SpectrumSource::Blackbody { kelvin: 3200.0 }.key(), "3200K");
        assert_eq!(
            SpectrumSource::Monochromatic { nanometres: 550.0 }.key(),
            "550nm"
        );
    }
}
