//! Emission spectra: the lighting half of the spectrum module.
//!
//! Mirrors the illuminant pipeline of Christoph Peters' spectral path tracer
//! (<https://momentsingraphics.de/SpectralRendering3Results.html>;
//! <https://github.com/MomentsInGraphics/path_tracer>, `spectral` branch,
//! `tools/illuminant_spectra.py`): a light's spectral power distribution is
//! baked into the two MIS sampling tables ([`EmissionSpectrum`]). The built-in
//! spectra are CIE illuminant A and arbitrary blackbodies (Planck's law), the
//! CIE D series via the S0/S1/S2 basis (D65/D50), CIE E and monochromatic
//! lines, the CIE F series FL2/FL7/FL11, and a loader for LSPDD lamp CSVs
//! (CC BY-NC-ND, loaded from disk rather than vendored).

#[cfg(test)]
use glam::Vec2;
use glam::{Vec3, Vec4};

mod bake;
mod illuminants;

pub use illuminants::{
    blackbody, d50, d65, fluorescent_fl2, fluorescent_fl7, fluorescent_fl11, illuminant_a,
    illuminant_e, monochromatic,
};

use super::{Error, LAMBDA_MAX, LAMBDA_MIN, Result, interp};
#[cfg(test)]
use super::{XYZ_TO_LINEAR_SRGB, cmf_xyz};
/// A spectral power distribution: piecewise linear over sorted wavelength
/// samples, clamped to the end values outside the sampled range (the
/// `numpy.interp` semantics the reference implementation relies on).
pub struct Spd {
    pub name: String,
    wavelengths: Vec<f32>,
    powers: Vec<f32>,
}

impl Spd {
    fn from_uniform_table(name: &str, first_nm: f32, step_nm: f32, powers: &[f32]) -> Self {
        assert!(
            !powers.is_empty(),
            "an SPD must contain at least one sample"
        );
        assert!(step_nm > 0.0, "an SPD sample interval must be positive");
        Self {
            name: name.to_string(),
            wavelengths: (0..powers.len())
                .map(|i| first_nm + step_nm * i as f32)
                .collect(),
            powers: powers.to_vec(),
        }
    }

    fn from_samples(name: String, mut samples: Vec<(f32, f32)>) -> Result<Self> {
        if samples.len() < 2 {
            return Err(Error::Invalid("an SPD needs at least two samples"));
        }
        samples.sort_by(|a, b| a.0.total_cmp(&b.0));
        if !samples
            .iter()
            .all(|&(nm, power)| nm.is_finite() && power.is_finite() && power >= 0.0)
        {
            return Err(Error::Invalid(
                "SPD contains a non-finite wavelength or invalid power",
            ));
        }
        if !samples.windows(2).all(|pair| pair[0].0 < pair[1].0) {
            return Err(Error::Invalid("SPD contains duplicate wavelengths"));
        }
        let (wavelengths, powers) = samples.into_iter().unzip();
        Ok(Self {
            name,
            wavelengths,
            powers,
        })
    }

    pub fn power(&self, nm: f32) -> f32 {
        interp(&self.wavelengths, &self.powers, nm)
    }

    /// ∫ flux dλ over the spectrum's own sample range (trapezoidal).
    pub fn integral(&self) -> f32 {
        self.wavelengths
            .windows(2)
            .zip(self.powers.windows(2))
            .map(|(wavelengths, powers)| {
                0.5 * (powers[0] + powers[1]) * (wavelengths[1] - wavelengths[0])
            })
            .sum()
    }

    /// CIE 1931 tristimulus of the spectrum (1 nm Riemann sum over CMF support).
    #[cfg(test)]
    pub fn xyz(&self) -> Vec3 {
        let mut xyz = Vec3::ZERO;
        let mut nm = LAMBDA_MIN;
        while nm <= LAMBDA_MAX {
            xyz += cmf_xyz(nm) * self.power(nm);
            nm += 1.0;
        }
        xyz
    }

    /// CIE 1931 chromaticity (x, y).
    #[cfg(test)]
    pub fn chromaticity(&self) -> Vec2 {
        let xyz = self.xyz();
        xyz.truncate() / xyz.element_sum().max(1e-12)
    }

    /// Aggregate linear sRGB of the spectrum (arbitrary absolute scale).
    #[cfg(test)]
    pub fn linear_srgb(&self) -> Vec3 {
        XYZ_TO_LINEAR_SRGB * self.xyz()
    }

    /// Bake the wavelength-importance-sampling table — the port of
    /// `prepare_illuminant_spectrum` from the reference `illuminant_spectra.py`.
    pub fn bake(&self, resolution: usize) -> EmissionSpectrum {
        bake::bake(self, resolution)
    }
}

/// A baked light spectrum, ready for GPU upload. Both tables store the same
/// per-wavelength texel `(phase, wavelength_nm, flux_shape, p_light)`: the phase
/// drives moment reflectance, the wavelength selects the film bin, `flux_shape`
/// scales the emitter into true spectral radiance, and `p_light` is the
/// light-importance pdf used in the spectral-film MIS weight.
pub struct EmissionSpectrum {
    pub name: String,
    /// Aggregate linear sRGB of the spectrum — the light's colour for RGB
    /// pipelines and UI (matches the reference's `total_rgb`).
    pub total_rgb: Vec3,
    /// ∫ flux dλ; the scalar brightness the per-texel weights are relative to.
    pub integral: f32,
    /// Strategy A: inverse-CDF table indexed by a uniform random number — draws
    /// wavelengths in proportion to `flux · rgb_importance`.
    pub texels: Vec<Vec4>,
    /// Strategy B: the same texel sampled on a uniform wavelength grid, indexed
    /// by `(λ − LAMBDA_MIN)/(LAMBDA_MAX − LAMBDA_MIN)` — the MIS partner that
    /// reaches the deep tails the CDF table cannot.
    pub lambda_texels: Vec<Vec4>,
}

impl EmissionSpectrum {
    /// Emitter scalar per nit: multiplying a wavelength sample by
    /// `luminance(rgb) · luminance_scale()` makes the mean sensor contribution
    /// match the luminance of `rgb`, an RGB-pipeline emitter value. Derivation:
    /// the mean texel weight estimates `∫rgb·flux / Z`, and
    /// `total_rgb = mean·∫flux`, so `Y(target)·∫flux / Y(total_rgb)` cancels the
    /// sampling constant Z out of the luminance entirely.
    pub fn luminance_scale(&self) -> f32 {
        self.integral / luminance(self.total_rgb).max(1e-9)
    }

    /// [`Self::luminance_scale`] applied to a target RGB emitter value.
    pub fn emission_scale(&self, rgb_emission: Vec3) -> f32 {
        luminance(rgb_emission) * self.luminance_scale()
    }
}

/// Rec. 709 luminance of a linear-sRGB triple.
pub fn luminance(rgb: Vec3) -> f32 {
    rgb.dot(Vec3::new(0.2126, 0.7152, 0.0722))
}

/// Look up a spectrum by name: `"A"`, `"D50"`, `"D65"`, `"E"`, `"FL2"`, `"FL7"`,
/// `"FL11"`, `"<temp>K"` (blackbody), or `"<wavelength>nm"` (monochromatic).
/// This is the vocabulary scene light descriptions will use.
pub fn named(name: &str) -> Option<Spd> {
    match name {
        "A" => return Some(illuminant_a()),
        "D50" => return Some(d50()),
        "D65" => return Some(d65()),
        "E" => return Some(illuminant_e()),
        "FL2" => return Some(fluorescent_fl2()),
        "FL7" => return Some(fluorescent_fl7()),
        "FL11" => return Some(fluorescent_fl11()),
        _ => {}
    }
    if let Some(kelvin) = name.strip_suffix('K') {
        return kelvin
            .trim()
            .parse::<f32>()
            .ok()
            .filter(|temperature| temperature.is_finite() && *temperature > 0.0)
            .map(blackbody);
    }
    if let Some(nm) = name.strip_suffix("nm") {
        return nm
            .trim()
            .parse::<f32>()
            .ok()
            .filter(|nm| (LAMBDA_MIN..=LAMBDA_MAX).contains(nm))
            .map(monochromatic);
    }
    None
}

/// Load a measured lamp spectrum in the LSPDD CSV format (metadata lines such as
/// `Category: …`, then `wavelength,flux` rows). The data set is CC BY-NC-ND
/// (Roby & Aubé, <https://lspdd.org>) — download it yourself, keep it out of the
/// repository. `examples/spectra/assets/lspdd/` is the version-control-ignored drop
/// folder the CLI resolves bare spectrum names against.
pub fn from_lspdd_csv(path: &std::path::Path) -> Result<Spd> {
    let text = std::fs::read_to_string(path).map_err(|source| Error::Read {
        path: path.to_owned(),
        source,
    })?;

    let mut name_parts = Vec::new();
    let mut samples: Vec<(f32, f32)> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some((key, value)) = line.split_once(':') {
            let value = value.trim();
            if matches!(key.trim(), "Category" | "Brand" | "Model")
                && !value.is_empty()
                && value != "null"
            {
                name_parts.push(value.to_string());
            }
            continue;
        }
        if let Some((nm, flux)) = line.split_once(',')
            && let (Ok(nm), Ok(flux)) = (nm.trim().parse::<f32>(), flux.trim().parse::<f32>())
        {
            samples.push((nm, flux));
        }
    }
    let name = if name_parts.is_empty() {
        path.display().to_string()
    } else {
        name_parts.join(" ")
    };
    Spd::from_samples(name, samples)
}

#[cfg(test)]
mod tests {
    use super::super::DEFAULT_RESOLUTION;
    use super::*;

    fn assert_chromaticity(spd: &Spd, expected: [f32; 2], tolerance: f32) {
        let Vec2 { x, y } = spd.chromaticity();
        assert!(
            (x - expected[0]).abs() < tolerance && (y - expected[1]).abs() < tolerance,
            "{}: chromaticity ({x:.4}, {y:.4}) != expected ({:.4}, {:.4})",
            spd.name,
            expected[0],
            expected[1]
        );
    }

    /// Published CIE chromaticities for the standard illuminants.
    #[test]
    fn standard_illuminant_chromaticities() {
        assert_chromaticity(&illuminant_a(), [0.4476, 0.4074], 2e-3);
        assert_chromaticity(&d65(), [0.3127, 0.3290], 2e-3);
        assert_chromaticity(&d50(), [0.3457, 0.3585], 2e-3);
        assert_chromaticity(&illuminant_e(), [0.3333, 0.3333], 2e-3);
        assert_chromaticity(&fluorescent_fl2(), [0.3721, 0.3751], 3e-3);
        assert_chromaticity(&fluorescent_fl7(), [0.3129, 0.3292], 3e-3);
        assert_chromaticity(&fluorescent_fl11(), [0.3805, 0.3769], 3e-3);
    }

    /// sRGB's white point is D65, so D65 must come out achromatic in linear sRGB.
    #[test]
    fn d65_is_srgb_white() {
        let [r, g, b] = d65().linear_srgb().to_array();
        assert!((r / g - 1.0).abs() < 0.02, "r/g = {}", r / g);
        assert!((b / g - 1.0).abs() < 0.02, "b/g = {}", b / g);
    }

    #[test]
    fn baked_spectrum_is_consistent() {
        for spd in [d65(), illuminant_a(), fluorescent_fl11()] {
            let baked = spd.bake(DEFAULT_RESOLUTION);
            assert_eq!(baked.texels.len(), DEFAULT_RESOLUTION);
            // Texel = (phase, wavelength_nm, flux_shape, p_light).
            let mut last_phase = f32::NEG_INFINITY;
            for texel in &baked.texels {
                assert!(texel.is_finite(), "{}", baked.name);
                assert!((-std::f32::consts::PI..=1e-5).contains(&texel.x));
                assert!(texel.x >= last_phase, "{}: phases must ascend", baked.name);
                assert!(
                    (LAMBDA_MIN..=LAMBDA_MAX).contains(&texel.y),
                    "{}: λ in range",
                    baked.name
                );
                assert!(texel.z >= 0.0, "{}: flux_shape non-negative", baked.name);
                assert!(texel.w >= 0.0, "{}: p_light non-negative", baked.name);
                last_phase = texel.x;
            }
            // The stratified estimate of the spectrum's colour must agree with
            // direct integration (identical pipelines up to discretisation).
            let direct = spd.linear_srgb();
            let direct_chroma = direct / direct.element_sum();
            let baked_chroma = baked.total_rgb / baked.total_rgb.element_sum();
            for (a, b) in direct_chroma.to_array().iter().zip(baked_chroma.to_array()) {
                assert!((a - b).abs() < 0.01, "{}: {a} vs {b}", baked.name);
            }
        }
    }

    #[test]
    fn blackbody_at_d65_cct_is_near_d65() {
        let bb = blackbody(6504.0).chromaticity();
        let d = d65().chromaticity();
        // Planckian locus vs daylight locus: close but not equal.
        assert!((bb - d).abs().max_element() < 0.012);
    }

    #[test]
    fn named_lookup_and_lspdd_loader() {
        assert!(named("D65").is_some());
        assert!(named("3200K").is_some());
        assert!(named("550nm").is_some());
        assert!(named("0K").is_none());
        assert!(named("NaNK").is_none());
        assert!(named("900nm").is_none());
        assert!(named("nonsense").is_none());

        let path = std::env::temp_dir().join("kiln_lspdd_test.csv");
        std::fs::write(
            &path,
            "Category : LED\nBrand : Test\nModel : null\n380.0,0.1\n400.0,0.5\n390.0,0.3\n",
        )
        .unwrap();
        let spd = from_lspdd_csv(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(spd.name, "LED Test");
        assert_eq!(spd.wavelengths, vec![380.0, 390.0, 400.0]);
        assert!((spd.power(395.0) - 0.4).abs() < 1e-6);
    }
}
