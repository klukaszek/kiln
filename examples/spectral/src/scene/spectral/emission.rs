//! Emission spectra: the lighting half of the spectral data.
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

use anyhow::Context;
use glam::{DVec3, Vec2, Vec3, Vec4};

use super::data::{CIE_FL2, CIE_FL7, CIE_FL11, D_SERIES_S0, D_SERIES_S1, D_SERIES_S2};
use super::{
    LAMBDA_MAX, LAMBDA_MIN, XYZ_TO_LINEAR_SRGB, cmf_linear_srgb, cmf_xyz, interp, rgb_importance,
    wavelength_to_phase,
};
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
        Self {
            name: name.to_string(),
            wavelengths: (0..powers.len())
                .map(|i| first_nm + step_nm * i as f32)
                .collect(),
            powers: powers.to_vec(),
        }
    }

    pub fn power(&self, nm: f32) -> f32 {
        interp(&self.wavelengths, &self.powers, nm)
    }

    /// ∫ flux dλ over the spectrum's own sample range (trapezoidal).
    pub fn integral(&self) -> f32 {
        let mut sum = 0.0;
        for i in 1..self.wavelengths.len() {
            sum += 0.5
                * (self.powers[i] + self.powers[i - 1])
                * (self.wavelengths[i] - self.wavelengths[i - 1]);
        }
        sum
    }

    /// CIE 1931 tristimulus of the spectrum (1 nm Riemann sum over CMF support).
    #[allow(dead_code)] // colorimetry diagnostics; exercised by the tests
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
    #[allow(dead_code)] // colorimetry diagnostics; exercised by the tests
    pub fn chromaticity(&self) -> Vec2 {
        let xyz = self.xyz();
        xyz.truncate() / xyz.element_sum().max(1e-12)
    }

    /// Aggregate linear sRGB of the spectrum (arbitrary absolute scale).
    #[allow(dead_code)] // colorimetry diagnostics; exercised by the tests
    pub fn linear_srgb(&self) -> Vec3 {
        XYZ_TO_LINEAR_SRGB * self.xyz()
    }

    /// Bake the wavelength-importance-sampling table — the port of
    /// `prepare_illuminant_spectrum` from the reference `illuminant_spectra.py`.
    pub fn bake(&self, resolution: usize) -> EmissionSpectrum {
        // Dense grid over the CMF support for CDF construction.
        const STEP: f32 = 0.1;
        let count = ((LAMBDA_MAX - LAMBDA_MIN) / STEP) as usize + 1;
        let dense_nm = |i: usize| LAMBDA_MIN + STEP * i as f32;

        // Normalize the CMF-derived importance to unit integral first, so the
        // stored weights keep the same scale as the reference.
        let mut rgb_importance_integral = 0.0f64;
        for i in 0..count {
            let edge = i == 0 || i == count - 1;
            rgb_importance_integral +=
                rgb_importance(dense_nm(i)) as f64 * if edge { 0.5 } else { 1.0 } * STEP as f64;
        }
        let importance_norm = 1.0 / rgb_importance_integral as f32;

        // Joint density ∝ flux · rgb_importance, accumulated into a CDF.
        let mut cdf = Vec::with_capacity(count);
        let mut accum = 0.0f64;
        for i in 0..count {
            let nm = dense_nm(i);
            accum += (self.power(nm) * rgb_importance(nm) * importance_norm) as f64;
            cdf.push(accum as f32);
        }
        let total = cdf[count - 1].max(1e-20);

        // Per-wavelength primitives shared by both sampling strategies. The
        // texel is (phase, λ, flux_shape, p_light), where `flux_shape =
        // power/∫g` and `p_light = power·importance·norm/∫g` (∫g ≈ total·STEP,
        // the light-importance pdf's normaliser). Both are densities per nm, so
        // they MIS-combine directly with the uniform pdf 1/Δλ. Picking
        // `flux_shape/p_light = 1/(importance·norm)` makes the uniform-free MIS
        // estimator collapse to the old flux-cancelled splat exactly.
        let inv_int_g = 1.0 / (total * STEP);
        let texel_at = |nm: f32| -> Vec4 {
            let power = self.power(nm);
            Vec4::new(
                wavelength_to_phase(nm),
                nm,
                power * inv_int_g,
                power * rgb_importance(nm) * importance_norm * inv_int_g,
            )
        };

        // Strategy A: invert the CDF at stratified bin centres (light-importance).
        let mut texels = Vec::with_capacity(resolution);
        let mut weight_sum = DVec3::ZERO;
        for bin in 0..resolution {
            let xi = (bin as f32 + 0.5) / resolution as f32;
            let target = xi * total;
            let i = cdf.partition_point(|&c| c < target).min(count - 1);
            let nm = if i == 0 {
                dense_nm(0)
            } else {
                let span = (cdf[i] - cdf[i - 1]).max(1e-20);
                dense_nm(i - 1) + STEP * (target - cdf[i - 1]) / span
            };

            // `weight_sum` still drives `total_rgb` (emission luminance match).
            let density = (rgb_importance(nm) * importance_norm).max(1e-12);
            weight_sum += (cmf_linear_srgb(nm) / density).as_dvec3();
            texels.push(texel_at(nm));
        }

        // Strategy B: a uniform-λ table so the MIS partner can be evaluated at
        // any wavelength — this is what gives the deep tails bounded-variance
        // coverage the CDF table can never reach.
        let lambda_texels = (0..resolution)
            .map(|bin| {
                let nm = LAMBDA_MIN + (bin as f32 + 0.5) / resolution as f32 * (LAMBDA_MAX - LAMBDA_MIN);
                texel_at(nm)
            })
            .collect();

        let integral = self.integral();
        let total_rgb = (weight_sum / resolution as f64).as_vec3() * integral;

        EmissionSpectrum {
            name: self.name.clone(),
            total_rgb,
            integral,
            texels,
            lambda_texels,
        }
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

// ---------------------------------------------------------------------------
// Spectrum constructors
// ---------------------------------------------------------------------------

/// Planck radiator, normalized to 100 at 560 nm (CIE convention).
/// c2 = 1.4388e7 nm·K (ITS-90).
pub fn blackbody(temperature_k: f32) -> Spd {
    const C2: f32 = 1.4388e7;
    let radiance = |nm: f32| {
        (560.0 / nm).powi(5) * (C2 / (560.0 * temperature_k)).exp_m1()
            / (C2 / (nm * temperature_k)).exp_m1()
    };
    let powers: Vec<f32> = (0..=((LAMBDA_MAX - LAMBDA_MIN) as usize))
        .map(|i| 100.0 * radiance(LAMBDA_MIN + i as f32))
        .collect();
    Spd::from_uniform_table(&format!("blackbody {temperature_k}K"), LAMBDA_MIN, 1.0, &powers)
}

/// CIE standard illuminant A: a Planck radiator at 2848 K under the 1931 value
/// of c2 (1.435e7 nm·K), i.e. 2848 · 1.4388/1.4350 ≈ 2855.54 K under ITS-90.
pub fn illuminant_a() -> Spd {
    let mut spd = blackbody(2848.0 * 1.4388 / 1.4350);
    spd.name = "A".to_string();
    spd
}

/// CIE illuminant E (equal energy).
pub fn illuminant_e() -> Spd {
    Spd::from_uniform_table("E", LAMBDA_MIN, LAMBDA_MAX - LAMBDA_MIN, &[100.0, 100.0])
}

/// CIE D-series daylight illuminant for a correlated colour temperature in
/// [4000 K, 25000 K], reconstructed from the S0/S1/S2 basis (CIE 15, with the
/// standard 3-decimal rounding of M1/M2).
pub fn illuminant_d(cct_kelvin: f32) -> Spd {
    let t = (cct_kelvin as f64).clamp(4000.0, 25000.0);
    let x = if t <= 7000.0 {
        0.244063 + 0.09911e3 / t + 2.9678e6 / (t * t) - 4.6070e9 / (t * t * t)
    } else {
        0.237040 + 0.24748e3 / t + 1.9018e6 / (t * t) - 2.0064e9 / (t * t * t)
    };
    let y = -3.000 * x * x + 2.870 * x - 0.275;
    let m = 0.0241 + 0.2562 * x - 0.7341 * y;
    let m1 = ((-1.3515 - 1.7703 * x + 5.9114 * y) / m * 1000.0).round() / 1000.0;
    let m2 = ((0.0300 - 31.4424 * x + 30.0717 * y) / m * 1000.0).round() / 1000.0;
    let powers: Vec<f32> = (0..D_SERIES_S0.len())
        .map(|i| D_SERIES_S0[i] + m1 as f32 * D_SERIES_S1[i] + m2 as f32 * D_SERIES_S2[i])
        .collect();
    Spd::from_uniform_table(&format!("D {cct_kelvin:.0}K"), 300.0, 5.0, &powers)
}

/// CIE standard illuminant D65 (6500 K nominal, c2-corrected to ITS-90).
pub fn d65() -> Spd {
    let mut spd = illuminant_d(6500.0 * 1.4388 / 1.4380);
    spd.name = "D65".to_string();
    spd
}

/// CIE standard illuminant D50 (5000 K nominal, c2-corrected to ITS-90).
pub fn d50() -> Spd {
    let mut spd = illuminant_d(5000.0 * 1.4388 / 1.4380);
    spd.name = "D50".to_string();
    spd
}

/// CIE FL2: cool-white halophosphate fluorescent (standard F-series class).
pub fn fluorescent_fl2() -> Spd {
    Spd::from_uniform_table("FL2", 380.0, 5.0, &CIE_FL2)
}

/// CIE FL7: broadband daylight-simulator fluorescent.
pub fn fluorescent_fl7() -> Spd {
    Spd::from_uniform_table("FL7", 380.0, 5.0, &CIE_FL7)
}

/// CIE FL11: narrowband triband fluorescent — the spikiest standard illuminant.
pub fn fluorescent_fl11() -> Spd {
    Spd::from_uniform_table("FL11", 380.0, 5.0, &CIE_FL11)
}

/// A (near-)monochromatic line at `nm`: a 2 nm-wide triangle, as in the
/// reference data set's synthetic entries.
pub fn monochromatic(nm: f32) -> Spd {
    Spd {
        name: format!("monochromatic {nm}nm"),
        wavelengths: vec![nm - 1.0, nm, nm + 1.0],
        powers: vec![0.0, 100.0, 0.0],
    }
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
        return kelvin.trim().parse().ok().map(blackbody);
    }
    if let Some(nm) = name.strip_suffix("nm") {
        return nm.trim().parse().ok().map(monochromatic);
    }
    None
}

/// Load a measured lamp spectrum in the LSPDD CSV format (metadata lines such as
/// `Category: …`, then `wavelength,flux` rows). The data set is CC BY-NC-ND
/// (Roby & Aubé, <https://lspdd.org>) — download it yourself, keep it out of the
/// repository. `examples/spectral/assets/lspdd/` is the version-control-ignored drop
/// folder the CLI resolves bare spectrum names against.
pub fn from_lspdd_csv(path: &std::path::Path) -> anyhow::Result<Spd> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading LSPDD spectrum {}", path.display()))?;

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
    anyhow::ensure!(
        samples.len() >= 2,
        "no spectral samples in {}",
        path.display()
    );
    samples.sort_by(|a, b| a.0.total_cmp(&b.0));

    let name = if name_parts.is_empty() {
        path.display().to_string()
    } else {
        name_parts.join(" ")
    };
    Ok(Spd {
        name,
        wavelengths: samples.iter().map(|s| s.0).collect(),
        powers: samples.iter().map(|s| s.1).collect(),
    })
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
                assert!((LAMBDA_MIN..=LAMBDA_MAX).contains(&texel.y), "{}: λ in range", baked.name);
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
