//! Spectral data for the path tracer: emission spectra (the lighting half) and
//! the moment-based reflectance representation (the material half). The two
//! share the colorimetry tables and the wavelength→phase warp, which is why
//! they live in one file.
//!
//! # Emission
//!
//! Mirrors the illuminant pipeline of Christoph Peters' spectral path tracer
//! (<https://momentsingraphics.de/SpectralRendering3Results.html>;
//! <https://github.com/MomentsInGraphics/path_tracer>, `spectral` branch,
//! `tools/illuminant_spectra.py`): a light's spectral power distribution is baked
//! into an inverse-CDF table that importance samples wavelengths proportionally to
//! `flux(λ) · Σ|sRGB CMF(λ)|`. Each table entry stores the ready-made Monte Carlo
//! weight — the linear-sRGB colour-matching triple divided by the sampling
//! density, the flux factor cancelling analytically against the emitter in the
//! integrand — and the wavelength pre-warped into the phase domain that the
//! moment-based reflectance evaluation consumes. At render time, turning a
//! uniform random number into a wavelength plus its sensor weight is one fetch.
//!
//! Built-in spectra:
//! - CIE standard illuminant A and arbitrary-temperature blackbodies (Planck's law);
//! - the CIE D series via the S0/S1/S2 basis (D65/D50 presets);
//! - CIE E (equal energy) and monochromatic lines;
//! - CIE F series FL2/FL7/FL11, one per fluorescent class — the spiky spectra
//!   where spectral transport visibly beats RGB;
//! - a loader for LSPDD (<https://lspdd.org>) lamp-measurement CSVs, the data set
//!   Peters ships. LSPDD data is CC BY-NC-ND (Roby & Aubé), so it is loaded from
//!   disk at run time rather than vendored into this repository.
//!
//! # Reflectance
//!
//! Materials carry a bounded reflectance spectrum represented by three real
//! trigonometric moments, reconstructed via the maximum-entropy spectral
//! estimate (Peters et al., "Using Moments to Represent Bounded Signals for
//! Spectral Rendering", SIGGRAPH 2019). [`fit_reflectance`] solves for moments
//! whose spectrum is a D65 metamer of a target linear-sRGB albedo, then runs
//! the `prep_reflectance` stage (a CPU port of the reference `spectra.glsl`) so
//! the shader only evaluates a three-coefficient Fourier series per hit —
//! possible because our albedos are flat colours rather than textures.
//!
//! Data provenance: CIE 1931 2° colour-matching functions at their native 5 nm
//! grid (the `ciexyz31.csv` from Peters' repository); D-series basis and F-series
//! tables as reproduced by colour-science (BSD-3); wavelength→phase warp and
//! moment algorithms from Peters' repository (BSD-3, © Christoph Peters).

use anyhow::Context;
use glam::{DVec3, Mat3, Vec2, Vec3, Vec4};

/// CMF support; spectra are integrated and sampled over this range (nanometres).
pub const LAMBDA_MIN: f32 = 360.0;
pub const LAMBDA_MAX: f32 = 830.0;
const CMF_STEP: f32 = 5.0;

/// Default width of a baked emission-spectrum table, matching the reference.
pub const DEFAULT_RESOLUTION: usize = 1024;

#[allow(clippy::excessive_precision)] // reference values, kept verbatim
const XYZ_TO_LINEAR_SRGB: Mat3 = Mat3::from_cols(
    Vec3::new(3.240_625_5, -0.968_930_7, 0.055_710_1),
    Vec3::new(-1.537_208_0, 1.875_756_1, -0.204_021_1),
    Vec3::new(-0.498_628_6, 0.041_517_5, 1.056_995_9),
);

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

        // Invert the CDF at stratified bin centres; store the sRGB weight (flux
        // cancelled) and the warped wavelength for each.
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

            let density = (rgb_importance(nm) * importance_norm).max(1e-12);
            let weight = cmf_linear_srgb(nm) / density;
            weight_sum += weight.as_dvec3();
            texels.push(weight.extend(wavelength_to_phase(nm)));
        }

        let integral = self.integral();
        let total_rgb = (weight_sum / resolution as f64).as_vec3() * integral;

        EmissionSpectrum {
            name: self.name.clone(),
            total_rgb,
            integral,
            texels,
        }
    }
}

/// A baked light spectrum, ready for GPU upload: an inverse-CDF table indexed by
/// a uniform random number. `xyz` of a texel is the linear-sRGB Monte Carlo
/// weight for the sampled wavelength, `w` is the wavelength warped to the phase
/// in [-π, 0] consumed by moment-based reflectance evaluation.
pub struct EmissionSpectrum {
    pub name: String,
    /// Aggregate linear sRGB of the spectrum — the light's colour for RGB
    /// pipelines and UI (matches the reference's `total_rgb`).
    pub total_rgb: Vec3,
    /// ∫ flux dλ; the scalar brightness the per-texel weights are relative to.
    pub integral: f32,
    pub texels: Vec<Vec4>,
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
/// repository.
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

// ---------------------------------------------------------------------------
// Colorimetry helpers
// ---------------------------------------------------------------------------

/// Wavelength → phase in [-π, 0], the XYZ warp shared between baked emission
/// spectra and the moment-based reflectance representation. Both sides must use
/// this exact table or reflectance lookups land on the wrong wavelengths.
pub fn wavelength_to_phase(nm: f32) -> f32 {
    let pos = ((nm - LAMBDA_MIN) / CMF_STEP).clamp(0.0, 94.0);
    let i = (pos as usize).min(93);
    let t = pos - i as f32;
    WAVELENGTH_WARP_PHASES[i] + t * (WAVELENGTH_WARP_PHASES[i + 1] - WAVELENGTH_WARP_PHASES[i])
}

/// CIE 1931 CMF triple at `nm`, treated as piecewise constant over its 5 nm
/// bins, as the reference does.
fn cmf_xyz(nm: f32) -> Vec3 {
    let i = ((nm - LAMBDA_MIN) / CMF_STEP).round().clamp(0.0, 94.0) as usize;
    Vec3::from_array(CIE_XYZ_1931[i])
}

fn cmf_linear_srgb(nm: f32) -> Vec3 {
    XYZ_TO_LINEAR_SRGB * cmf_xyz(nm)
}

/// The wavelength-importance factor: Σ|linear-sRGB CMF|.
fn rgb_importance(nm: f32) -> f32 {
    cmf_linear_srgb(nm).abs().element_sum()
}

/// `numpy.interp` semantics: piecewise linear, clamped to end values.
fn interp(xs: &[f32], ys: &[f32], x: f32) -> f32 {
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[xs.len() - 1] {
        return ys[ys.len() - 1];
    }
    let hi = xs.partition_point(|&v| v <= x);
    let (lo, hi) = (hi - 1, hi);
    let t = (x - xs[lo]) / (xs[hi] - xs[lo]).max(1e-12);
    ys[lo] + t * (ys[hi] - ys[lo])
}

// ---------------------------------------------------------------------------
// Moment-based reflectance (CPU port of the reference `spectra.glsl`, f64).
// A reflectance spectrum over the phase domain [-π, 0] is described by three
// real trigonometric moments; the MESE reconstruction turns them into three
// Lagrange multipliers, and evaluation at a phase is a tiny Fourier series.
// The prep step happens here, once per material; shaders only evaluate.
// ---------------------------------------------------------------------------

/// A reflectance spectrum fitted to a target albedo.
pub struct ReflectanceSpectrum {
    /// The three real trigonometric moments (DC term in `[0, 1]`).
    pub trig_moments: [f32; 3],
    /// Lagrange multipliers for shader-side evaluation (`eval_reflectance`).
    pub lagranges: [f32; 3],
    /// Max channel error of the fit's round-trip RGB, for diagnostics.
    pub fit_error: f32,
}

/// Fit three trigonometric moments whose MESE spectrum is a D65 metamer of
/// `target` (linear sRGB, components clamped to [0, 1]). Newton iteration with
/// a numeric Jacobian; flat spectra and the Cornell palette converge in a few
/// steps, saturated colours fall back to the best iterate found.
pub fn fit_reflectance(target: Vec3) -> ReflectanceSpectrum {
    let target = target.as_dvec3().clamp(DVec3::ZERO, DVec3::ONE);

    // Precompute per-nanometre (phase, xyz·D65) samples and the normalization
    // that makes a unit reflectance come out exactly white.
    let d65 = d65();
    let mut samples = Vec::with_capacity(471);
    let mut y_norm = 0.0f64;
    let mut nm = LAMBDA_MIN;
    while nm <= LAMBDA_MAX {
        let cmf_d65 = cmf_xyz(nm).as_dvec3() * d65.power(nm) as f64;
        samples.push((f64::from(wavelength_to_phase(nm)), cmf_d65));
        y_norm += cmf_d65.y;
        nm += 1.0;
    }

    let xyz_to_srgb = XYZ_TO_LINEAR_SRGB.as_dmat3();
    let forward = |moments: [f64; 3]| -> DVec3 {
        let lagranges = prep_reflectance(moments);
        let mut xyz = DVec3::ZERO;
        for (phase, cmf_d65) in &samples {
            xyz += *cmf_d65 * eval_reflectance(*phase, lagranges);
        }
        xyz_to_srgb * (xyz / y_norm)
    };

    let residual_norm = |rgb: DVec3| -> f64 { (rgb - target).length() };

    let mut moments = [(target.element_sum() / 3.0).clamp(0.01, 0.99), 0.0, 0.0];
    let mut best = (residual_norm(forward(moments)), moments);
    for _ in 0..40 {
        let rgb = forward(moments);
        let error = residual_norm(rgb);
        if error < best.0 {
            best = (error, moments);
        }
        if error < 1e-6 {
            break;
        }

        // Numeric Jacobian, central differences.
        const H: f64 = 1e-4;
        let mut jacobian = [[0.0f64; 3]; 3];
        for k in 0..3 {
            let mut hi = moments;
            let mut lo = moments;
            hi[k] += H;
            lo[k] -= H;
            let column = (forward(hi) - forward(lo)) / (2.0 * H);
            for r in 0..3 {
                jacobian[r][k] = column[r];
            }
        }
        let Some(step) = solve_3x3(jacobian, (target - rgb).to_array()) else {
            break;
        };

        // Backtracking line search keeps the iteration from overshooting on
        // saturated targets.
        let mut alpha = 1.0;
        let mut advanced = false;
        for _ in 0..8 {
            let mut candidate = std::array::from_fn(|k| moments[k] + alpha * step[k]);
            candidate[0] = candidate[0].clamp(1e-3, 0.999);
            if residual_norm(forward(candidate)) < error {
                moments = candidate;
                advanced = true;
                break;
            }
            alpha *= 0.5;
        }
        if !advanced {
            break;
        }
    }

    let rgb = forward(moments);
    let error = residual_norm(rgb);
    let (final_error, final_moments) = if error <= best.0 { (error, moments) } else { best };
    let lagranges = prep_reflectance(final_moments);
    ReflectanceSpectrum {
        trig_moments: final_moments.map(|m| m as f32),
        lagranges: lagranges.map(|l| l as f32),
        fit_error: final_error as f32,
    }
}

/// Evaluate the MESE reflectance at a phase in [-π, 0] given Lagrange
/// multipliers from [`prep_reflectance`]. Mirrors the shader-side evaluation.
pub fn eval_reflectance(phase: f64, lagranges: [f64; 3]) -> f64 {
    let (cos_1, sin_1) = ((-phase).cos(), (-phase).sin());
    let cos_2 = cos_1 * cos_1 - sin_1 * sin_1;
    let series = 2.0 * (lagranges[1] * cos_1 + lagranges[2] * cos_2 + 0.5 * lagranges[0]);
    series.atan() * std::f64::consts::FRAC_1_PI + 0.5
}

type Complex = [f64; 2];

fn cmul(a: Complex, b: Complex) -> Complex {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

fn cconj(z: Complex) -> Complex {
    [z[0], -z[1]]
}

fn cscale(s: f64, z: Complex) -> Complex {
    [s * z[0], s * z[1]]
}

fn cadd(a: Complex, b: Complex) -> Complex {
    [a[0] + b[0], a[1] + b[1]]
}

/// `i·z` — multiplication by the imaginary unit.
fn crot(z: Complex) -> Complex {
    [-z[1], z[0]]
}

/// Trigonometric → exponential moments (Eq. 6/7, Peters et al. 2019).
fn trig_to_exp_moments(trig: [f64; 3]) -> [Complex; 3] {
    use std::f64::consts::{FRAC_PI_2, PI, TAU};
    let moment_0_phase = PI * trig[0] - FRAC_PI_2;
    let mut e0 = cscale(1.0 / (4.0 * PI), [moment_0_phase.cos(), moment_0_phase.sin()]);
    let e1 = cscale(trig[1] * TAU, crot(e0));
    let e2 = cadd(cscale(trig[2] * TAU, crot(e0)), cscale(trig[1] * PI, crot(e1)));
    e0 = cscale(2.0, e0);
    [e0, e1, e2]
}

/// Levinson's algorithm with biasing for a 3×3 complex Toeplitz system
/// (Alg. 2 of Peters et al., "Spectral mollification..."; line-for-line port).
fn levinson_3_biased(first_column: &mut [Complex; 3]) -> [Complex; 3] {
    let mut one_minus_bias = 0.9999;
    let mut corrected_factor = 1.0 / (1.0 - one_minus_bias * one_minus_bias);
    let mut solution = [[0.0; 2]; 3];
    solution[0] = [1.0 / first_column[0][0], 0.0];

    let mut scaled_center = [0.0, 0.0];
    let mut dot_product = cadd(cscale(solution[0][0], first_column[1]), scaled_center);
    let mut dot_sq = dot_product[0] * dot_product[0] + dot_product[1] * dot_product[1];
    let mut factor = 1.0 / (1.0 - dot_sq);
    if factor < 0.0 {
        dot_product = cscale(one_minus_bias / dot_sq.sqrt(), dot_product);
        first_column[1] = cscale(
            1.0 / solution[0][0],
            [dot_product[0] - scaled_center[0], dot_product[1] - scaled_center[1]],
        );
        factor = corrected_factor;
        one_minus_bias = 0.0;
        corrected_factor = 1.0;
    }
    let flipped_1 = [solution[0][0], 0.0];
    solution[0] = [factor * solution[0][0], 0.0];
    solution[1] = cscale(factor, cscale(-flipped_1[0], dot_product));

    scaled_center = cmul(solution[1], first_column[1]);
    dot_product = cadd(cscale(solution[0][0], first_column[2]), scaled_center);
    dot_sq = dot_product[0] * dot_product[0] + dot_product[1] * dot_product[1];
    factor = 1.0 / (1.0 - dot_sq);
    if factor < 0.0 {
        dot_product = cscale(one_minus_bias / dot_sq.sqrt(), dot_product);
        first_column[2] = cscale(
            1.0 / solution[0][0],
            [dot_product[0] - scaled_center[0], dot_product[1] - scaled_center[1]],
        );
        factor = corrected_factor;
    }
    let flipped_1 = cconj(solution[1]);
    let flipped_2 = [solution[0][0], 0.0];
    solution[0] = [factor * solution[0][0], 0.0];
    solution[1] = cscale(
        factor,
        cadd(cmul(cscale(-1.0, flipped_1), dot_product), solution[1]),
    );
    solution[2] = cscale(factor, cscale(-flipped_2[0], dot_product));
    solution
}

fn real_autocorrelation_3(signal: [Complex; 3]) -> [Complex; 3] {
    [
        cadd(
            cadd(cmul(signal[0], cconj(signal[0])), cmul(signal[1], cconj(signal[1]))),
            cmul(signal[2], cconj(signal[2])),
        ),
        cadd(cmul(signal[0], cconj(signal[1])), cmul(signal[1], cconj(signal[2]))),
        cmul(signal[0], cconj(signal[2])),
    ]
}

/// First sum of Eq. 10 (Peters et al. 2019).
fn imag_correlation_3(lhs: [Complex; 3], rhs: [Complex; 3]) -> [f64; 3] {
    [
        lhs[0][0] * rhs[0][1] + lhs[0][1] * rhs[0][0]
            + lhs[1][0] * rhs[1][1] + lhs[1][1] * rhs[1][0]
            + lhs[2][0] * rhs[2][1] + lhs[2][1] * rhs[2][0],
        lhs[1][0] * rhs[0][1] + lhs[1][1] * rhs[0][0]
            + lhs[2][0] * rhs[1][1] + lhs[2][1] * rhs[1][0],
        lhs[2][0] * rhs[0][1] + lhs[2][1] * rhs[0][0],
    ]
}

/// Trigonometric moments → Lagrange multipliers (the per-material prep stage;
/// end of Sec. 3.6, Peters et al. 2019, with biasing).
pub fn prep_reflectance(mut trig_moments: [f64; 3]) -> [f64; 3] {
    use std::f64::consts::{PI, TAU};
    trig_moments[0] = trig_moments[0].clamp(1e-4, 0.9999);
    let mut exp_moments = trig_to_exp_moments(trig_moments);
    let mut eval_poly = levinson_3_biased(&mut exp_moments);
    for coeff in &mut eval_poly {
        *coeff = cscale(TAU, *coeff);
    }
    let autocorrelation = real_autocorrelation_3(eval_poly);
    exp_moments[0] = cscale(0.5, exp_moments[0]);
    let normalization = 1.0 / (PI * eval_poly[0][0]);
    let correlation = imag_correlation_3(autocorrelation, exp_moments);
    correlation.map(|c| normalization * c)
}

// Index loops read more like the textbook elimination than split-borrow iterators.
#[allow(clippy::needless_range_loop)]
fn solve_3x3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    // Gaussian elimination with partial pivoting.
    for col in 0..3 {
        let pivot = (col..3).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..3 {
            let f = a[row][col] / a[col][col];
            for k in col..3 {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0.0f64; 3];
    for row in (0..3).rev() {
        let mut sum = b[row];
        for k in row + 1..3 {
            sum -= a[row][k] * x[k];
        }
        x[row] = sum / a[row][row];
    }
    Some(x)
}

// ---------------------------------------------------------------------------
// Data tables (generated from the sources named in the module docs; do not
// hand-edit — values are verified by the chromaticity tests below)
// ---------------------------------------------------------------------------

#[allow(clippy::excessive_precision)]
const CIE_XYZ_1931: [[f32; 3]; 95] = [
    [0.0001299, 3.917e-06, 0.0006061], [0.0002321, 6.965e-06, 0.001086],
    [0.0004149, 1.239e-05, 0.001946], [0.0007416, 2.202e-05, 0.003486],
    [0.001368, 3.9e-05, 0.006450001], [0.002236, 6.4e-05, 0.01054999],
    [0.004243, 0.00012, 0.02005001], [0.00765, 0.000217, 0.03621],
    [0.01431, 0.000396, 0.06785001], [0.02319, 0.00064, 0.1102],
    [0.04351, 0.00121, 0.2074], [0.07763, 0.00218, 0.3713],
    [0.13438, 0.004, 0.6456], [0.21477, 0.0073, 1.0390501],
    [0.2839, 0.0116, 1.3856], [0.3285, 0.01684, 1.62296],
    [0.34828, 0.023, 1.74706], [0.34806, 0.0298, 1.7826],
    [0.3362, 0.038, 1.77211], [0.3187, 0.048, 1.7441],
    [0.2908, 0.06, 1.6692], [0.2511, 0.0739, 1.5281],
    [0.19536, 0.09098, 1.28764], [0.1421, 0.1126, 1.0419],
    [0.09564, 0.13902, 0.8129501], [0.05795001, 0.1693, 0.6162],
    [0.03201, 0.20802, 0.46518], [0.0147, 0.2586, 0.3533],
    [0.0049, 0.323, 0.272], [0.0024, 0.4073, 0.2123],
    [0.0093, 0.503, 0.1582], [0.0291, 0.6082, 0.1117],
    [0.06327, 0.71, 0.07824999], [0.1096, 0.7932, 0.05725001],
    [0.1655, 0.862, 0.04216], [0.2257499, 0.9148501, 0.02984],
    [0.2904, 0.954, 0.0203], [0.3597, 0.9803, 0.0134],
    [0.4334499, 0.9949501, 0.008749999], [0.5120501, 1.0, 0.005749999],
    [0.5945, 0.995, 0.0039], [0.6784, 0.9786, 0.002749999],
    [0.7621, 0.952, 0.0021], [0.8425, 0.9154, 0.0018],
    [0.9163, 0.87, 0.001650001], [0.9786, 0.8163, 0.0014],
    [1.0263, 0.757, 0.0011], [1.0567, 0.6949, 0.001],
    [1.0622, 0.631, 0.0008], [1.0456, 0.5668, 0.0006],
    [1.0026, 0.503, 0.00034], [0.9384, 0.4412, 0.00024],
    [0.8544499, 0.381, 0.00019], [0.7514, 0.321, 0.0001],
    [0.6424, 0.265, 4.999999e-05], [0.5419, 0.217, 3e-05],
    [0.4479, 0.175, 2e-05], [0.3608, 0.1382, 1e-05],
    [0.2835, 0.107, 0.0], [0.2187, 0.0816, 0.0],
    [0.1649, 0.061, 0.0], [0.1212, 0.04458, 0.0],
    [0.0874, 0.032, 0.0], [0.0636, 0.0232, 0.0],
    [0.04677, 0.017, 0.0], [0.0329, 0.01192, 0.0],
    [0.0227, 0.00821, 0.0], [0.01584, 0.005723, 0.0],
    [0.01135916, 0.004102, 0.0], [0.008110916, 0.002929, 0.0],
    [0.005790346, 0.002091, 0.0], [0.004109457, 0.001484, 0.0],
    [0.002899327, 0.001047, 0.0], [0.00204919, 0.00074, 0.0],
    [0.001439971, 0.00052, 0.0], [0.0009999493, 0.0003611, 0.0],
    [0.0006900786, 0.0002492, 0.0], [0.0004760213, 0.0001719, 0.0],
    [0.0003323011, 0.00012, 0.0], [0.0002348261, 8.48e-05, 0.0],
    [0.0001661505, 6e-05, 0.0], [0.000117413, 4.24e-05, 0.0],
    [8.307527e-05, 3e-05, 0.0], [5.870652e-05, 2.12e-05, 0.0],
    [4.150994e-05, 1.499e-05, 0.0], [2.935326e-05, 1.06e-05, 0.0],
    [2.067383e-05, 7.4657e-06, 0.0], [1.455977e-05, 5.2578e-06, 0.0],
    [1.025398e-05, 3.7029e-06, 0.0], [7.221456e-06, 2.6078e-06, 0.0],
    [5.085868e-06, 1.8366e-06, 0.0], [3.581652e-06, 1.2934e-06, 0.0],
    [2.522525e-06, 9.1093e-07, 0.0], [1.776509e-06, 6.4153e-07, 0.0],
    [1.251141e-06, 4.5181e-07, 0.0],
];

#[allow(clippy::excessive_precision)]
const D_SERIES_S0: [f32; 107] = [
    0.04, 3.02, 6.0, 17.8, 29.6, 42.45,
    55.3, 56.3, 57.3, 59.55, 61.8, 61.65,
    61.5, 65.15, 68.8, 66.1, 63.4, 64.6,
    65.8, 80.3, 94.8, 99.8, 104.8, 105.35,
    105.9, 101.35, 96.8, 105.35, 113.9, 119.75,
    125.6, 125.55, 125.5, 123.4, 121.3, 121.3,
    121.3, 117.4, 113.5, 113.3, 113.1, 111.95,
    110.8, 108.65, 106.5, 107.65, 108.8, 107.05,
    105.3, 104.85, 104.4, 102.2, 100.0, 98.0,
    96.0, 95.55, 95.1, 92.1, 89.1, 89.8,
    90.5, 90.4, 90.3, 89.35, 88.4, 86.2,
    84.0, 84.55, 85.1, 83.5, 81.9, 82.25,
    82.6, 83.75, 84.9, 83.1, 81.3, 76.6,
    71.9, 73.1, 74.3, 75.35, 76.4, 69.85,
    63.3, 67.5, 71.7, 74.35, 77.0, 71.1,
    65.2, 56.45, 47.7, 58.15, 68.6, 66.8,
    65.0, 65.5, 66.0, 63.5, 61.0, 57.15,
    53.3, 56.1, 58.9, 60.4, 61.9,
];

#[allow(clippy::excessive_precision)]
const D_SERIES_S1: [f32; 107] = [
    0.02, 2.26, 4.5, 13.45, 22.4, 32.2,
    42.0, 41.3, 40.6, 41.1, 41.6, 39.8,
    38.0, 40.2, 42.4, 40.45, 38.5, 36.75,
    35.0, 39.2, 43.4, 44.85, 46.3, 45.1,
    43.9, 40.5, 37.1, 36.9, 36.7, 36.3,
    35.9, 34.25, 32.6, 30.25, 27.9, 26.1,
    24.3, 22.2, 20.1, 18.15, 16.2, 14.7,
    13.2, 10.9, 8.6, 7.35, 6.1, 5.15,
    4.2, 3.05, 1.9, 0.95, 0.0, -0.8,
    -1.6, -2.55, -3.5, -3.5, -3.5, -4.65,
    -5.8, -6.5, -7.2, -7.9, -8.6, -9.05,
    -9.5, -10.2, -10.9, -10.8, -10.7, -11.35,
    -12.0, -13.0, -14.0, -13.8, -13.6, -12.8,
    -12.0, -12.65, -13.3, -13.1, -12.9, -11.75,
    -10.6, -11.1, -11.6, -11.9, -12.2, -11.2,
    -10.2, -9.0, -7.8, -9.5, -11.2, -10.8,
    -10.4, -10.5, -10.6, -10.15, -9.7, -9.0,
    -8.3, -8.8, -9.3, -9.55, -9.8,
];

#[allow(clippy::excessive_precision)]
const D_SERIES_S2: [f32; 107] = [
    0.0, 1.0, 2.0, 3.0, 4.0, 6.25,
    8.5, 8.15, 7.8, 7.25, 6.7, 6.0,
    5.3, 5.7, 6.1, 4.55, 3.0, 2.1,
    1.2, 0.05, -1.1, -0.8, -0.5, -0.6,
    -0.7, -0.95, -1.2, -1.9, -2.6, -2.75,
    -2.9, -2.85, -2.8, -2.7, -2.6, -2.6,
    -2.6, -2.2, -1.8, -1.65, -1.5, -1.4,
    -1.3, -1.25, -1.2, -1.1, -1.0, -0.75,
    -0.5, -0.4, -0.3, -0.15, 0.0, 0.1,
    0.2, 0.35, 0.5, 1.3, 2.1, 2.65,
    3.2, 3.65, 4.1, 4.4, 4.7, 4.9,
    5.1, 5.9, 6.7, 7.0, 7.3, 7.95,
    8.6, 9.2, 9.8, 10.0, 10.2, 9.25,
    8.3, 8.95, 9.6, 9.05, 8.5, 7.75,
    7.0, 7.3, 7.6, 7.8, 8.0, 7.35,
    6.7, 5.95, 5.2, 6.3, 7.4, 7.1,
    6.8, 6.9, 7.0, 6.7, 6.4, 5.95,
    5.5, 5.8, 6.1, 6.3, 6.5,
];

#[allow(clippy::excessive_precision)]
const CIE_FL2: [f32; 81] = [
    1.18, 1.48, 1.84, 2.15, 3.44, 15.69,
    3.85, 3.74, 4.19, 4.62, 5.06, 34.98,
    11.81, 6.27, 6.63, 6.93, 7.19, 7.4,
    7.54, 7.62, 7.65, 7.62, 7.62, 7.45,
    7.28, 7.15, 7.05, 7.04, 7.16, 7.47,
    8.04, 8.88, 10.01, 24.88, 16.64, 14.59,
    16.16, 17.56, 18.62, 21.47, 22.79, 19.29,
    18.66, 17.73, 16.54, 15.21, 13.8, 12.36,
    10.95, 9.65, 8.4, 7.32, 6.31, 5.43,
    4.68, 4.02, 3.45, 2.96, 2.55, 2.19,
    1.89, 1.64, 1.53, 1.27, 1.1, 0.99,
    0.88, 0.76, 0.68, 0.61, 0.56, 0.54,
    0.51, 0.47, 0.47, 0.43, 0.46, 0.47,
    0.4, 0.33, 0.27,
];

#[allow(clippy::excessive_precision)]
const CIE_FL7: [f32; 81] = [
    2.56, 3.18, 3.84, 4.53, 6.15, 19.37,
    7.37, 7.05, 7.71, 8.41, 9.15, 44.14,
    17.52, 11.35, 12.0, 12.58, 13.08, 13.45,
    13.71, 13.88, 13.95, 13.93, 13.82, 13.64,
    13.43, 13.25, 13.08, 12.93, 12.78, 12.6,
    12.44, 12.33, 12.26, 29.52, 17.05, 12.44,
    12.58, 12.72, 12.83, 15.46, 16.75, 12.83,
    12.67, 12.45, 12.19, 11.89, 11.6, 11.35,
    11.12, 10.95, 10.76, 10.42, 10.11, 10.04,
    10.02, 10.11, 9.87, 8.65, 7.27, 6.44,
    5.83, 5.41, 5.04, 4.57, 4.12, 3.77,
    3.46, 3.08, 2.73, 2.47, 2.25, 2.06,
    1.9, 1.75, 1.62, 1.54, 1.45, 1.32,
    1.17, 0.99, 0.81,
];

#[allow(clippy::excessive_precision)]
const CIE_FL11: [f32; 81] = [
    0.91, 0.63, 0.46, 0.37, 1.29, 12.68,
    1.59, 1.79, 2.46, 3.33, 4.49, 33.94,
    12.13, 6.95, 7.19, 7.12, 6.72, 6.13,
    5.46, 4.79, 5.66, 14.29, 14.96, 8.97,
    4.72, 2.33, 1.47, 1.1, 0.89, 0.83,
    1.18, 4.9, 39.59, 72.84, 32.61, 7.52,
    2.83, 1.96, 1.67, 4.43, 11.28, 14.76,
    12.73, 9.74, 7.33, 9.72, 55.27, 42.58,
    13.18, 13.16, 12.26, 5.11, 2.07, 2.34,
    3.58, 3.01, 2.48, 2.14, 1.54, 1.33,
    1.46, 1.94, 2.0, 1.2, 1.35, 4.1,
    5.58, 2.51, 0.57, 0.27, 0.23, 0.21,
    0.24, 0.24, 0.2, 0.24, 0.32, 0.26,
    0.16, 0.12, 0.09,
];

#[allow(clippy::excessive_precision, clippy::approx_constant)]
const WAVELENGTH_WARP_PHASES: [f32; 95] = [
    -3.141592654, -3.141592654, -3.141592654, -3.141592654, -3.141591857, -3.141590597,
    -3.141590237, -3.141432053, -3.140119041, -3.137863071, -3.133438967, -3.123406739,
    -3.106095749, -3.073470612, -3.024748900, -2.963566246, -2.894461907, -2.819659701,
    -2.741784136, -2.660533432, -2.576526605, -2.490368187, -2.407962868, -2.334138406,
    -2.269339880, -2.213127747, -2.162806279, -2.114787412, -2.065873394, -2.012511127,
    -1.952877310, -1.886377224, -1.813129945, -1.735366957, -1.655108108, -1.573400329,
    -1.490781436, -1.407519056, -1.323814008, -1.239721795, -1.155352390, -1.071041833,
    -0.986956525, -0.903007113, -0.819061538, -0.735505101, -0.653346027, -0.573896987,
    -0.498725202, -0.428534515, -0.363884284, -0.304967687, -0.251925536, -0.205301867,
    -0.165356255, -0.131442191, -0.102998719, -0.079687644, -0.061092401, -0.046554594,
    -0.035419229, -0.027113640, -0.021085743, -0.016716885, -0.013468661, -0.011125245,
    -0.009497032, -0.008356318, -0.007571826, -0.006902676, -0.006366945, -0.005918355,
    -0.005533442, -0.005193920, -0.004886397, -0.004601975, -0.004334090, -0.004077698,
    -0.003829183, -0.003585923, -0.003346286, -0.003109231, -0.002873996, -0.002640047,
    -0.002406990, -0.002174598, -0.001942639, -0.001711031, -0.001479624, -0.001248405,
    -0.001017282, -0.000786134, -0.000557770, -0.000332262, 0.000000000,
];

#[cfg(test)]
mod tests {
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
    fn warp_spans_negative_pi_to_zero_monotonically() {
        assert!((wavelength_to_phase(LAMBDA_MIN) + std::f32::consts::PI).abs() < 1e-5);
        assert!(wavelength_to_phase(LAMBDA_MAX).abs() < 1e-5);
        let mut last = f32::NEG_INFINITY;
        for i in 0..=470 {
            let phase = wavelength_to_phase(360.0 + i as f32);
            assert!(phase >= last);
            last = phase;
        }
    }

    #[test]
    fn baked_spectrum_is_consistent() {
        for spd in [d65(), illuminant_a(), fluorescent_fl11()] {
            let baked = spd.bake(DEFAULT_RESOLUTION);
            assert_eq!(baked.texels.len(), DEFAULT_RESOLUTION);
            let mut last_phase = f32::NEG_INFINITY;
            for texel in &baked.texels {
                assert!(texel.is_finite(), "{}", baked.name);
                assert!((-std::f32::consts::PI..=1e-5).contains(&texel.w));
                assert!(texel.w >= last_phase, "{}: phases must ascend", baked.name);
                last_phase = texel.w;
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

    /// The Cornell palette must round-trip through moments → MESE → D65 → sRGB.
    #[test]
    fn reflectance_fit_round_trips_cornell_palette() {
        let palette = [
            ("white", [0.725, 0.71, 0.68]),
            ("green", [0.14, 0.45, 0.091]),
            ("red", [0.63, 0.065, 0.05]),
            ("grey", [0.5, 0.5, 0.5]),
        ];
        for (name, rgb) in palette {
            let fit = fit_reflectance(Vec3::from_array(rgb));
            assert!(
                fit.fit_error < 0.01,
                "{name}: fit error {} (moments {:?})",
                fit.fit_error,
                fit.trig_moments
            );
            // The reconstructed spectrum must stay a valid reflectance.
            let lagranges = fit.lagranges.map(f64::from);
            for i in 0..=470 {
                let phase = f64::from(wavelength_to_phase(360.0 + i as f32));
                let rho = eval_reflectance(phase, lagranges);
                assert!((0.0..=1.0).contains(&rho), "{name}: rho({phase}) = {rho}");
            }
        }
    }

    /// A flat grey has analytic moments (m = [albedo, 0, 0]); the MESE of those
    /// moments must reproduce the constant spectrum.
    #[test]
    fn flat_spectrum_is_fixed_point() {
        let lagranges = prep_reflectance([0.5, 0.0, 0.0]);
        for phase in [-3.0, -2.0, -1.0, -0.1] {
            let rho = eval_reflectance(phase, lagranges);
            assert!((rho - 0.5).abs() < 1e-3, "rho({phase}) = {rho}");
        }
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
