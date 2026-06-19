//! Moment-based reflectance: the material half of the spectral data.
//!
//! Materials carry a bounded reflectance spectrum represented by three real
//! trigonometric moments, reconstructed via the maximum-entropy spectral
//! estimate (Peters et al., "Using Moments to Represent Bounded Signals for
//! Spectral Rendering", SIGGRAPH 2019). [`fit_reflectance`] solves for moments
//! whose spectrum is a D65 metamer of a target linear-sRGB albedo, then
//! [`prep_reflectance`] (a CPU port of the reference `spectra.glsl`) lowers them
//! to Lagrange multipliers so the shader only evaluates a three-coefficient
//! Fourier series per hit — possible because our albedos are flat colours.
//!
//! The moment algorithms are ported from Peters' repository (BSD-3, © Christoph
//! Peters).

use glam::{DVec3, Vec3};

use super::{LAMBDA_MAX, LAMBDA_MIN, XYZ_TO_LINEAR_SRGB, cmf_xyz, d65, wavelength_to_phase};
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
