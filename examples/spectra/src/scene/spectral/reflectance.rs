//! Moment-based reflectance: the material half of the spectral data.
//!
//! Materials carry a bounded reflectance spectrum represented by three real
//! trigonometric moments, reconstructed via the maximum-entropy spectral
//! estimate (Peters et al., "Using Moments to Represent Bounded Signals for
//! Spectral Rendering", SIGGRAPH 2019). [`fit_reflectance`] solves for moments
//! whose spectrum is a D65 metamer of a target linear-sRGB albedo, then
//! [`prep_reflectance`] lowers them to Lagrange multipliers used to build GPU
//! lookup tables.
//!
//! The moment algorithms are ported from Peters' repository (BSD-3, © Christoph
//! Peters).

use std::sync::OnceLock;

use glam::{DMat3, DVec3, Vec3};

mod moments;

pub use moments::eval_reflectance;
use moments::prep_reflectance;

use super::{LAMBDA_MAX, LAMBDA_MIN, XYZ_TO_LINEAR_SRGB, cmf_xyz, d65, wavelength_to_phase};

struct ReflectanceColorimetry {
    samples: Vec<(f64, DVec3)>,
    y_normalization: f64,
    xyz_to_srgb: DMat3,
}

impl ReflectanceColorimetry {
    fn new() -> Self {
        let illuminant = d65();
        let mut y_normalization = 0.0;
        let samples = (LAMBDA_MIN as u32..=LAMBDA_MAX as u32)
            .map(|nm| {
                let nm = nm as f32;
                let weighted_cmf = cmf_xyz(nm).as_dvec3() * f64::from(illuminant.power(nm));
                y_normalization += weighted_cmf.y;
                (f64::from(wavelength_to_phase(nm)), weighted_cmf)
            })
            .collect();
        Self {
            samples,
            y_normalization,
            xyz_to_srgb: XYZ_TO_LINEAR_SRGB.as_dmat3(),
        }
    }

    fn evaluate(&self, moments: [f64; 3]) -> DVec3 {
        let lagranges = prep_reflectance(moments);
        let xyz = self
            .samples
            .iter()
            .map(|&(phase, cmf)| cmf * eval_reflectance(phase, lagranges))
            .sum::<DVec3>();
        self.xyz_to_srgb * (xyz / self.y_normalization)
    }
}

fn colorimetry() -> &'static ReflectanceColorimetry {
    static COLORIMETRY: OnceLock<ReflectanceColorimetry> = OnceLock::new();
    COLORIMETRY.get_or_init(ReflectanceColorimetry::new)
}

/// A reflectance spectrum fitted to a target albedo.
pub struct ReflectanceSpectrum {
    /// The three real trigonometric moments (DC term in `[0, 1]`).
    pub trig_moments: [f32; 3],
    /// Lagrange multipliers for spectral evaluation.
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
    let forward = |moments| colorimetry().evaluate(moments);

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
    let (final_error, final_moments) = if error <= best.0 {
        (error, moments)
    } else {
        best
    };
    let lagranges = prep_reflectance(final_moments);
    ReflectanceSpectrum {
        trig_moments: final_moments.map(|m| m as f32),
        lagranges: lagranges.map(|l| l as f32),
        fit_error: final_error as f32,
    }
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
