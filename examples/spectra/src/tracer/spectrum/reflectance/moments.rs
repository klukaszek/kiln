//! Maximum-entropy reconstruction from trigonometric moments.
//!
//! This is the small, fixed-size complex algebra from Peters et al. 2019. A
//! named type keeps the port close to the equations without encoding complex
//! numbers as anonymous two-element arrays.

use std::ops::{Add, Mul, Neg, Sub};

#[derive(Clone, Copy, Default)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    const ZERO: Self = Self { re: 0.0, im: 0.0 };

    const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    const fn real(re: f64) -> Self {
        Self::new(re, 0.0)
    }

    const fn conjugate(self) -> Self {
        Self::new(self.re, -self.im)
    }

    const fn rotate_i(self) -> Self {
        Self::new(-self.im, self.re)
    }

    fn norm_squared(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

impl Add for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl Mul for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul<f64> for Complex {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl Neg for Complex {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.re, -self.im)
    }
}

/// Evaluate the MESE reflectance at a phase in [-π, 0].
pub fn eval_reflectance(phase: f64, lagranges: [f64; 3]) -> f64 {
    let cos_1 = (-phase).cos();
    let sin_1 = (-phase).sin();
    let cos_2 = cos_1 * cos_1 - sin_1 * sin_1;
    let series = lagranges[0] + 2.0 * (lagranges[1] * cos_1 + lagranges[2] * cos_2);
    series.atan() * std::f64::consts::FRAC_1_PI + 0.5
}

/// Trigonometric moments → Lagrange multipliers (the per-material prep stage;
/// end of Sec. 3.6, Peters et al. 2019, with biasing).
pub(super) fn prep_reflectance(mut moments: [f64; 3]) -> [f64; 3] {
    use std::f64::consts::{PI, TAU};

    moments[0] = moments[0].clamp(1e-4, 0.9999);
    let mut exponential = trig_to_exponential(moments);
    let mut polynomial = levinson_3_biased(&mut exponential);
    for coefficient in &mut polynomial {
        *coefficient = *coefficient * TAU;
    }

    let autocorrelation = real_autocorrelation_3(polynomial);
    exponential[0] = exponential[0] * 0.5;
    let normalization = 1.0 / (PI * polynomial[0].re);
    imag_correlation_3(autocorrelation, exponential).map(|value| normalization * value)
}

/// Trigonometric → exponential moments (Eq. 6/7, Peters et al. 2019).
fn trig_to_exponential(trig: [f64; 3]) -> [Complex; 3] {
    use std::f64::consts::{FRAC_PI_2, PI, TAU};

    let phase = PI * trig[0] - FRAC_PI_2;
    let e0 = Complex::new(phase.cos(), phase.sin()) * (1.0 / (4.0 * PI));
    let e1 = e0.rotate_i() * (trig[1] * TAU);
    let e2 = e0.rotate_i() * (trig[2] * TAU) + e1.rotate_i() * (trig[1] * PI);
    [e0 * 2.0, e1, e2]
}

/// Levinson's algorithm with biasing for a 3×3 complex Toeplitz system
/// (Alg. 2 of Peters et al., "Spectral mollification...").
fn levinson_3_biased(first_column: &mut [Complex; 3]) -> [Complex; 3] {
    let mut bias = 0.9999;
    let mut corrected_factor = 1.0 / (1.0 - bias * bias);
    let mut solution = [Complex::ZERO; 3];
    solution[0] = Complex::real(1.0 / first_column[0].re);

    let mut center = Complex::ZERO;
    let mut dot = first_column[1] * solution[0].re + center;
    let mut factor = 1.0 / (1.0 - dot.norm_squared());
    if factor < 0.0 {
        dot = dot * (bias / dot.norm_squared().sqrt());
        first_column[1] = (dot - center) * (1.0 / solution[0].re);
        factor = corrected_factor;
        bias = 0.0;
        corrected_factor = 1.0;
    }
    let flipped_1 = solution[0].re;
    solution[0] = Complex::real(factor * solution[0].re);
    solution[1] = dot * (-flipped_1 * factor);

    center = solution[1] * first_column[1];
    dot = first_column[2] * solution[0].re + center;
    factor = 1.0 / (1.0 - dot.norm_squared());
    if factor < 0.0 {
        dot = dot * (bias / dot.norm_squared().sqrt());
        first_column[2] = (dot - center) * (1.0 / solution[0].re);
        factor = corrected_factor;
    }
    let flipped_1 = solution[1].conjugate();
    let flipped_2 = solution[0].re;
    solution[0] = Complex::real(factor * solution[0].re);
    solution[1] = ((-flipped_1) * dot + solution[1]) * factor;
    solution[2] = dot * (-flipped_2 * factor);
    solution
}

fn real_autocorrelation_3(signal: [Complex; 3]) -> [Complex; 3] {
    [
        signal[0] * signal[0].conjugate()
            + signal[1] * signal[1].conjugate()
            + signal[2] * signal[2].conjugate(),
        signal[0] * signal[1].conjugate() + signal[1] * signal[2].conjugate(),
        signal[0] * signal[2].conjugate(),
    ]
}

/// First sum of Eq. 10 (Peters et al. 2019).
fn imag_correlation_3(lhs: [Complex; 3], rhs: [Complex; 3]) -> [f64; 3] {
    [
        lhs[0].re * rhs[0].im
            + lhs[0].im * rhs[0].re
            + lhs[1].re * rhs[1].im
            + lhs[1].im * rhs[1].re
            + lhs[2].re * rhs[2].im
            + lhs[2].im * rhs[2].re,
        lhs[1].re * rhs[0].im
            + lhs[1].im * rhs[0].re
            + lhs[2].re * rhs[1].im
            + lhs[2].im * rhs[1].re,
        lhs[2].re * rhs[0].im + lhs[2].im * rhs[0].re,
    ]
}
