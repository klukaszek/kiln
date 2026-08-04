//! Analytic and tabulated standard illuminants.

use super::super::data::{CIE_FL2, CIE_FL7, CIE_FL11, D_SERIES_S0, D_SERIES_S1, D_SERIES_S2};
use super::super::{LAMBDA_MAX, LAMBDA_MIN};
use super::Spd;

/// Planck radiator, normalized to 100 at 560 nm (CIE convention).
pub fn blackbody(temperature_k: f32) -> Spd {
    const C2: f32 = 1.4388e7;
    let radiance = |nm: f32| {
        (560.0 / nm).powi(5) * (C2 / (560.0 * temperature_k)).exp_m1()
            / (C2 / (nm * temperature_k)).exp_m1()
    };
    let powers: Vec<_> = (0..=(LAMBDA_MAX - LAMBDA_MIN) as usize)
        .map(|offset| 100.0 * radiance(LAMBDA_MIN + offset as f32))
        .collect();
    Spd::from_uniform_table(
        &format!("blackbody {temperature_k}K"),
        LAMBDA_MIN,
        1.0,
        &powers,
    )
}

pub fn illuminant_a() -> Spd {
    let mut spd = blackbody(2848.0 * 1.4388 / 1.4350);
    spd.name = "A".to_string();
    spd
}

pub fn illuminant_e() -> Spd {
    Spd::from_uniform_table("E", LAMBDA_MIN, LAMBDA_MAX - LAMBDA_MIN, &[100.0, 100.0])
}

/// CIE D-series daylight reconstructed from the S0/S1/S2 basis.
pub fn illuminant_d(cct_kelvin: f32) -> Spd {
    let temperature = f64::from(cct_kelvin).clamp(4000.0, 25_000.0);
    let x = if temperature <= 7000.0 {
        0.244_063 + 0.099_11e3 / temperature + 2.967_8e6 / temperature.powi(2)
            - 4.607_0e9 / temperature.powi(3)
    } else {
        0.237_040 + 0.247_48e3 / temperature + 1.901_8e6 / temperature.powi(2)
            - 2.006_4e9 / temperature.powi(3)
    };
    let y = -3.0 * x * x + 2.87 * x - 0.275;
    let denominator = 0.0241 + 0.2562 * x - 0.7341 * y;
    let m1 = round_to_thousandth((-1.3515 - 1.7703 * x + 5.9114 * y) / denominator);
    let m2 = round_to_thousandth((0.0300 - 31.4424 * x + 30.0717 * y) / denominator);
    let powers: Vec<_> = D_SERIES_S0
        .iter()
        .zip(D_SERIES_S1.iter().zip(D_SERIES_S2.iter()))
        .map(|(&s0, (&s1, &s2))| s0 + m1 as f32 * s1 + m2 as f32 * s2)
        .collect();
    Spd::from_uniform_table(&format!("D {cct_kelvin:.0}K"), 300.0, 5.0, &powers)
}

fn round_to_thousandth(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

pub fn d65() -> Spd {
    let mut spd = illuminant_d(6500.0 * 1.4388 / 1.4380);
    spd.name = "D65".to_string();
    spd
}

pub fn d50() -> Spd {
    let mut spd = illuminant_d(5000.0 * 1.4388 / 1.4380);
    spd.name = "D50".to_string();
    spd
}

pub fn fluorescent_fl2() -> Spd {
    Spd::from_uniform_table("FL2", 380.0, 5.0, &CIE_FL2)
}

pub fn fluorescent_fl7() -> Spd {
    Spd::from_uniform_table("FL7", 380.0, 5.0, &CIE_FL7)
}

pub fn fluorescent_fl11() -> Spd {
    Spd::from_uniform_table("FL11", 380.0, 5.0, &CIE_FL11)
}

/// A two-nanometre triangular monochromatic line.
pub fn monochromatic(nm: f32) -> Spd {
    Spd {
        name: format!("monochromatic {nm}nm"),
        wavelengths: vec![nm - 1.0, nm, nm + 1.0],
        powers: vec![0.0, 100.0, 0.0],
    }
}
