//! GPU sampling-table construction for an emission spectrum.

use glam::{DVec3, Vec4};

use super::super::{LAMBDA_MAX, LAMBDA_MIN, cmf_linear_srgb, rgb_importance, wavelength_to_phase};
use super::{EmissionSpectrum, Spd};

const CDF_STEP_NM: f32 = 0.1;

pub(super) fn bake(spd: &Spd, resolution: usize) -> EmissionSpectrum {
    assert!(resolution > 0, "spectrum table resolution must be non-zero");

    let sample_count = ((LAMBDA_MAX - LAMBDA_MIN) / CDF_STEP_NM) as usize + 1;
    let wavelength = |index: usize| LAMBDA_MIN + CDF_STEP_NM * index as f32;

    let importance_integral: f64 = (0..sample_count)
        .map(|index| {
            let trapezoid_weight = if index == 0 || index + 1 == sample_count {
                0.5
            } else {
                1.0
            };
            f64::from(rgb_importance(wavelength(index)) * CDF_STEP_NM * trapezoid_weight)
        })
        .sum();
    let importance_normalization = (1.0 / importance_integral) as f32;

    // Joint density proportional to flux × sensor importance.
    let mut cumulative = Vec::with_capacity(sample_count);
    let mut total = 0.0f64;
    for index in 0..sample_count {
        let nm = wavelength(index);
        total += f64::from(spd.power(nm) * rgb_importance(nm) * importance_normalization);
        cumulative.push(total as f32);
    }
    let total = cumulative.last().copied().unwrap_or_default().max(1e-20);
    let inverse_density_integral = 1.0 / (total * CDF_STEP_NM);

    // Both strategies store (phase, wavelength, flux shape, light PDF).
    let texel_at = |nm: f32| {
        let power = spd.power(nm);
        Vec4::new(
            wavelength_to_phase(nm),
            nm,
            power * inverse_density_integral,
            power * rgb_importance(nm) * importance_normalization * inverse_density_integral,
        )
    };

    let mut sensor_sum = DVec3::ZERO;
    let texels = (0..resolution)
        .map(|bin| {
            let target = (bin as f32 + 0.5) / resolution as f32 * total;
            let index = cumulative
                .partition_point(|&value| value < target)
                .min(sample_count - 1);
            let nm = if index == 0 {
                wavelength(0)
            } else {
                let span = (cumulative[index] - cumulative[index - 1]).max(1e-20);
                wavelength(index - 1) + CDF_STEP_NM * (target - cumulative[index - 1]) / span
            };

            let density = (rgb_importance(nm) * importance_normalization).max(1e-12);
            sensor_sum += (cmf_linear_srgb(nm) / density).as_dvec3();
            texel_at(nm)
        })
        .collect();

    // The MIS partner samples the full sensor range uniformly, covering the
    // wavelengths whose light-importance probability approaches zero.
    let lambda_texels = (0..resolution)
        .map(|bin| {
            let unit = (bin as f32 + 0.5) / resolution as f32;
            texel_at(LAMBDA_MIN + unit * (LAMBDA_MAX - LAMBDA_MIN))
        })
        .collect();

    let integral = spd.integral();
    EmissionSpectrum {
        name: spd.name.clone(),
        total_rgb: (sensor_sum / resolution as f64).as_vec3() * integral,
        integral,
        texels,
        lambda_texels,
    }
}
