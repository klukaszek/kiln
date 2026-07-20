//! Spectral data for the path tracer, split into the lighting and material
//! halves over a shared colorimetry core:
//!
//! - [`emission`] — SPDs, the baked MIS sampling tables, illuminant constructors.
//! - [`reflectance`] — the Peters moment-based reflectance representation.
//! - [`data`] — the CIE/illuminant constant tables (module-private).
//!
//! This file holds what both halves share: the wavelength range, the spectral
//! film's bin layout + sensor response, and the colorimetry helpers
//! (`wavelength_to_phase`, the CMFs, `interp`). The public API of the submodules
//! is re-exported here, so callers use `spectral::Spd`, `spectral::fit_reflectance`,
//! etc. regardless of which file an item lives in.
//!
//! Data provenance: CIE 1931 2° colour-matching functions at their native 5 nm
//! grid (the `ciexyz31.csv` from Peters' repository); D-series basis and F-series
//! tables as reproduced by colour-science (BSD-3); wavelength→phase warp and
//! moment algorithms from Peters' repository (BSD-3, © Christoph Peters).

use glam::{Mat3, Vec3};

mod data;
mod emission;
mod reflectance;

pub use emission::{EmissionSpectrum, Spd, d65, from_lspdd_csv, named};
pub use reflectance::{ReflectanceSpectrum, eval_reflectance, fit_reflectance};

use data::{CIE_XYZ_1931, WAVELENGTH_WARP_PHASES};
/// CMF support; spectra are integrated and sampled over this range (nanometres).
pub const LAMBDA_MIN: f32 = 360.0;
pub const LAMBDA_MAX: f32 = 830.0;
const CMF_STEP: f32 = 5.0;

/// Default width of a baked emission-spectrum table, matching the reference.
pub const DEFAULT_RESOLUTION: usize = 1024;

/// Number of wavelength bins in the spectral film. The film stores this many
/// per-band radiance estimates per pixel, so it is the spectral-resolution /
/// GPU-memory knob: cost is `width*height*BINS*4` bytes. Bins span
/// [`LAMBDA_MIN`] to [`LAMBDA_MAX`]
/// uniformly in wavelength, matching how a spectrometer reports bands.
pub const SPECTRAL_BINS: usize = 32;

/// Width of one spectral bin, in nanometres.
pub const SPECTRAL_BIN_WIDTH: f32 = (LAMBDA_MAX - LAMBDA_MIN) / SPECTRAL_BINS as f32;

/// Centre wavelength (nm) of spectral bin `j`.
pub fn spectral_bin_center(j: usize) -> f32 {
    LAMBDA_MIN + (j as f32 + 0.5) * SPECTRAL_BIN_WIDTH
}

/// The spectral bin a wavelength falls in (clamped to the valid range).
pub fn spectral_bin_of(nm: f32) -> usize {
    (((nm - LAMBDA_MIN) / SPECTRAL_BIN_WIDTH) as usize).min(SPECTRAL_BINS - 1)
}

/// Mean linear-sRGB colour-matching response over each spectral bin. The film
/// stores band-integrated radiance per bin; the display/readout recovers RGB as
/// `Σ_j cmf_bin[j] · radiance[j]`, so this is the sensor side of the spectral
/// estimator — independent of the light, computed once and shared CPU/GPU.
pub fn cmf_bins_linear_srgb() -> Vec<Vec3> {
    let mut sums = vec![Vec3::ZERO; SPECTRAL_BINS];
    let mut counts = vec![0u32; SPECTRAL_BINS];
    let mut nm = LAMBDA_MIN;
    while nm <= LAMBDA_MAX {
        let j = spectral_bin_of(nm);
        sums[j] += cmf_linear_srgb(nm);
        counts[j] += 1;
        nm += 1.0;
    }
    sums.iter()
        .zip(&counts)
        .map(|(&sum, &n)| sum / n.max(1) as f32)
        .collect()
}

#[allow(clippy::excessive_precision)] // reference values, kept verbatim
const XYZ_TO_LINEAR_SRGB: Mat3 = Mat3::from_cols(
    Vec3::new(3.240_625_5, -0.968_930_7, 0.055_710_1),
    Vec3::new(-1.537_208_0, 1.875_756_1, -0.204_021_1),
    Vec3::new(-0.498_628_6, 0.041_517_5, 1.056_995_9),
);

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
