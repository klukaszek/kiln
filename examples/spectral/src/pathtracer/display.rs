//! Presentation of the spectral film: the fullscreen blit that resolves each
//! pixel's per-bin radiance to linear sRGB (`Σ_j cmf[j]·band`) and tonemaps it,
//! plus the matching CPU resolve/tonemap used for PNG readback and spectral
//! export.

use glam::Vec3;
use kiln_rhi::Format;

pub const SOURCE: &str = /*slang*/
    r#"
struct VOut {
    float4 pos : SV_Position;
};

[shader("vertex")]
VOut displayVs(uint vid : SV_VertexID)
{
    float2 p = float2(float((vid << 1u) & 2u), float(vid & 2u));
    VOut o;
    o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

[shader("fragment")]
float4 displayFs(VOut i, uniform DisplayRoot* r) : SV_Target
{
    uint width = r.dims.x;
    uint height = r.dims.y;
    uint bins = r.dims.z;
    bool targetIsSrgb = r.dims.w != 0u;
    uint filmW = r.film_dims.x;
    uint filmH = r.film_dims.y;
    uint stride = r.film_dims.z;
    uint x = min((uint)i.pos.x, width - 1u);
    uint y = min((uint)i.pos.y, height - 1u);
    // Nearest-neighbour upscale when the film renders below display resolution.
    uint fx = min(x * filmW / width, filmW - 1u);
    uint fy = min(y * filmH / height, filmH - 1u);
    uint base = (fy * filmW + fx) * stride;

    // Resolve the per-pixel spectrum to linear sRGB: Σ_j cmf[j]·band_radiance[j],
    // band_radiance = bin / count (the per-path MIS estimator already sums its
    // lanes). Per-pixel count keeps mid-render and render-scaled pixels exposed.
    float count = r.film[base + bins];
    float inv = 1.0 / max(count, 1.0);
    float3 c = float3(0.0);
    for (uint j = 0u; j < bins; j++) {
        c += r.cmf[j].xyz * (r.film[base + j] * inv);
    }

    c *= 0.25; // exposure
    c = c / (c + float3(1.0));
    if (!targetIsSrgb) {
        c = pow(max(c, float3(0.0)), float3(1.0 / 2.2));
    }
    return float4(c, 1.0);
}
"#;

/// CPU twin of `displayFs`'s tonemap for headless readback: takes one resolved
/// linear-sRGB channel (`Σ_j cmf·band` already summed on the CPU) and applies
/// exposure, Reinhard, and gamma.
pub fn tonemap_linear(linear: f32) -> u8 {
    let exposed = linear * 0.25;
    let reinhard = exposed / (exposed + 1.0);
    let srgb = reinhard.max(0.0).powf(1.0 / 2.2);
    (srgb.clamp(0.0, 1.0) * 255.0).round() as u8
}

pub fn format_is_srgb(format: Format) -> bool {
    matches!(format, Format::R8G8B8A8Srgb | Format::B8G8R8A8Srgb)
}

// ---------------------------------------------------------------------------
// CPU resolve of the spectral film — the headless twin of `displayFs`. Each
// film row is `[bin_0..bin_N, count]` (stride = bins + 1).
// ---------------------------------------------------------------------------

/// Resolve one film row to a pre-exposure linear-sRGB colour: `Σ_j cmf[j] ·
/// (bin_j / count)` (the per-path MIS estimator already sums its lanes).
fn resolve_linear(row: &[f32], bins: usize, cmf: &[Vec3]) -> Vec3 {
    let inv = 1.0 / row[bins].max(1.0);
    cmf.iter()
        .enumerate()
        .map(|(j, c)| *c * (row[j] * inv))
        .sum()
}

/// Tonemap the whole film to RGBA8 for PNG readback, mirroring the blit.
pub fn film_to_rgba8(rows: &[f32], stride: usize, cmf: &[Vec3]) -> Vec<u8> {
    let bins = stride - 1;
    let mut rgba = Vec::with_capacity(rows.len() / stride * 4);
    for row in rows.chunks_exact(stride) {
        let lin = resolve_linear(row, bins, cmf);
        rgba.extend_from_slice(&[
            tonemap_linear(lin.x),
            tonemap_linear(lin.y),
            tonemap_linear(lin.z),
            255,
        ]);
    }
    rgba
}

/// Per-pixel band-integrated spectral radiance `bin / count`, row-major
/// `[height][width][bins]` — the raw spectral capture, flux restored.
pub fn film_to_bands(rows: &[f32], stride: usize) -> Vec<f32> {
    let bins = stride - 1;
    let mut out = Vec::with_capacity(rows.len() / stride * bins);
    for row in rows.chunks_exact(stride) {
        let inv = 1.0 / row[bins].max(1.0);
        out.extend(row[..bins].iter().map(|&b| b * inv));
    }
    out
}
