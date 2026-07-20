//! Presentation and CPU readback of the spectral film.

use glam::{UVec2, Vec3};
use kiln_rhi::Format;

use super::schedule::SpatialSchedule;

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
    uint width = r.display_width;
    uint height = r.display_height;
    uint bins = r.spectral_bins;
    bool targetIsSrgb = r.target_is_srgb != 0u;
    uint filmW = r.film_width;
    uint filmH = r.film_height;
    uint stride = r.film_stride;
    uint pixelStride = max(r.pixel_stride, 1u);
    uint x = min((uint)i.pos.x, width - 1u);
    uint y = min((uint)i.pos.y, height - 1u);
    // Nearest-neighbour upscale when the film renders below display resolution.
    uint fx = min(x * filmW / width, filmW - 1u);
    uint fy = min(y * filmH / height, filmH - 1u);
    uint phase = (fy % pixelStride) * pixelStride + fx % pixelStride;
    uint sampleCount = r.completed_samples + (phase < r.remaining_phases ? 1u : 0u);

    // Preview the newest phase until every pixel has a sample.
    if (sampleCount == 0u && r.remaining_phases > 0u) {
        uint sampledPhase = r.remaining_phases - 1u;
        uint tileX = (fx / pixelStride) * pixelStride;
        uint tileY = (fy / pixelStride) * pixelStride;
        fx = min(tileX + sampledPhase % pixelStride, filmW - 1u);
        fy = min(tileY + sampledPhase / pixelStride, filmH - 1u);
        sampleCount = r.completed_samples + 1u;
    }
    uint base = (fy * filmW + fx) * stride;
    float inv = 1.0 / max((float)sampleCount, 1.0);
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

/// Resolve one film row to a pre-exposure linear-sRGB colour: `Σ_j cmf[j] ·
/// (bin_j / count)` (the per-path MIS estimator already sums its lanes).
fn resolve_linear(row: &[f32], sample_count: u32, cmf: &[Vec3]) -> Vec3 {
    let inv = 1.0 / sample_count.max(1) as f32;
    cmf.iter()
        .enumerate()
        .map(|(j, c)| *c * (row[j] * inv))
        .sum()
}

/// Tonemap the whole film to RGBA8 for PNG readback, mirroring the blit.
pub fn film_to_rgba8(
    rows: &[f32],
    stride: usize,
    extent: UVec2,
    schedule: SpatialSchedule,
    pass_count: u32,
    cmf: &[Vec3],
) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(rows.len() / stride * 4);
    let coordinates = (0..extent.y).flat_map(|y| (0..extent.x).map(move |x| (x, y)));
    for ((x, y), row) in coordinates.zip(rows.chunks_exact(stride)) {
        let sample_count = schedule.sample_count_for_pixel(pass_count, x, y);
        let lin = resolve_linear(row, sample_count, cmf);
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
pub fn film_to_bands(
    rows: &[f32],
    stride: usize,
    extent: UVec2,
    schedule: SpatialSchedule,
    pass_count: u32,
) -> Vec<f32> {
    let bins = stride;
    let mut out = Vec::with_capacity(rows.len() / stride * bins);
    let coordinates = (0..extent.y).flat_map(|y| (0..extent.x).map(move |x| (x, y)));
    for ((x, y), row) in coordinates.zip(rows.chunks_exact(stride)) {
        let sample_count = schedule.sample_count_for_pixel(pass_count, x, y);
        let inv = 1.0 / sample_count.max(1) as f32;
        out.extend(row.iter().map(|&b| b * inv));
    }
    out
}
