//! Presentation and CPU readback of the spectral film.

use std::fmt::Write as _;

use glam::{UVec2, Vec3};
use kiln_rhi::Format;

use crate::scene::spectral;

use super::schedule::SpatialSchedule;

const BODY: &str = /*slang*/
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
    bool targetIsSrgb = r.target_is_srgb != 0u;
    uint filmW = r.film_width;
    uint filmH = r.film_height;
    uint pixelStride = DISPLAY_PIXEL_STRIDE;
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
    // The film is pixel-major and SPECTRAL_BINS is a multiple of four. Read
    // four adjacent bands at a time so the display pass issues vector loads.
    uint base = (fy * filmW + fx) * DISPLAY_BIN_VECS;
    float inv = 1.0 / max((float)sampleCount, 1.0);
    float3 c = float3(0.0);
    [ForceUnroll]
    for (uint v = 0u; v < DISPLAY_BIN_VECS; v++) {
        float4 bins = r.film[base + v] * inv;
        c += DISPLAY_CMF[v * 4u + 0u].xyz * bins.x;
        c += DISPLAY_CMF[v * 4u + 1u].xyz * bins.y;
        c += DISPLAY_CMF[v * 4u + 2u].xyz * bins.z;
        c += DISPLAY_CMF[v * 4u + 3u].xyz * bins.w;
    }

    c *= 0.25; // exposure
    c = c / (c + float3(1.0));
    if (!targetIsSrgb) {
        c = pow(max(c, float3(0.0)), float3(1.0 / 2.2));
    }
    return float4(c, 1.0);
}
"#;

pub fn source(pixel_stride: u32) -> String {
    assert!(
        spectral::SPECTRAL_BINS.is_multiple_of(4),
        "spectral display requires a multiple of four bins"
    );

    let cmf = spectral::cmf_bins_linear_srgb();
    let mut cmf_source = String::with_capacity(cmf.len() * 48);
    cmf_source.push_str("static const float4 DISPLAY_CMF[DISPLAY_BINS] = {\n");
    for color in cmf {
        writeln!(
            cmf_source,
            "    float4({}, {}, {}, 0.0),",
            color.x, color.y, color.z
        )
        .expect("writing to a String cannot fail");
    }
    cmf_source.push_str("};\n");

    format!(
        "static const uint DISPLAY_BINS = {}u;\n\
         static const uint DISPLAY_BIN_VECS = {}u;\n\
         static const uint DISPLAY_PIXEL_STRIDE = {pixel_stride}u;\n\n\
         {cmf_source}\n{BODY}",
        spectral::SPECTRAL_BINS,
        spectral::SPECTRAL_BINS / 4,
    )
}

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
