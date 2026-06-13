//! Presentation of the accumulation buffer: the fullscreen blit shader for the
//! swapchain, and the matching CPU tonemap used for PNG readback.

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
    uint sampleCount = r.dims.z;
    bool targetIsSrgb = r.dims.w != 0u;
    uint x = min((uint)i.pos.x, width - 1u);
    uint y = min((uint)i.pos.y, height - 1u);
    uint pixel = y * width + x;
    float invSamples = 1.0 / max((float)sampleCount, 1.0);
    float3 c = r.accum[pixel].xyz * invSamples * 0.25;
    c = c / (c + float3(1.0));
    if (!targetIsSrgb) {
        c = pow(max(c, float3(0.0)), float3(1.0 / 2.2));
    }
    return float4(c, 1.0);
}
"#;

/// CPU twin of `displayFs` for headless readback: exposure, Reinhard, gamma.
pub fn tonemap_channel(value: f32, sample_count: f32) -> u8 {
    let linear = (value / sample_count) * 0.25;
    let reinhard = linear / (linear + 1.0);
    let srgb = reinhard.max(0.0).powf(1.0 / 2.2);
    (srgb.clamp(0.0, 1.0) * 255.0).round() as u8
}

pub fn format_is_srgb(format: Format) -> bool {
    matches!(format, Format::R8G8B8A8Srgb | Format::B8G8R8A8Srgb)
}
