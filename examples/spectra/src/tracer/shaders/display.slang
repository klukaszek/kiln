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
    uint pixelStride = r.pixel_stride;
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
    float inv = 1.0 / max((float)sampleCount, 1.0);
    uint pixel = fy * filmW + fx;
    float3 c;
    if (r.spectral_capture == 0u) {
        c = r.film[pixel].xyz * inv;
    } else {
        uint base = pixel * SPECTRAL_BIN_VECS;
        c = float3(0.0);
        [ForceUnroll]
        for (uint v = 0u; v < SPECTRAL_BIN_VECS; v++) {
            float4 bins = r.film[base + v] * inv;
            c += r.cmf[v * 4u + 0u].xyz * bins.x;
            c += r.cmf[v * 4u + 1u].xyz * bins.y;
            c += r.cmf[v * 4u + 2u].xyz * bins.z;
            c += r.cmf[v * 4u + 3u].xyz * bins.w;
        }
    }

    c *= EXPOSURE;
    c = c / (c + float3(1.0));
    if (!targetIsSrgb) {
        c = pow(max(c, float3(0.0)), float3(1.0 / DISPLAY_GAMMA));
    }
    return float4(c, 1.0);
}
