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
    uint x = min((uint)i.pos.x, width - 1u);
    uint y = min((uint)i.pos.y, height - 1u);
    uint pixel = y * width + x;
    float invSamples = 1.0 / max((float)sampleCount, 1.0);
    float3 c = r.accum[pixel].xyz * invSamples * 0.25;
    c = c / (c + float3(1.0));
    c = pow(max(c, float3(0.0)), float3(1.0 / 2.2));
    return float4(c, 1.0);
}
"#;
