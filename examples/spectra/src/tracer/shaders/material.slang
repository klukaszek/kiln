// Evaluating a material at a hit point.
//
// `GpuMaterial` is what the host authored: constants plus optional texture bindings. `Surface` is
// what the BSDF needs: spectral reflectance and GGX coefficients derived from whatever those
// inputs turned out to be here. Everything texture-shaped lives on this side of that boundary, so
// the integrator never samples a map itself.

static const uint NO_MAP = 0xffffffffu;

struct Surface {
    float4 rho;             // spectral reflectance at the sampled wavelengths
    float alpha;            // GGX roughness^2
    float alpha2;
    float metallic;
    float f0_dielectric;
    float spec_prob;        // probability of sampling the specular lobe
};

float2 flip_v(float2 uv)
{
    // Image rows arrive top-down; UVs run bottom-up.
    return float2(uv.x, 1.0 - uv.y);
}

float4 sample_map(TraceRoot* r, uint binding, float2 uv)
{
    GpuTextureBinding bound = r.texture_bindings[binding];
    Texture2D texture = bound.image;
    SamplerState sampler = bound.sampler;
    return texture.SampleLevel(sampler, flip_v(uv), 0.0);
}

// Scalar maps pack their binding in the low 24 bits and the channel to read in the high byte.
float sample_scalar(TraceRoot* r, uint packed, float2 uv, float fallback)
{
    if (packed == NO_MAP) {
        return fallback;
    }
    float4 texel = sample_map(r, packed & 0x00ffffffu, uv);
    return texel[packed >> 24];
}

Surface evaluate_surface(TraceRoot* r, uint materialId, float2 uv, uint4 spectralIndices)
{
    GpuMaterial m = r.materials[materialId];

    Surface s;
    float3 albedo = m.base_color.rgb;
    if (m.base_color_map == NO_MAP) {
        // No map: the host already fitted this material's constant colour to a spectrum.
        s.rho = lookup_reflectance(r, materialId, spectralIndices);
    } else {
        albedo *= sample_map(r, m.base_color_map, uv).rgb;
        s.rho = texture_reflectance(r, albedo, spectralIndices);
    }

    // Both may vary per texel, so the coefficients that depend on them are derived here rather
    // than baked into the material record.
    float roughness = max(sample_scalar(r, m.roughness_map, uv, m.roughness), MIN_GGX_ROUGHNESS);
    s.metallic = saturate(sample_scalar(r, m.metallic_map, uv, m.metallic));
    s.alpha = roughness * roughness;
    s.alpha2 = s.alpha * s.alpha;

    float f0_root = (m.ior - 1.0) / (m.ior + 1.0);
    s.f0_dielectric = f0_root * f0_root;

    // Lobe weighting is a photometric decision, so it reads the RGB albedo rather than the
    // spectral reflectance, matching how the host weighted untextured materials.
    float diffuse_luminance = dot(albedo, REC709_LUMA);
    float f0_luminance = lerp(s.f0_dielectric, diffuse_luminance, s.metallic);
    float diffuse_weight = (1.0 - s.metallic) * diffuse_luminance;
    s.spec_prob = clamp(f0_luminance / max(f0_luminance + diffuse_weight, 1e-4), 0.05, 0.95);
    return s;
}
