static const float PI = 3.14159265358979;
static const float INV_PI = 0.31830988618379;

float3 cosine_sample_hemisphere(float2 u)
{
    float r = sqrt(u.x);
    float phi = 2.0 * PI * u.y;
    return float3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u.x)));
}

void tangent_basis(float3 n, out float3 t, out float3 b)
{
    float3 helper = abs(n.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(0.0, 1.0, 0.0);
    t = normalize(cross(helper, n));
    b = cross(n, t);
}

float power_heuristic(float a, float b)
{
    float a2 = a * a;
    return a2 / max(a2 + b * b, 1e-8);
}

// Four hero wavelengths drawn by two MIS strategies: `n_light` lanes from the
// light-importance CDF table (rotated by 1/n_light over the CDF), `n_uniform`
// lanes uniform in wavelength (rotated by 1/n_uniform over [λ_min, λ_max]). Both
// tables are indexed by a stratum u in [0,1) and share the texel layout
// (phase, λ, flux_shape, p_light), so a lane is one fetch regardless of
// strategy. The path carries only the two strata `xi`, `xiU` and refetches per
// lane, keeping wavelength data out of the bounce loop's live set.
static const float SPECTRAL_LAMBDA_MIN = 360.0;
static const float SPECTRAL_LAMBDA_MAX = 830.0;

uint lane_table_index(TraceRoot* r, uint k, float xi, float xiU)
{
    uint len = max(r.spectrum_len, 1u);
    uint nLight = r.light_lane_count;
    if (k < nLight) {
        float u = frac(xi + (float)k / (float)nLight);
        return min((uint)(u * (float)len), len - 1u);
    }
    uint nUniform = max(r.uniform_lane_count, 1u);
    float u = frac(xiU + (float)(k - nLight) / (float)nUniform);
    return min((uint)(u * (float)len), len - 1u);
}

uint4 sample_spectral_indices(TraceRoot* r, float xi, float xiU)
{
    uint4 indices;
    [ForceUnroll]
    for (uint k = 0u; k < 4u; k++) {
        indices[k] = lane_table_index(r, k, xi, xiU);
    }
    return indices;
}

float4 fetch_lane(TraceRoot* r, uint k, uint4 indices)
{
    return k < r.light_lane_count ? r.spectrum[indices[k]] : r.lambda[indices[k]];
}

// Pre-evaluated maximum-entropy reflectance.
float4 lookup_reflectance(TraceRoot* r, uint materialId, uint4 indices)
{
    uint len = max(r.spectrum_len, 1u);
    uint base = materialId * (2u * len);
    uint nLight = r.light_lane_count;
    float4 rho;
    [ForceUnroll]
    for (uint k = 0u; k < 4u; k++) {
        uint tableOffset = k < nLight ? 0u : len;
        rho[k] = r.reflectance[base + tableOffset + indices[k]];
    }
    return rho;
}

// Splat one path's hero-wavelength radiances into the pixel's spectral film bins
// with the balance-heuristic MIS weight: each lane contributes the true band
// radiance L(λ) = flux_shape·radiance, divided by `n_light·p_light + n_uniform·
// p_uniform`. The uniform term keeps the denominator bounded below in the deep
// tails (where p_light → 0), so the edge-bin blow-up is gone. Single thread owns
// the pixel this frame and frames are barrier-serialised, so the += is atomic-free.
void splat_spectral(TraceRoot* r, uint pixel, uint4 indices, float4 radiance)
{
    uint bins = r.spectral_bins;
    uint stride = bins;
    float nLight = (float)r.light_lane_count;
    float nUniform = (float)r.uniform_lane_count;
    float binWidth = (SPECTRAL_LAMBDA_MAX - SPECTRAL_LAMBDA_MIN) / (float)bins;
    float pUniform = 1.0 / (SPECTRAL_LAMBDA_MAX - SPECTRAL_LAMBDA_MIN);
    [ForceUnroll]
    for (uint k = 0u; k < 4u; k++) {
        float4 texel = fetch_lane(r, k, indices);
        float lambda = texel.y;
        float fluxShape = texel.z;
        float pLight = texel.w;
        float radianceL = fluxShape * radiance[k];
        float denom = max(nLight * pLight + nUniform * pUniform, 1e-20);
        uint bin = min((uint)((lambda - SPECTRAL_LAMBDA_MIN) / binWidth), bins - 1u);
        r.film[pixel * stride + bin] += radianceL / denom;
    }
}
