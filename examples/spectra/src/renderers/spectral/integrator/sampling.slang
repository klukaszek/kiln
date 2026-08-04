static const float PI = 3.14159265358979;
static const float INV_PI = 0.31830988618379;

float3 cosine_sample_hemisphere(float2 u)
{
    float r = sqrt(u.x);
    float phi = 2.0 * PI * u.y;
    return float3(r * cos(phi), r * sin(phi), sqrt(1.0 - u.x));
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
    return a2 / (a2 + b * b);
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
    uint len = SPECTRUM_LEN;
    if (k < LIGHT_LANE_COUNT) {
        float u = frac(xi + (float)k / (float)LIGHT_LANE_COUNT);
        return (uint)(u * (float)len);
    }
    float u = frac(xiU + (float)(k - LIGHT_LANE_COUNT) / (float)UNIFORM_LANE_COUNT);
    return (uint)(u * (float)len);
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
    return k < LIGHT_LANE_COUNT ? r.spectrum[indices[k]] : r.lambda[indices[k]];
}

// Pre-evaluated maximum-entropy reflectance.
float4 lookup_reflectance(TraceRoot* r, uint materialId, uint4 indices)
{
    uint len = SPECTRUM_LEN;
    uint base = materialId * (2u * len);
    float4 rho;
    [ForceUnroll]
    for (uint k = 0u; k < 4u; k++) {
        uint tableOffset = k < LIGHT_LANE_COUNT ? 0u : len;
        rho[k] = r.reflectance[base + tableOffset + indices[k]];
    }
    return rho;
}

float4 lookup_texture_basis(TraceRoot* r, uint basisId, uint4 indices)
{
    uint len = SPECTRUM_LEN;
    uint base = basisId * (2u * len);
    float4 rho;
    [ForceUnroll]
    for (uint k = 0u; k < 4u; k++) {
        uint tableOffset = k < LIGHT_LANE_COUNT ? 0u : len;
        rho[k] = r.texture_basis[base + tableOffset + indices[k]];
    }
    return rho;
}

// Smits-style decomposition into bounded spectra fitted to the seven non-black
// RGB cube corners. The coefficients are non-negative and sum to max(rgb), so
// the reconstructed reflectance remains in [0, 1] while preserving texel hue.
float4 texture_reflectance(TraceRoot* r, float3 rgb, uint4 indices)
{
    static const uint WHITE = 0u;
    static const uint CYAN = 1u;
    static const uint MAGENTA = 2u;
    static const uint YELLOW = 3u;
    static const uint RED = 4u;
    static const uint GREEN = 5u;
    static const uint BLUE = 6u;

    rgb = saturate(rgb);
    if (rgb.r <= rgb.g && rgb.r <= rgb.b) {
        float4 rho = rgb.r * lookup_texture_basis(r, WHITE, indices);
        if (rgb.g <= rgb.b) {
            return rho
                + (rgb.g - rgb.r) * lookup_texture_basis(r, CYAN, indices)
                + (rgb.b - rgb.g) * lookup_texture_basis(r, BLUE, indices);
        }
        return rho
            + (rgb.b - rgb.r) * lookup_texture_basis(r, CYAN, indices)
            + (rgb.g - rgb.b) * lookup_texture_basis(r, GREEN, indices);
    }
    if (rgb.g <= rgb.r && rgb.g <= rgb.b) {
        float4 rho = rgb.g * lookup_texture_basis(r, WHITE, indices);
        if (rgb.r <= rgb.b) {
            return rho
                + (rgb.r - rgb.g) * lookup_texture_basis(r, MAGENTA, indices)
                + (rgb.b - rgb.r) * lookup_texture_basis(r, BLUE, indices);
        }
        return rho
            + (rgb.b - rgb.g) * lookup_texture_basis(r, MAGENTA, indices)
            + (rgb.r - rgb.b) * lookup_texture_basis(r, RED, indices);
    }

    float4 rho = rgb.b * lookup_texture_basis(r, WHITE, indices);
    if (rgb.r <= rgb.g) {
        return rho
            + (rgb.r - rgb.b) * lookup_texture_basis(r, YELLOW, indices)
            + (rgb.g - rgb.r) * lookup_texture_basis(r, GREEN, indices);
    }
    return rho
        + (rgb.g - rgb.b) * lookup_texture_basis(r, YELLOW, indices)
        + (rgb.r - rgb.g) * lookup_texture_basis(r, RED, indices);
}

// Splat one path's hero-wavelength radiances into the pixel's spectral film bins
// with the balance-heuristic MIS weight: each lane contributes the true band
// radiance L(λ) = flux_shape·radiance, divided by `n_light·p_light + n_uniform·
// p_uniform`. The uniform term keeps the denominator bounded below in the deep
// tails (where p_light → 0), so the edge-bin blow-up is gone. Single thread owns
// the pixel this frame and frames are barrier-serialised, so the += is atomic-free.
void splat_spectral(TraceRoot* r, uint pixel, uint4 indices, float4 radiance)
{
    float nLight = (float)LIGHT_LANE_COUNT;
    float nUniform = (float)UNIFORM_LANE_COUNT;
    float binWidth = (SPECTRAL_LAMBDA_MAX - SPECTRAL_LAMBDA_MIN) / (float)SPECTRAL_BINS;
    float pUniform = 1.0 / (SPECTRAL_LAMBDA_MAX - SPECTRAL_LAMBDA_MIN);
    [ForceUnroll]
    for (uint k = 0u; k < 4u; k++) {
        float4 texel = fetch_lane(r, k, indices);
        float lambda = texel.y;
        float fluxShape = texel.z;
        float pLight = texel.w;
        float radianceL = fluxShape * radiance[k];
        float denom = nLight * pLight + nUniform * pUniform;
        uint bin = min((uint)((lambda - SPECTRAL_LAMBDA_MIN) / binWidth), SPECTRAL_BINS - 1u);
        r.film[pixel * SPECTRAL_BINS + bin] += radianceL / denom;
    }
}
