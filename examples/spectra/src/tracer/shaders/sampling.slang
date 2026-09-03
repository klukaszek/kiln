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

static const float SPECTRAL_LAMBDA_MIN = 360.0;
static const float SPECTRAL_LAMBDA_MAX = 830.0;

uint lane_table_index(uint k, float xi)
{
    float u = (xi + (float)k) / (float)WAVELENGTH_LANE_COUNT;
    return min((uint)(u * (float)SPECTRUM_LEN), SPECTRUM_LEN - 1u);
}

uint4 sample_spectral_indices(float xi)
{
    uint4 indices;
    [ForceUnroll]
    for (uint k = 0u; k < 4u; k++) {
        indices[k] = lane_table_index(k, xi);
    }
    return indices;
}

float4 fetch_lane(TraceRoot* r, uint k, uint4 indices)
{
    return r.spectrum[indices[k]];
}

// Pre-evaluated maximum-entropy reflectance.
float4 lookup_reflectance(TraceRoot* r, uint materialId, uint4 indices)
{
    uint len = SPECTRUM_LEN;
    uint base = materialId * len;
    float4 rho;
    [ForceUnroll]
    for (uint k = 0u; k < 4u; k++) {
        rho[k] = r.reflectance[base + indices[k]];
    }
    return rho;
}

float4 lookup_texture_basis(TraceRoot* r, uint basisId, uint4 indices)
{
    uint len = SPECTRUM_LEN;
    uint base = basisId * len;
    float4 rho;
    [ForceUnroll]
    for (uint k = 0u; k < 4u; k++) {
        rho[k] = r.texture_basis[base + indices[k]];
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

void splat_spectral(TraceRoot* r, uint pixel, uint4 indices, float4 radiance)
{
    if (r.settings.w == 0u) {
        float3 rgb = float3(0.0);
        [ForceUnroll]
        for (uint k = 0u; k < 4u; k++) {
            rgb += r.sensor_spectrum[indices[k]].xyz * radiance[k];
        }
        uint base = pixel * 4u;
        float scale = 1.0 / (float)WAVELENGTH_LANE_COUNT;
        r.film[base + 0u] += rgb.x * scale;
        r.film[base + 1u] += rgb.y * scale;
        r.film[base + 2u] += rgb.z * scale;
        return;
    }

    float binWidth = (SPECTRAL_LAMBDA_MAX - SPECTRAL_LAMBDA_MIN) / (float)SPECTRAL_BINS;
    [ForceUnroll]
    for (uint k = 0u; k < 4u; k++) {
        float4 texel = fetch_lane(r, k, indices);
        float lambda = texel.y;
        float fluxShape = texel.z;
        float pLight = texel.w;
        float radianceL = fluxShape * radiance[k];
        float denom = (float)WAVELENGTH_LANE_COUNT * pLight;
        uint bin = min((uint)((lambda - SPECTRAL_LAMBDA_MIN) / binWidth), SPECTRAL_BINS - 1u);
        r.film[pixel * SPECTRAL_BINS + bin] += radianceL / denom;
    }
}
