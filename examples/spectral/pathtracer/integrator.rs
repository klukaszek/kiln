//! The spectral light-transport kernel: geometry helpers, BSDF warps, next-event
//! estimation with multiple importance sampling, and the `traceMain` entry point.
//!
//! Each path traces four stratified wavelengths drawn from the light's baked
//! sampling table (`TraceRoot::spectrum`) — one Sobol' draw rotated by k/4
//! through the inverse CDF (hero-wavelength style; the geometric path is shared
//! because nothing disperses yet). Each fetched texel carries the linear-sRGB
//! sensor weight for its wavelength and the phase in [-π, 0]. Reflectances are
//! moment-based spectra evaluated at those phases from per-material Lagrange
//! multipliers (`GpuMaterial::lagrange_emission.xyz`, prepared on the CPU), so
//! path throughput is a `float4` of reflectance products; the sensor weights
//! colour the contribution once at accumulation. Emitter brightness is the
//! precomputed scalar in `lagrange_emission.w` — the light's flux shape
//! cancelled against the sampling density when the table was baked.
//!
//! Random numbers come from the sampler module's `sample_4d`; dimension groups
//! are allocated as 0 = camera jitter + wavelength, then per bounce one group
//! for light sampling and one for the BSDF direction + Russian roulette.

pub const THREADS_X: u32 = 8;
pub const THREADS_Y: u32 = 8;

const GEOMETRY: &str = /*slang*/
    r#"
float3 normal_for_primitive(Vertex* verts, uint primitiveId)
{
    float3 p0 = verts[primitiveId * 3u + 0u].pos.xyz;
    float3 p1 = verts[primitiveId * 3u + 1u].pos.xyz;
    float3 p2 = verts[primitiveId * 3u + 2u].pos.xyz;
    return normalize(cross(p1 - p0, p2 - p0));
}

float triangle_area(Vertex* verts, uint primitiveId)
{
    float3 p0 = verts[primitiveId * 3u + 0u].pos.xyz;
    float3 p1 = verts[primitiveId * 3u + 1u].pos.xyz;
    float3 p2 = verts[primitiveId * 3u + 2u].pos.xyz;
    return 0.5 * length(cross(p1 - p0, p2 - p0));
}

float3 sample_triangle(Vertex* verts, uint primitiveId, float2 u)
{
    float su0 = sqrt(u.x);
    float b0 = 1.0 - su0;
    float b1 = u.y * su0;
    float b2 = 1.0 - b0 - b1;
    float3 p0 = verts[primitiveId * 3u + 0u].pos.xyz;
    float3 p1 = verts[primitiveId * 3u + 1u].pos.xyz;
    float3 p2 = verts[primitiveId * 3u + 2u].pos.xyz;
    return p0 * b0 + p1 * b1 + p2 * b2;
}

bool any_nonzero(float3 v)
{
    return any(v > float3(0.0));
}
"#;

const SAMPLING: &str = /*slang*/
    r#"
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

// Moment-based reflectance (Peters et al. 2019): evaluate the maximum-entropy
// spectrum at four warped wavelengths at once, given the Lagrange multipliers
// prepared on the CPU from the material's three trigonometric moments.
float4 eval_reflectance(float4 phases, float3 lagranges)
{
    float4 cos1 = cos(-phases);
    float4 sin1 = sin(-phases);
    float4 cos2 = cos1 * cos1 - sin1 * sin1;
    float4 series = 2.0 * (lagranges.y * cos1 + lagranges.z * cos2 + 0.5 * lagranges.x);
    return atan(series) * INV_PI + 0.5;
}

// Four stratified wavelength samples from the light's baked inverse-CDF table:
// one uniform draw, rotated by k/4 (Cranley-Patterson over the CDF). Each texel
// is (linear-sRGB sensor weight, phase).
struct WavePacket {
    float4 phases;
    float3 weights[4];
};

WavePacket sample_wavelengths(TraceRoot* r, float xi)
{
    uint len = max(r.dims1.w, 1u);
    WavePacket wave;
    [ForceUnroll]
    for (uint k = 0u; k < 4u; k++) {
        float u = frac(xi + 0.25 * (float)k);
        float4 texel = r.spectrum[min((uint)(u * (float)len), len - 1u)];
        wave.weights[k] = texel.xyz;
        wave.phases[k] = texel.w;
    }
    return wave;
}

// Average the per-wavelength radiance into the sensor's linear-sRGB response.
float3 resolve_to_srgb(WavePacket wave, float4 radiance)
{
    return 0.25 * (wave.weights[0] * radiance.x + wave.weights[1] * radiance.y +
                   wave.weights[2] * radiance.z + wave.weights[3] * radiance.w);
}

// ---------------------------------------------------------------------------
// BSDF: Lambert diffuse + single-scatter GGX specular (height-correlated
// Smith, VNDF sampling). Per wavelength, Fresnel F0 is the moment-based
// reflectance spectrum for metals and the IOR-derived constant for
// dielectrics, so a copper F0 disperses exactly like a copper albedo would.
// Directions are sampled from a lobe mixture and every evaluation returns the
// full mixture pdf, so NEE/BSDF MIS stays exact for both lobes.
// ---------------------------------------------------------------------------

struct Surface {
    float3 lagranges;   // moment reflectance (albedo / metal F0 tint)
    float alpha;        // GGX roughness (perceptual roughness squared)
    float alpha2;
    float metallic;
    float f0_dielectric;
    float spec_prob;    // wavelength-independent lobe-sampling probability
};

Surface make_surface(GpuMaterial mat)
{
    Surface s;
    s.lagranges = mat.lagrange_emission.xyz;
    // Clamp away the delta-lobe limit: a perfect mirror needs dedicated
    // specular-event handling (MIS weight 1, hero-collapse on dispersion).
    float roughness = clamp(mat.base_roughness.w, 0.045, 1.0);
    s.alpha = roughness * roughness;
    s.alpha2 = s.alpha * s.alpha;
    s.metallic = mat.emission_metallic.w;
    float ior = max(mat.specular_ior.w, 1.0001);
    float f0_root = (ior - 1.0) / (ior + 1.0);
    s.f0_dielectric = f0_root * f0_root;

    float diffuse_lum = dot(mat.base_roughness.xyz, float3(0.2126, 0.7152, 0.0722));
    float f0_lum = lerp(s.f0_dielectric, diffuse_lum, s.metallic);
    float diff_weight = (1.0 - s.metallic) * diffuse_lum;
    s.spec_prob = clamp(f0_lum / max(f0_lum + diff_weight, 1e-4), 0.05, 0.95);
    return s;
}

float ggx_d(float alpha2, float cos_h)
{
    float d = cos_h * cos_h * (alpha2 - 1.0) + 1.0;
    return alpha2 / max(PI * d * d, 1e-8);
}

float smith_g1(float alpha2, float cos_theta)
{
    return 2.0 * cos_theta
        / max(cos_theta + sqrt(alpha2 + (1.0 - alpha2) * cos_theta * cos_theta), 1e-8);
}

// Height-correlated Smith visibility: G2 / (4 cosO cosI).
float ggx_visibility(float alpha2, float cos_o, float cos_i)
{
    float a = cos_i * sqrt(alpha2 + (1.0 - alpha2) * cos_o * cos_o);
    float b = cos_o * sqrt(alpha2 + (1.0 - alpha2) * cos_i * cos_i);
    return 0.5 / max(a + b, 1e-8);
}

// pdf of wi under VNDF half-vector sampling: G1(wo) D / (4 cosO).
float ggx_sample_pdf(float alpha2, float cos_o, float cos_h)
{
    return smith_g1(alpha2, cos_o) * ggx_d(alpha2, cos_h) / max(4.0 * cos_o, 1e-8);
}

// Heitz 2018: sample the GGX distribution of visible normals (tangent space).
float3 sample_ggx_vndf(float3 wo_local, float alpha, float2 u)
{
    float3 vh = normalize(float3(alpha * wo_local.x, alpha * wo_local.y, wo_local.z));
    float len_sq = vh.x * vh.x + vh.y * vh.y;
    float3 t1 = len_sq > 1e-12 ? float3(-vh.y, vh.x, 0.0) / sqrt(len_sq) : float3(1.0, 0.0, 0.0);
    float3 t2 = cross(vh, t1);
    float r = sqrt(u.x);
    float phi = 2.0 * PI * u.y;
    float p1 = r * cos(phi);
    float p2 = r * sin(phi);
    float s = 0.5 * (1.0 + vh.z);
    p2 = (1.0 - s) * sqrt(max(0.0, 1.0 - p1 * p1)) + s * p2;
    float3 nh = p1 * t1 + p2 * t2 + sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * vh;
    return normalize(float3(alpha * nh.x, alpha * nh.y, max(nh.z, 1e-6)));
}

// Full BSDF × cosine for direction wi, per wavelength, plus the mixture pdf.
float4 bsdf_eval(Surface s, float4 rho, float3 n, float3 wo, float3 wi, out float pdf)
{
    float cos_o = dot(n, wo);
    float cos_i = dot(n, wi);
    if (cos_o <= 0.0 || cos_i <= 0.0) {
        pdf = 0.0;
        return float4(0.0);
    }
    float3 h = normalize(wo + wi);
    float cos_h = saturate(dot(n, h));
    // Saturated: an epsilon above 1 would feed pow() a negative base (NaN).
    float h_dot_o = saturate(dot(h, wo));

    float4 f0 = lerp(float4(s.f0_dielectric), rho, s.metallic);
    float4 fresnel = f0 + (1.0 - f0) * pow(1.0 - h_dot_o, 5.0);
    float4 specular = fresnel * (ggx_d(s.alpha2, cos_h) * ggx_visibility(s.alpha2, cos_o, cos_i));
    float4 diffuse = rho * ((1.0 - s.metallic) * INV_PI);

    pdf = s.spec_prob * ggx_sample_pdf(s.alpha2, cos_o, cos_h)
        + (1.0 - s.spec_prob) * cos_i * INV_PI;
    return (diffuse + specular) * cos_i;
}

// Sample wi from the lobe mixture; returns the throughput multiplier
// f·cos/pdf per wavelength. u.xy drives the chosen lobe, u.w picks it.
float4 bsdf_sample(Surface s, float4 rho, float3 n, float3 wo,
                   float4 u, out float3 wi, out float pdf)
{
    float3 t, b;
    tangent_basis(n, t, b);
    float3 wo_local = float3(dot(wo, t), dot(wo, b), dot(wo, n));

    float3 wi_local;
    if (u.w < s.spec_prob) {
        float3 h = sample_ggx_vndf(wo_local, s.alpha, u.xy);
        wi_local = reflect(-wo_local, h);
        if (wi_local.z <= 0.0) {
            pdf = 0.0;
            wi = n;
            return float4(0.0);
        }
    } else {
        wi_local = cosine_sample_hemisphere(u.xy);
    }
    wi = normalize(t * wi_local.x + b * wi_local.y + n * wi_local.z);

    float4 f_cos = bsdf_eval(s, rho, n, wo, wi, pdf);
    if (pdf <= 1e-9) {
        pdf = 0.0;
        return float4(0.0);
    }
    return f_cos / pdf;
}
"#;

const RADIOSITY: &str = /*slang*/
    r#"
// Next-event estimation: pick one emissive triangle uniformly, sample a point
// on it, and weight against the BSDF strategy with the power heuristic. The
// matching BSDF-hit term in the integrator carries the complementary weight,
// so light reaching the camera through either strategy is counted exactly once.
// Returns per-wavelength radiance (sensor weights applied at accumulation).
float4 sample_direct_light(TraceRoot* r,
                           RaytracingAccelerationStructure tlas,
                           float3 p,
                           float3 n,
                           float3 wo,
                           Surface surf,
                           float4 rho,
                           float4 u)
{
    uint lightCount = r.dims1.y;
    if (lightCount == 0u) {
        return float4(0.0);
    }

    uint pick = min((uint)(u.x * (float)lightCount), lightCount - 1u);
    uint lightPrim = r.light_triangles[pick];
    GpuMaterial lightMat = r.materials[r.triangle_materials[lightPrim]];
    float leScale = lightMat.lagrange_emission.w;

    float3 lp = sample_triangle(r.verts, lightPrim, u.yz);
    float3 ln = normal_for_primitive(r.verts, lightPrim);
    float3 toLight = lp - p;
    float dist2 = max(dot(toLight, toLight), 1e-6);
    float dist = sqrt(dist2);
    float3 wi = toLight / dist;
    float cosLight = abs(dot(ln, wi));
    if (dot(n, wi) <= 0.0 || cosLight <= 1e-6) {
        return float4(0.0);
    }

    float bsdfPdf;
    float4 fCos = bsdf_eval(surf, rho, n, wo, wi, bsdfPdf);
    if (bsdfPdf <= 0.0) {
        return float4(0.0);
    }

    RayDesc shadow;
    shadow.Origin = p + n * 0.002;
    shadow.Direction = wi;
    shadow.TMin = 0.001;
    shadow.TMax = max(dist - 0.004, 0.001);
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> sq;
    sq.TraceRayInline(tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, shadow);
    while (sq.Proceed()) {}
    if (sq.CommittedStatus() != COMMITTED_NOTHING) {
        return float4(0.0);
    }

    float area = triangle_area(r.verts, lightPrim);
    // Solid-angle pdf of this NEE sample, light selection included.
    float lightPdf = dist2 / max(cosLight * area * (float)lightCount, 1e-8);
    float misWeight = power_heuristic(lightPdf, bsdfPdf);
    return fCos * (leScale / lightPdf) * misWeight;
}
"#;

const PATH_INTEGRATOR: &str = /*slang*/
    r#"
[shader("compute")]
[numthreads(8, 8, 1)]
void traceMain(uint3 tid : SV_DispatchThreadID,
               uniform TraceRoot* r,
               uniform RaytracingAccelerationStructure tlas)
{
    uint width = r.dims0.x;
    uint height = r.dims0.y;
    uint sampleStart = r.dims0.z;
    uint maxSpp = r.dims0.w;
    uint sampleBatch = max(r.dims1.z, 1u);
    if (tid.x >= width || tid.y >= height || sampleStart >= maxSpp) {
        return;
    }

    uint pixel = tid.y * width + tid.x;
    float3 sampleRadiance = float3(0.0);
    uint samplesTaken = 0u;

    for (uint sampleOffset = 0u; sampleOffset < sampleBatch; sampleOffset++) {
        uint sampleIndex = sampleStart + sampleOffset;
        if (sampleIndex >= maxSpp) {
            break;
        }
        samplesTaken++;

        // Group 0 = camera jitter + wavelength; per bounce, one group for light
        // sampling and one for BSDF direction + Russian roulette.
        float4 camU = sample_4d(pixel, sampleIndex, 0u);
        float2 jitter = camU.xy;
        float2 uv = (float2((float)tid.x, (float)tid.y) + jitter) / float2((float)width, (float)height);
        float2 ndc = float2(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y);
        float3 rayDir = normalize(
            r.cam_forward.xyz +
            r.cam_right.xyz * (ndc.x * r.lens.y * r.lens.x) +
            r.cam_up.xyz * (ndc.y * r.lens.y));

        // Four stratified wavelengths per path, importance sampled from the
        // light's baked table.
        WavePacket wave = sample_wavelengths(r, camU.z);

        RayDesc ray;
        ray.Origin = r.cam_pos.xyz;
        ray.Direction = rayDir;
        ray.TMin = 0.001;
        ray.TMax = 1000.0;

        float4 radiance = float4(0.0);   // spectral radiance per wavelength
        float4 throughput = float4(1.0); // reflectance products along the path
        float prevBsdfPdf = 0.0;

        for (uint bounce = 0u; bounce < 4u; bounce++) {
            RayQuery<RAY_FLAG_NONE> q;
            q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);
            while (q.Proceed()) {}

            if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
                // Nothing outside the box; escaped rays carry no energy.
                break;
            }

            uint prim = q.CommittedPrimitiveIndex();
            uint matId = r.triangle_materials[prim];
            GpuMaterial mat = r.materials[matId];
            float t = q.CommittedRayT();
            float3 p = ray.Origin + ray.Direction * t;
            float3 n = normal_for_primitive(r.verts, prim);
            if (dot(n, -ray.Direction) < 0.0) {
                n = -n;
            }

            if (mat.lagrange_emission.w > 0.0) {
                // MIS against the NEE strategy that could have sampled this
                // point from the previous vertex; camera hits keep full weight.
                float misWeight = 1.0;
                if (bounce > 0u) {
                    float cosLight = abs(dot(n, ray.Direction));
                    float area = triangle_area(r.verts, prim);
                    float lightPdf =
                        (t * t) / max(cosLight * area * (float)r.dims1.y, 1e-8);
                    misWeight = power_heuristic(prevBsdfPdf, lightPdf);
                }
                radiance += throughput * (mat.lagrange_emission.w * misWeight);
                break;
            }

            float3 wo = -ray.Direction;
            Surface surf = make_surface(mat);
            float4 rho = eval_reflectance(wave.phases, surf.lagranges);

            float4 lightU = sample_4d(pixel, sampleIndex, 1u + bounce * 2u);
            radiance += throughput * sample_direct_light(r, tlas, p, n, wo, surf, rho, lightU);

            float4 bsdfU = sample_4d(pixel, sampleIndex, 2u + bounce * 2u);
            float3 wi;
            float4 sampleWeight = bsdf_sample(surf, rho, n, wo, bsdfU, wi, prevBsdfPdf);
            if (prevBsdfPdf <= 0.0) {
                break;
            }
            ray.Origin = p + n * 0.002;
            ray.Direction = wi;
            ray.TMin = 0.001;
            ray.TMax = 1000.0;
            throughput *= sampleWeight;

            if (bounce >= 2u) {
                // Russian roulette on the hero wavelength; all lanes share the
                // path, so dividing every lane by the same probability is
                // unbiased.
                float keep = clamp(throughput.x, 0.05, 0.95);
                if (bsdfU.z > keep) {
                    break;
                }
                throughput /= keep;
            }
        }

        sampleRadiance += resolve_to_srgb(wave, radiance);
    }

    if (samplesTaken > 0u) {
        r.accum[pixel] += float4(sampleRadiance, (float)samplesTaken);
    }
}
"#;

/// The transport kernel's Slang source. Expects the sampler source and the
/// `Vertex`/`GpuMaterial`/`TraceRoot` declarations to be prepended.
pub fn source() -> String {
    [GEOMETRY, SAMPLING, RADIOSITY, PATH_INTEGRATOR].concat()
}
