// ---------------------------------------------------------------------------
// BSDF: Lambert diffuse + single-scatter GGX specular (height-correlated
// Smith, VNDF sampling). Per wavelength, Fresnel F0 is the moment-based
// reflectance spectrum for metals and the IOR-derived constant for
// dielectrics, so a copper F0 disperses exactly like a copper albedo would.
// Directions are sampled from a lobe mixture and every evaluation returns the
// full mixture pdf, so NEE/BSDF MIS stays exact for both lobes.
// ---------------------------------------------------------------------------

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
float4 bsdf_eval(GpuBsdf s, float4 rho, float3 n, float3 wo, float3 wi, out float pdf)
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
float4 bsdf_sample(GpuBsdf s, float4 rho, float3 n, float3 wo,
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
