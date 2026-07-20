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
                           GpuBsdf surf,
                           float4 rho,
                           float4 u)
{
    uint lightCount = r.light_count;
    if (lightCount == 0u) {
        return float4(0.0);
    }

    uint pick = min((uint)(u.x * (float)lightCount), lightCount - 1u);
    GpuLight light = r.lights[pick];
    float su0 = sqrt(u.y);
    float b1 = u.z * su0;
    float b2 = su0 - b1;
    float3 lp = light.p0_emission.xyz
        + light.edge1_area.xyz * b1
        + light.edge2.xyz * b2;
    float3 ln = light.normal.xyz;
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
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE> sq;
    sq.TraceRayInline(
        tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE, 0xFF, shadow);
    while (sq.Proceed()) {}
    if (sq.CommittedStatus() != COMMITTED_NOTHING) {
        return float4(0.0);
    }

    // Solid-angle pdf of this NEE sample, light selection included.
    float lightPdf =
        dist2 / max(cosLight * light.edge1_area.w * (float)lightCount, 1e-8);
    float misWeight = power_heuristic(lightPdf, bsdfPdf);
    return fCos * (light.p0_emission.w / lightPdf) * misWeight;
}
