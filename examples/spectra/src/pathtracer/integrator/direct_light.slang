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
                           float3 u)
{
    uint lightCount = LIGHT_COUNT;

    uint pick = min((uint)(u.x * (float)lightCount), lightCount - 1u);
    float su0 = sqrt(u.y);
    float b1 = u.z * su0;
    float b2 = su0 - b1;
    float3 lp;
    float3 ln;
    float lightArea;
    float lightEmission;
    {
        GpuLight light = r.lights[pick];
        lp = light.p0_emission.xyz
            + light.edge1_area.xyz * b1
            + light.edge2.xyz * b2;
        ln = light.normal.xyz;
        lightArea = light.edge1_area.w;
        lightEmission = light.p0_emission.w;
    }
    float3 toLight = lp - p;
    float dist2 = max(dot(toLight, toLight), 1e-6);
    float dist = sqrt(dist2);
    float3 wi = toLight / dist;
    float cosLight = abs(dot(ln, wi));
    if (dot(n, wi) <= 0.0 || cosLight <= 1e-6) {
        return float4(0.0);
    }

    bool occluded;
    {
        RayDesc shadow;
        shadow.Origin = p + n * 0.002;
        shadow.Direction = wi;
        shadow.TMin = 0.001;
        shadow.TMax = max(dist - 0.004, 0.001);
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE> sq;
        sq.TraceRayInline(
            tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE, 0xFF, shadow);
        while (sq.Proceed()) {}
        occluded = sq.CommittedStatus() != COMMITTED_NOTHING;
    }
    if (occluded) {
        return float4(0.0);
    }

    // Keep BSDF state out of the shadow traversal and avoid evaluating it for blocked samples.
    float bsdfPdf;
    float4 fCos = bsdf_eval(surf, rho, n, wo, wi, bsdfPdf);
    if (bsdfPdf <= 0.0) {
        return float4(0.0);
    }

    // Solid-angle pdf of this NEE sample, light selection included.
    float lightPdf =
        dist2 / max(cosLight * lightArea * (float)lightCount, 1e-8);
    float misWeight = power_heuristic(lightPdf, bsdfPdf);
    return fCos * (lightEmission / lightPdf) * misWeight;
}
