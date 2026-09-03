float4 sample_direct_light(TraceRoot* r,
                           RaytracingAccelerationStructure tlas,
                           float3 p,
                           float3 n,
                           float3 wo,
                           Surface surf,
                           float3 u,
                           uint4 spectralIndices)
{
    uint lightCount = r.settings.x;
    uint pick = (uint)(u.x * (float)lightCount);
    GpuLight light = r.lights[pick];
    uint kind = (uint)light.normal.w;
    float lightEmission = light.p0_emission.w;
    if (lightEmission <= 0.0) {
        return float4(0.0);
    }
    float3 lp;
    float3 wi;
    float3 ln;
    float dist;
    float dist2;
    bool delta = false;
    bool infinite = false;
    float lightArea = 0.0;
    float lightPdf = 0.0;

    if (kind == LIGHT_MESH) {
        uint cdfStart = (uint)light.edge2.x;
        uint triangleCount = (uint)light.edge2.y;
        uint lo = 0u;
        uint hi = triangleCount;
        while (lo < hi) {
            uint mid = (lo + hi) / 2u;
            if (r.mesh_light_cdf[cdfStart + mid] < u.y) {
                lo = mid + 1u;
            } else {
                hi = mid;
            }
        }
        uint triangleIndex = lo;
        float previousCdf = triangleIndex == 0u
            ? 0.0
            : r.mesh_light_cdf[cdfStart + triangleIndex - 1u];
        float currentCdf = r.mesh_light_cdf[cdfStart + triangleIndex];
        float triangleU = saturate(
            (u.y - previousCdf) / max(currentCdf - previousCdf, 1e-8));
        GpuMeshLightTriangle triangle = r.mesh_light_triangles[cdfStart + triangleIndex];
        float su0 = sqrt(triangleU);
        float b1 = u.z * su0;
        float b2 = su0 - b1;
        lp = triangle.p0.xyz + triangle.edge1.xyz * b1 + triangle.edge2.xyz * b2;
        ln = triangle.normal_area.xyz;
        lightArea = light.edge1_area.w;
        float3 toLight = lp - p;
        dist2 = max(dot(toLight, toLight), 1e-6);
        dist = sqrt(dist2);
        wi = toLight / dist;
        float cosLight = abs(dot(ln, wi));
        if (cosLight <= 1e-6) {
            return float4(0.0);
        }
    } else if (kind == LIGHT_POINT) {
        lp = light.p0_emission.xyz;
        float3 toLight = lp - p;
        dist2 = max(dot(toLight, toLight), 1e-6);
        dist = sqrt(dist2);
        wi = toLight / dist;
        delta = true;
    } else if (kind == LIGHT_SPHERE) {
        float z = 1.0 - 2.0 * u.y;
        float radial = sqrt(max(0.0, 1.0 - z * z));
        float phi = 6.28318530718 * u.z;
        ln = float3(radial * cos(phi), radial * sin(phi), z);
        lp = light.p0_emission.xyz + ln * light.normal.x;
        lightArea = light.edge1_area.w;
        float3 toLight = lp - p;
        dist2 = max(dot(toLight, toLight), 1e-6);
        dist = sqrt(dist2);
        wi = toLight / dist;
        float cosLight = max(dot(ln, -wi), 0.0);
        if (cosLight <= 1e-6) {
            return float4(0.0);
        }
    } else if (kind == LIGHT_DIRECTIONAL || kind == LIGHT_DOME) {
        wi = normalize(-light.edge1_area.xyz);
        dist = 1e6;
        dist2 = 1.0;
        delta = true;
        infinite = true;
    } else {
        if (kind == LIGHT_TRIANGLE) {
            float su0 = sqrt(u.y);
            float b1 = u.z * su0;
            float b2 = su0 - b1;
            lp = light.p0_emission.xyz
                + light.edge1_area.xyz * b1
                + light.edge2.xyz * b2;
        } else if (kind == LIGHT_RECT) {
            lp = light.p0_emission.xyz
                + light.edge1_area.xyz * u.y
                + light.edge2.xyz * u.z;
        } else {
            float radius = sqrt(u.y);
            float angle = 6.28318530718 * u.z;
            lp = light.p0_emission.xyz
                + light.edge1_area.xyz * (radius * cos(angle))
                + light.edge2.xyz * (radius * sin(angle));
        }
        ln = light.normal.xyz;
        lightArea = light.edge1_area.w;
        float3 toLight = lp - p;
        dist2 = max(dot(toLight, toLight), 1e-6);
        dist = sqrt(dist2);
        wi = toLight / dist;
        float cosLight = abs(dot(ln, wi));
        if (cosLight <= 1e-6) {
            return float4(0.0);
        }
    }

    if (dot(n, wi) <= 0.0) {
        return float4(0.0);
    }

    RayDesc shadow;
    shadow.Origin = p + n * 0.002;
    shadow.Direction = wi;
    shadow.TMin = 0.001;
    shadow.TMax = infinite ? 1e6 : max(dist - 0.004, 0.001);
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE> sq;
    sq.TraceRayInline(
        tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE, 0xFF, shadow);
    while (sq.Proceed()) {}
    if (sq.CommittedStatus() != COMMITTED_NOTHING) {
        return float4(0.0);
    }

    float bsdfPdf;
    float4 fCos = bsdf_eval(surf, n, wo, wi, bsdfPdf);
    float4 spectrumScale;
    uint spectrumBase = (uint)light.edge2.w * SPECTRUM_LEN;
    [ForceUnroll]
    for (uint k = 0u; k < 4u; k++) {
        spectrumScale[k] = r.light_spectrum[spectrumBase + spectralIndices[k]];
    }
    if (delta) {
        return fCos * (lightEmission * spectrumScale / dist2);
    }

    if (lightPdf == 0.0) {
        float cosLight = abs(dot(ln, wi));
        lightPdf = dist2 / max(cosLight * lightArea * (float)lightCount, 1e-8);
    }
    float misWeight = kind == LIGHT_MESH ? power_heuristic(lightPdf, bsdfPdf) : 1.0;
    return fCos * (lightEmission * spectrumScale / lightPdf) * misWeight;
}
