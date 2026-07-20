[shader("compute")]
[numthreads(8, 8, 1)]
void traceMain(uint3 tid : SV_DispatchThreadID, uniform TraceRoot* r)
{
    // Bindless TLAS: the handle in the root resolves to the acceleration structure (no descriptor
    // binding / argument-table slot). See docs/design/vulkan-binding-convention.md.
    RaytracingAccelerationStructure tlas = r.tlas;
    uint width = r.film_width;
    uint height = r.film_height;
    uint passStart = r.pass_start;
    uint maxPasses = r.target_passes;
    uint sampleBatch = max(r.pass_count, 1u);
    uint pixelStride = max(r.pixel_stride, 1u);
    uint tileWidth = (width + pixelStride - 1u) / pixelStride;
    uint tileHeight = (height + pixelStride - 1u) / pixelStride;
    uint phaseCount = r.phase_count;
    if (tid.x >= tileWidth || tid.y >= tileHeight || passStart >= maxPasses) {
        return;
    }

    for (uint passOffset = 0u; passOffset < sampleBatch; passOffset++) {
        uint passIndex = passStart + passOffset;
        if (passIndex >= maxPasses) {
            break;
        }
        uint sampleIndex = passIndex / phaseCount;
        uint phase = passIndex - sampleIndex * phaseCount;
        uint2 phaseOffset = uint2(phase % pixelStride, phase / pixelStride);
        uint2 pixelCoord = tid.xy * pixelStride + phaseOffset;
        if (pixelCoord.x >= width || pixelCoord.y >= height) {
            continue;
        }
        uint pixel = pixelCoord.y * width + pixelCoord.x;
        uint pixelSeed = hash_u32(pixel);

        // Group 0 = camera jitter + wavelength; per bounce, one group for light
        // sampling and one for BSDF direction + Russian roulette.
        float4 camU = sample_4d(pixelSeed, sampleIndex, 0u);
        float2 jitter = camU.xy;
        float2 uv = (float2(pixelCoord) + jitter) / float2((float)width, (float)height);
        float2 ndc = float2(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y);
        float3 rayDir = normalize(
            r.cam_forward.xyz +
            r.cam_right.xyz * (ndc.x * r.lens.y * r.lens.x) +
            r.cam_up.xyz * (ndc.y * r.lens.y));

        // Light-CDF and uniform-wavelength strata.
        float xi = camU.z;
        float xiU = camU.w;
        uint4 spectralIndices = sample_spectral_indices(r, xi, xiU);

        RayDesc ray;
        ray.Origin = r.cam_pos.xyz;
        ray.Direction = rayDir;
        ray.TMin = 0.001;
        ray.TMax = 1000.0;

        float4 radiance = float4(0.0);   // spectral radiance per wavelength
        float4 throughput = float4(1.0); // reflectance products along the path
        float prevBsdfPdf = 0.0;

        for (uint bounce = 0u; bounce < 4u; bounce++) {
            // Geometry is all OPAQUE in the BLAS, so force-opaque lets the
            // hardware commit hits during traversal instead of returning each
            // candidate to this Proceed() loop.
            RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
            q.TraceRayInline(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, ray);
            while (q.Proceed()) {}

            if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
                // Nothing outside the box; escaped rays carry no energy.
                break;
            }

            uint prim = q.CommittedPrimitiveIndex();
            GpuTriangle triangle = r.triangles[prim];
            uint matId = triangle.material_id;
            float t = q.CommittedRayT();
            float3 p = ray.Origin + ray.Direction * t;
            float3 n = triangle.normal_area.xyz;
            if (dot(n, -ray.Direction) < 0.0) {
                n = -n;
            }

            if (triangle.emission > 0.0) {
                // MIS against the NEE strategy that could have sampled this
                // point from the previous vertex; camera hits keep full weight.
                float misWeight = 1.0;
                if (bounce > 0u) {
                    float cosLight = abs(dot(n, ray.Direction));
                    float lightPdf = (t * t) / max(
                            cosLight * triangle.normal_area.w * (float)r.light_count, 1e-8);
                    misWeight = power_heuristic(prevBsdfPdf, lightPdf);
                }
                radiance += throughput * (triangle.emission * misWeight);
                break;
            }

            GpuBsdf surf = r.bsdfs[matId];
            float3 wo = -ray.Direction;
            float4 rho = lookup_reflectance(r, matId, spectralIndices);

            float4 lightU = sample_4d(pixelSeed, sampleIndex, 1u + bounce * 2u);
            radiance += throughput * sample_direct_light(r, tlas, p, n, wo, surf, rho, lightU);

            float4 bsdfU = sample_4d(pixelSeed, sampleIndex, 2u + bounce * 2u);
            float3 wi;
            float4 sampleWeight = bsdf_sample(
                surf, rho, n, wo, bsdfU, wi, prevBsdfPdf);
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

        splat_spectral(r, pixel, spectralIndices, radiance);
    }
}
