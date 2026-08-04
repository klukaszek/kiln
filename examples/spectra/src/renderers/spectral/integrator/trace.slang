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
    uint sampleBatch = r.pass_count;
    uint tileWidth = (width + PIXEL_STRIDE - 1u) / PIXEL_STRIDE;
    uint tileHeight = (height + PIXEL_STRIDE - 1u) / PIXEL_STRIDE;
    if (tid.x >= tileWidth || tid.y >= tileHeight) {
        return;
    }

    for (uint passOffset = 0u; passOffset < sampleBatch; passOffset++) {
        uint passIndex = passStart + passOffset;
        uint sampleIndex = passIndex / PHASE_COUNT;
        uint phase = passIndex - sampleIndex * PHASE_COUNT;
        uint2 phaseOffset = uint2(phase % PIXEL_STRIDE, phase / PIXEL_STRIDE);
        uint2 pixelCoord = tid.xy * PIXEL_STRIDE + phaseOffset;
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

        float3 rayOrigin = r.cam_pos.xyz;
        float3 rayDirection = rayDir;

        float4 radiance = float4(0.0);   // spectral radiance per wavelength
        float4 throughput = float4(1.0); // reflectance products along the path
        float prevBsdfPdf = 0.0;

        for (uint bounce = 0u; bounce < 4u; bounce++) {
            uint prim;
            float hitT;
            float2 barycentrics;
            {
                // Keep the traversal object and RayDesc out of the shading
                // portion of the bounce. This limits the state that must
                // survive the ray-query operation on architectures that spill
                // query state to memory.
                RayDesc queryRay;
                queryRay.Origin = rayOrigin;
                queryRay.Direction = rayDirection;
                queryRay.TMin = 0.001;
                queryRay.TMax = 1000.0;

                // Geometry is all OPAQUE in the BLAS, so force-opaque lets the
                // hardware commit hits during traversal instead of returning
                // each candidate to this Proceed() loop.
                RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
                q.TraceRayInline(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, queryRay);
                while (q.Proceed()) {}

                if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
                    // Nothing outside the box; escaped rays carry no energy.
                    break;
                }
                prim = q.CommittedPrimitiveIndex();
                hitT = q.CommittedRayT();
                barycentrics = q.CommittedTriangleBarycentrics();
            }

            // Read only the primitive fields needed before branching. The
            // material ID is deferred until after the emissive-hit case, so
            // emissive triangles do not carry or fetch material state.
            float4 normalArea = r.triangles[prim].normal_area;
            float3 n = normalArea.xyz;
            float emission = r.triangles[prim].emission;
            float3 p = rayOrigin + rayDirection * hitT;
            if (dot(n, -rayDirection) < 0.0) {
                n = -n;
            }

            if (emission > 0.0) {
                // MIS against the NEE strategy that could have sampled this
                // point from the previous vertex; camera hits keep full weight.
                float misWeight = 1.0;
                if (bounce > 0u) {
                    float cosLight = abs(dot(n, rayDirection));
                    float area = normalArea.w;
                    float lightPdf = (hitT * hitT) / max(
                            cosLight * area * (float)LIGHT_COUNT, 1e-8);
                    misWeight = power_heuristic(prevBsdfPdf, lightPdf);
                }
                radiance += throughput * (emission * misWeight);
                break;
            }

            uint matId = r.triangles[prim].material_id;
            GpuBsdf surf = r.bsdfs[matId];
            float3 wo = -rayDirection;
            float4 rho = lookup_reflectance(r, matId, spectralIndices);
            if (surf.texture_id != 0xffffffffu) {
                GpuTriangle triangle = r.triangles[prim];
                float bary0 = 1.0 - barycentrics.x - barycentrics.y;
                float2 texcoord = bary0 * triangle.uv01.xy
                    + barycentrics.x * triangle.uv01.zw
                    + barycentrics.y * triangle.uv2.xy;
                GpuMaterialTexture materialTexture = r.material_textures[surf.texture_id];
                GpuTextureBinding binding = r.texture_bindings[materialTexture.binding_id];
                Texture2D texture = binding.image;
                SamplerState sampler = binding.sampler;
                float3 texel = texture.SampleLevel(
                    sampler, float2(texcoord.x, 1.0 - texcoord.y), 0.0).rgb;
                rho = texture_reflectance(
                    r, texel * materialTexture.factor.rgb, spectralIndices);
            }

            // A very sharp, highly specular lobe is effectively delta-like:
            // random-point NEE almost never lands inside it, but still pays for
            // a full shadow traversal. Keep NEE for every diffuse/glossy case.
            bool near_delta = surf.alpha2 <= 1e-5 && surf.spec_prob >= 0.9;
            if (!near_delta) {
                float3 lightU = sample_3d(pixelSeed, sampleIndex, 1u + bounce * 2u);
                radiance += throughput * sample_direct_light(r, tlas, p, n, wo, surf, rho, lightU);
            }

            float4 bsdfU = sample_4d(pixelSeed, sampleIndex, 2u + bounce * 2u);
            float3 wi;
            float4 sampleWeight = bsdf_sample(
                surf, rho, n, wo, bsdfU, wi, prevBsdfPdf);
            if (prevBsdfPdf <= 0.0) {
                break;
            }
            rayOrigin = p + n * 0.002;
            rayDirection = wi;
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
