[shader("compute")]
[numthreads(8, 8, 1)]
void traceMain(uint3 tid : SV_DispatchThreadID, uniform TraceRoot* r)
{
    RaytracingAccelerationStructure tlas = r.tlas;
    uint width = r.film_width;
    uint height = r.film_height;
    uint passStart = r.pass_start;
    uint sampleBatch = r.pass_count;
    uint pixelStride = r.settings.y;
    uint phaseCount = r.settings.z;
    uint tileWidth = (width + pixelStride - 1u) / pixelStride;
    uint tileHeight = (height + pixelStride - 1u) / pixelStride;
    if (tid.x >= tileWidth || tid.y >= tileHeight) {
        return;
    }

    for (uint passOffset = 0u; passOffset < sampleBatch; passOffset++) {
        uint passIndex = passStart + passOffset;
        uint sampleIndex = passIndex / phaseCount;
        uint phase = passIndex - sampleIndex * phaseCount;
        uint2 phaseOffset = uint2(phase % pixelStride, phase / pixelStride);
        uint2 pixelCoord = tid.xy * pixelStride + phaseOffset;
        if (pixelCoord.x >= width || pixelCoord.y >= height) {
            continue;
        }
        uint pixel = pixelCoord.y * width + pixelCoord.x;
        uint pixelSeed = hash_u32(pixel);

        float3 camU = sample_3d(pixelSeed, sampleIndex, 0u);
        float2 jitter = camU.xy;
        float2 uv = (float2(pixelCoord) + jitter) / float2((float)width, (float)height);
        float2 ndc = float2(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y);
        float3 rayDir = normalize(
            r.cam_forward.xyz +
            r.cam_right.xyz * (ndc.x * r.lens.y * r.lens.x) +
            r.cam_up.xyz * (ndc.y * r.lens.y));

        float xi = camU.z;
        uint4 spectralIndices = sample_spectral_indices(xi);

        float3 rayOrigin = r.cam_pos.xyz;
        float3 rayDirection = rayDir;

        float4 radiance = float4(0.0);   // spectral radiance per wavelength
        float4 throughput = float4(1.0); // reflectance products along the path
        float prevBsdfPdf = 0.0;
        bool prevNeeEnabled = false;

        for (uint bounce = 0u; bounce < 4u; bounce++) {
            uint prim;
            uint instanceId;
            float hitT;
            float2 barycentrics;
            {
                RayDesc queryRay;
                queryRay.Origin = rayOrigin;
                queryRay.Direction = rayDirection;
                queryRay.TMin = 0.001;
                queryRay.TMax = 1000.0;

                RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
                q.TraceRayInline(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, queryRay);
                while (q.Proceed()) {}

                if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
                    break;
                }
                instanceId = q.CommittedInstanceID();
                prim = r.instances[instanceId].triangle_base + q.CommittedPrimitiveIndex();
                hitT = q.CommittedRayT();
                barycentrics = q.CommittedTriangleBarycentrics();
            }

            GpuTriangle triangle = r.triangles[prim];
            GpuInstance instance = r.instances[instanceId];
            float3 localNormal = triangle.normal_area.xyz;
            float3 transformedNormal =
                instance.normal_x.xyz * localNormal.x +
                instance.normal_y.xyz * localNormal.y +
                instance.normal_z_determinant.xyz * localNormal.z;
            float3 n = normalize(transformedNormal);
            float emission = triangle.emission;
            float3 p = rayOrigin + rayDirection * hitT;
            if (dot(n, -rayDirection) < 0.0) {
                n = -n;
            }

            if (emission > 0.0) {
                float4 spectrumScale;
                uint spectrumBase = triangle.material_id * SPECTRUM_LEN;
                [ForceUnroll]
                for (uint k = 0u; k < 4u; k++) {
                    spectrumScale[k] = r.material_emission_spectrum[
                        spectrumBase + spectralIndices[k]];
                }
                float misWeight = 1.0;
                if (bounce > 0u && prevNeeEnabled) {
                    GpuLight hitLight = r.lights[r.emissive_hits[prim].light_index];
                    float cosLight = abs(dot(n, rayDirection));
                    float lightPdf = (hitT * hitT) / max(
                        cosLight * hitLight.edge1_area.w * (float)r.settings.x, 1e-8);
                    misWeight = power_heuristic(prevBsdfPdf, lightPdf);
                }
                radiance += throughput * (emission * spectrumScale * misWeight);
                break;
            }

            uint matId = triangle.material_id;
            GpuBsdf surf = r.bsdfs[matId];
            float3 wo = -rayDirection;
            float4 rho = lookup_reflectance(r, matId, spectralIndices);
            if (surf.texture_id != 0xffffffffu) {
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

            bool near_delta = surf.alpha2 <= 1e-5 && surf.spec_prob >= 0.9;
            bool neeEnabled = !near_delta && r.settings.x > 0u;
            if (neeEnabled) {
                float3 lightU = sample_3d(pixelSeed, sampleIndex, 1u + bounce * 2u);
                radiance += throughput * sample_direct_light(
                    r, tlas, p, n, wo, surf, rho, lightU, spectralIndices);
            }

            float4 bsdfU = sample_4d(pixelSeed, sampleIndex, 2u + bounce * 2u);
            prevNeeEnabled = neeEnabled;
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
                float keep = clamp(
                    (throughput.x + throughput.y + throughput.z + throughput.w) * 0.25,
                    0.05,
                    0.95);
                if (bsdfU.z > keep) {
                    break;
                }
                throughput /= keep;
            }
        }

        splat_spectral(r, pixel, spectralIndices, radiance);
    }
}
