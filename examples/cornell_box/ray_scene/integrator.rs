pub const DEFAULT_TARGET_SPP: u32 = 1024;
pub const DEFAULT_SAMPLES_PER_FRAME: u32 = 16;
pub const THREADS_X: u32 = 8;
pub const THREADS_Y: u32 = 8;

const RNG: &str = /*slang*/
    r#"
uint hash_u32(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float hash_to_unit(uint x)
{
    return ((float)(hash_u32(x) & 0x00ffffffu) + 0.5) / 16777216.0;
}

float sample_dim(uint pixel, uint sampleIndex, uint dim)
{
    uint stream = hash_u32(pixel ^ hash_u32(dim * 0x9e3779b9u));
    float offset = hash_to_unit(stream);
    float alpha = 0.1 + 0.8 * hash_to_unit(dim ^ 0x68bc21ebu);
    return frac(offset + alpha * (float)(sampleIndex + 1u));
}
"#;

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
float3 cosine_sample_hemisphere(float2 u)
{
    float r = sqrt(u.x);
    float phi = 6.28318530718 * u.y;
    return float3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u.x)));
}

float3 tangent_to_world(float3 localDir, float3 n)
{
    float3 helper = abs(n.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(0.0, 1.0, 0.0);
    float3 tangent = normalize(cross(helper, n));
    float3 bitangent = cross(n, tangent);
    return normalize(tangent * localDir.x + bitangent * localDir.y + n * localDir.z);
}
"#;

const RADIOSITY: &str = /*slang*/
    r#"
float3 direct_lighting(TraceRoot* r,
                       RaytracingAccelerationStructure tlas,
                       float3 p,
                       float3 n,
                       float3 albedo,
                       uint pixel,
                       uint sampleIndex,
                       uint dimBase)
{
    float3 radiance = float3(0.0);
    uint lightCount = r.dims1.y;
    for (uint i = 0u; i < lightCount; i++) {
        uint lightPrim = r.light_triangles[i];
        uint lightMatId = r.triangle_materials[lightPrim];
        GpuMaterial lightMat = r.materials[lightMatId];
        float3 Le = lightMat.emission_metallic.xyz;
        uint lightDim = dimBase + i * 2u;
        float3 lp = sample_triangle(
            r.verts,
            lightPrim,
            float2(
                sample_dim(pixel, sampleIndex, lightDim),
                sample_dim(pixel, sampleIndex, lightDim + 1u)));
        float3 ln = normal_for_primitive(r.verts, lightPrim);
        float3 toLight = lp - p;
        float dist2 = max(dot(toLight, toLight), 1e-6);
        float dist = sqrt(dist2);
        float3 wi = toLight / dist;
        float cosSurface = max(dot(n, wi), 0.0);
        float cosLight = abs(dot(ln, -wi));
        if (cosSurface <= 0.0 || cosLight <= 0.0) {
            continue;
        }

        RayDesc shadow;
        shadow.Origin = p + n * 0.002;
        shadow.Direction = wi;
        shadow.TMin = 0.001;
        shadow.TMax = max(dist - 0.004, 0.001);
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> sq;
        sq.TraceRayInline(tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, shadow);
        sq.Proceed();
        if (sq.CommittedStatus() != COMMITTED_NOTHING) {
            continue;
        }

        float area = triangle_area(r.verts, lightPrim);
        radiance += (albedo * (1.0 / 3.14159265)) * Le * (cosSurface * cosLight * area / dist2);
    }
    return radiance;
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

        float2 jitter = float2(sample_dim(pixel, sampleIndex, 0u), sample_dim(pixel, sampleIndex, 1u));
        float2 uv = (float2((float)tid.x, (float)tid.y) + jitter) / float2((float)width, (float)height);
        float2 ndc = float2(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y);
        float3 rayDir = normalize(
            r.cam_forward.xyz +
            r.cam_right.xyz * (ndc.x * r.lens.y * r.lens.x) +
            r.cam_up.xyz * (ndc.y * r.lens.y));

        RayDesc ray;
        ray.Origin = r.cam_pos.xyz;
        ray.Direction = rayDir;
        ray.TMin = 0.001;
        ray.TMax = 1000.0;

        float3 throughput = float3(1.0);

        [unroll]
        for (uint bounce = 0u; bounce < 4u; bounce++) {
            RayQuery<RAY_FLAG_NONE> q;
            q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);
            q.Proceed();

            if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
                if (bounce == 0u) {
                    sampleRadiance += throughput * float3(0.015, 0.018, 0.025);
                }
                break;
            }

            uint prim = q.CommittedPrimitiveIndex();
            uint matId = r.triangle_materials[prim];
            GpuMaterial mat = r.materials[matId];
            float3 albedo = mat.base_roughness.xyz;
            float3 emission = mat.emission_metallic.xyz;
            float t = q.CommittedRayT();
            float3 p = ray.Origin + ray.Direction * t;
            float3 n = normal_for_primitive(r.verts, prim);
            if (dot(n, -ray.Direction) < 0.0) {
                n = -n;
            }

            if (any_nonzero(emission)) {
                sampleRadiance += throughput * emission;
                break;
            }

            uint bounceDim = 2u + bounce * 64u;
            sampleRadiance += throughput * direct_lighting(
                r,
                tlas,
                p,
                n,
                albedo,
                pixel,
                sampleIndex,
                bounceDim);

            float3 localBounce = cosine_sample_hemisphere(float2(
                sample_dim(pixel, sampleIndex, bounceDim + 48u),
                sample_dim(pixel, sampleIndex, bounceDim + 49u)));
            ray.Origin = p + n * 0.002;
            ray.Direction = tangent_to_world(localBounce, n);
            ray.TMin = 0.001;
            ray.TMax = 1000.0;
            throughput *= albedo;

            if (bounce >= 2u) {
                float keep = clamp(max(max(throughput.x, throughput.y), throughput.z), 0.05, 0.95);
                if (sample_dim(pixel, sampleIndex, bounceDim + 50u) > keep) {
                    break;
                }
                throughput /= keep;
            }
        }
    }

    if (samplesTaken > 0u) {
        r.accum[pixel] += float4(sampleRadiance, (float)samplesTaken);
    }
}
"#;

pub fn source() -> String {
    [RNG, GEOMETRY, SAMPLING, RADIOSITY, PATH_INTEGRATOR].concat()
}
