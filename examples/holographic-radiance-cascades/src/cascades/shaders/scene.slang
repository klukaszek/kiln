// Analytic 2D primitives: distance, normals, outline tessellation, and the two ways the solver
// asks what a ray hits — sphere marching against the SDF, and the hardware BVH.
//
// Ported from the SlangPy original. The one structural change is how scene data is reached: there
// are no tensors, so every buffer arrives as a device pointer on the root struct and the indexing
// that slangpy generated is written out. A packed primitive is four `float4` rows, so row `j` of
// primitive `i` is `scene[i * 4 + j]`.

float2 unit(float a) { return float2(cos(a), sin(a)); }

struct Prim
{
    int kind;
    float2 a;       // circle/box centre, or segment start
    float2 b;       // box half-extent, or segment end
    float r;        // circle radius, box corner radius, segment thickness
    float2 rot;     // box rotation as (cos, sin)
    float3 emission;
    float3 albedo;
};

// Distance only needs the geometry rows. Pulling emission and albedo too doubles the loads in the
// inner loop of every scene query.
Prim loadPrimGeometry(Ptr<float4, Access.Read> scene, int i)
{
    float4 v0 = scene[i * 4 + 0];
    float4 v1 = scene[i * 4 + 1];

    Prim p;
    p.kind = int(v0.x);
    p.a = v0.yz;
    p.r = v0.w;
    p.b = v1.xy;
    p.rot = v1.zw;
    p.emission = float3(0.0);
    p.albedo = float3(0.0);
    return p;
}

Prim loadPrim(Ptr<float4, Access.Read> scene, int i)
{
    Prim p = loadPrimGeometry(scene, i);
    p.emission = scene[i * 4 + 2].rgb;
    p.albedo = scene[i * 4 + 3].rgb;
    return p;
}

float sdCircle(float2 p, float2 c, float r)
{
    return length(p - c) - r;
}

// rot is (cos, sin) of the box angle, precomputed on the host. Recomputing the pair here costs two
// transcendentals per primitive per pixel, and resolve runs over every output pixel.
float2 toBoxLocal(float2 p, float2 c, float2 rot)
{
    float2 q = p - c;
    return float2(rot.x * q.x + rot.y * q.y, -rot.y * q.x + rot.x * q.y);
}

float sdBox(float2 p, float2 c, float2 h, float2 rot, float round)
{
    float2 q = toBoxLocal(p, c, rot);
    float2 d = abs(q) - max(h - round, 0.0);
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0) - round;
}

float sdSegment(float2 p, float2 a, float2 b, float r)
{
    float2 pa = p - a;
    float2 ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

float primDist(Prim p, float2 pos)
{
    if (p.kind == KIND_CIRCLE)
        return sdCircle(pos, p.a, p.r);
    if (p.kind == KIND_BOX)
        return sdBox(pos, p.a, p.b, p.rot, p.r);
    return sdSegment(pos, p.a, p.b, p.r);
}

struct SceneHit
{
    float d;
    int id;
};

// Vertex e of a primitive's outline polygon. Indexed rather than walked by arc length, so corners
// are always sample points -- sampling a box's perimeter uniformly chamfers it, which shows up
// directly as a transmittance error.
float2 outlineVertex(Prim p, int e)
{
    if (p.kind == KIND_CIRCLE)
        return p.a + p.r * unit(TWO_PI * float(e) / float(EDGES_PER_PRIM));

    if (p.kind == KIND_BOX)
    {
        // Each quarter spends its whole budget on one corner arc; the straight edges are the links
        // between quarters. At zero corner radius the arc collapses onto the corner and the polygon
        // is exactly the rectangle.
        const int per_corner = EDGES_PER_PRIM / 4;
        int q = e / per_corner;
        int j = e - q * per_corner;

        float2 inner = max(p.b - p.r, 0.0);
        float2 signs = float2(q == 0 || q == 1 ? 1.0 : -1.0, q == 1 || q == 2 ? 1.0 : -1.0);
        float a = (-0.5 + float(q)) * PI + float(j) / float(per_corner - 1) * 0.5 * PI;
        float2 local = inner * signs + p.r * unit(a);

        return p.a + float2(p.rot.x * local.x - p.rot.y * local.y, p.rot.y * local.x + p.rot.x * local.y);
    }

    const int per_cap = EDGES_PER_PRIM / 2;
    int cap = e / per_cap;
    int j = e - cap * per_cap;

    float2 axis = p.b - p.a;
    float len = length(axis);
    float2 dir = len > 1e-6 ? axis / len : float2(1.0, 0.0);
    float base = atan2(dir.y, dir.x);
    float a = base - 0.5 * PI + float(cap) * PI + float(j) / float(per_cap - 1) * PI;
    return (cap == 0 ? p.b : p.a) + p.r * unit(a);
}

// Outward normal of one primitive, analytically. Finite differences would need four more sweeps
// over every primitive, and are noisy on a box's medial axis.
float2 primNormal(Prim p, float2 pos)
{
    if (p.kind == KIND_CIRCLE)
        return normalize(pos - p.a);

    if (p.kind == KIND_BOX)
    {
        float2 q = toBoxLocal(pos, p.a, p.rot);
        float2 d = abs(q) - max(p.b - p.r, 0.0);
        float2 g = d.x > 0.0 || d.y > 0.0
            ? normalize(max(d, 0.0)) * sign(q)
            : (d.x > d.y ? float2(sign(q.x), 0.0) : float2(0.0, sign(q.y)));
        return float2(p.rot.x * g.x - p.rot.y * g.y, p.rot.y * g.x + p.rot.x * g.y);
    }

    float2 ba = p.b - p.a;
    float h = clamp(dot(pos - p.a, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return normalize(pos - (p.a + ba * h));
}

// --- scene queries ----------------------------------------------------------

// Broadphase lookup. One uniform grid cell lists the primitives whose surface can reach it, so a
// distance query costs the cell's occupancy rather than the whole scene.
SceneHit sceneEvalGrid(Root* r, float2 pos)
{
    SceneHit h;
    h.d = 1e20;
    h.id = -1;

    int2 cell = clamp(int2(float2(r.cells) * pos / float2(r.res)), int2(0, 0), r.cells - 1);
    uint n = min(r.cell_count[cell.y * r.cells.x + cell.x], uint(GRID_CAPACITY));

    for (uint k = 0; k < n; ++k)
    {
        int i = int(r.cell_prims[(cell.y * r.cells.x + cell.x) * GRID_CAPACITY + int(k)]);
        float d = primDist(loadPrimGeometry(r.scene, i), pos);
        if (d < h.d)
        {
            h.d = d;
            h.id = i;
        }
    }
    return h;
}

// Whole-scene distance, with no broadphase. Only the brute-force reference uses this: it is an
// implementation independent of both the grid and the BVH, so agreement with it checks those too.
SceneHit sceneEval(Ptr<float4, Access.Read> scene, int count, float2 pos)
{
    SceneHit h;
    h.d = 1e20;
    h.id = -1;
    for (int i = 0; i < count; ++i)
    {
        float d = primDist(loadPrimGeometry(scene, i), pos);
        if (d < h.d)
        {
            h.d = d;
            h.id = i;
        }
    }
    return h;
}

// Sphere-march one radiance interval along [tMin, tMax).
float4 traceInterval(
    Ptr<float4, Access.Read> scene,
    int count,
    float2 origin,
    float2 dir,
    float tMin,
    float tMax,
    int maxSteps,
    float eps)
{
    float t = tMin;
    for (int i = 0; i < maxSteps; ++i)
    {
        SceneHit h = sceneEval(scene, count, origin + dir * t);
        if (h.d < eps)
            return float4(loadPrim(scene, h.id).emission, 0.0);
        t += max(h.d, eps);
        if (t >= tMax)
            break;
    }
    return float4(0.0, 0.0, 0.0, 1.0);
}

// --- hardware ray tracing ---------------------------------------------------

// Surfaces re-emit what reached them last frame. The paper models the world as participating media
// and gets diffuse bounce by feeding output fluence back through another iteration; this is that
// feedback, applied at the surface.
float3 reflected(Root* r, float2 p, float3 albedo)
{
    if (r.bounce <= 0.0)
        return float3(0.0);
    int2 q = clamp(int2(p), int2(0, 0), r.res - 1);
    return albedo * r.bounce * r.history[q.y * r.res.x + q.x].rgb / TWO_PI;
}

// Trace one radiance interval along [tMin, tMax). Returns (radiance, transmittance): a hit is
// opaque, so transmittance is 0 there and 1 for a ray that escapes.
float4 traceIntervalRT(Root* r, float2 origin, float2 dir, float tMin, float tMax)
{
    RayDesc ray;
    ray.Origin = float3(origin, 0.0);
    ray.Direction = float3(dir, 0.0);
    ray.TMin = max(tMin, 1e-3);
    ray.TMax = max(tMax, ray.TMin);

    if (ray.TMax <= ray.TMin)
        return float4(0.0, 0.0, 0.0, 1.0);

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(r.tlas, RAY_FLAG_NONE, 0xff, ray);
    // Geometry is flagged opaque, so this commits the closest hit and exits.
    while (q.Proceed()) {}

    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        Prim hit = loadPrim(r.scene, int(r.tri_prim[int(q.CommittedPrimitiveIndex())]));
        // Step back along the ray so the feedback is read just outside the surface, where the
        // fluence is meaningful.
        float2 outside = origin + dir * (q.CommittedRayT() - 1.5);
        return float4(hit.emission + reflected(r, outside, hit.albedo), 0.0);
    }
    return float4(0.0, 0.0, 0.0, 1.0);
}

// Step from inside a primitive to just outside it along its normal. Sampling the field inside a
// solid gives zero, which renders every occluder as a black silhouette. Every primitive here is
// convex, so one step lands outside that primitive; a single check catches the case where it lands
// inside a neighbouring one.
bool surfaceSample(Root* r, Prim prim, float2 p, float d, out float2 outside)
{
    float2 n = primNormal(prim, p);
    if (dot(n, n) < 1e-10)
    {
        outside = p;
        return false;
    }
    outside = p + n * (-d + r.surface_offset);
    return sceneEvalGrid(r, outside).d > 0.0;
}
