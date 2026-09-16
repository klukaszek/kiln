// Holographic Radiance Cascades (Freeman, Sannikov & Margel, arXiv:2505.02041).
//
// Two structures are built over probes `p = (x*2^n, y)`, with `v_n(k) = (2^n, 2k - 2^n)`:
//
//   T_n(p, k)  radiance/transmittance of the segment `p -> p + v_n(k)`
//   R_n(p, i)  angular fluence of the cone between `v_n(i-1/2)` and `v_n(i+1/2)`
//
// `T` is built bottom-up and only levels 0..DIRECT_TRACE_LEVELS-1 are traced; `R` is swept
// top-down and casts no rays. Fluence splits into four quadrants, each solved in a canonical frame
// whose +x axis is the gather direction, then summed.
//
// Ported from the SlangPy original. slangpy generated one kernel per function from `call_id()`;
// here each pass is an explicit entry point that unflattens its own dispatch index, and the
// tensors it read are device pointers on the root struct.

// --- canonical frame --------------------------------------------------------

void quadFrame(Root* r, out int2 ex, out int2 ey, out int2 org)
{
    int w = r.res.x - 1;
    int h = r.res.y - 1;
    if (r.quadrant == 0) { ex = int2(1, 0); ey = int2(0, 1); org = int2(0, 0); }
    else if (r.quadrant == 1) { ex = int2(0, 1); ey = int2(-1, 0); org = int2(w, 0); }
    else if (r.quadrant == 2) { ex = int2(-1, 0); ey = int2(0, -1); org = int2(w, h); }
    else { ex = int2(0, -1); ey = int2(1, 0); org = int2(0, h); }
}

float2 canonToWorld(Root* r, float2 uv)
{
    int2 ex, ey, org;
    quadFrame(r, ex, ey, org);
    return float2(org) + uv.x * float2(ex) + uv.y * float2(ey) + 0.5;
}

// Offset to the next probe in direction k at level n.
int2 stepVector(int n, int k)
{
    int s = 1 << n;
    return int2(s, 2 * k - s);
}

// Angular size of cone index ci at level n; its edges are v_n(ci), v_n(ci+1).
float coneArc(int n, int ci)
{
    float2 a = float2(stepVector(n, ci));
    float2 b = float2(stepVector(n, ci + 1));
    return atan2(a.x * b.y - a.y * b.x, a.x * b.x + a.y * b.y);
}

// Probe columns *stored* at level n.
//
// The construction reads one column past the end -- the merge reprojects to `nxi + 1`, and the
// resolve's R_0 lookup is offset a probe. That column always sits at or beyond the domain edge,
// and every direction at every level points further out from there, so it holds nothing a ray
// could find: its segments all miss and its cones carry only sky. It is evaluated on the two
// lookup paths below instead of being stored, which is 14% of all cascade entries, and the largest
// share of it at the top levels where the column count has bottomed out at one but the direction
// count is still doubling.
int levelWidth(int canonX, int n)
{
    return (canonX + (1 << n) - 1) >> n;
}

// --- storage ----------------------------------------------------------------

// Half has a hard ceiling at 65504. A cone's fluence is its arc times the radiance in it, so
// emission above ~4e4 overflows to inf, and the inf then meets a zero transmittance in the merge
// and turns the whole field to NaN. Saturate on store instead.
static const float HALF_MAX = 65000.0;

Half4 storeHalf(float4 v)
{
    return Half4(clamp(v, -HALF_MAX, HALF_MAX));
}

Half3 storeCone(float3 v)
{
    float3 c = clamp(v, -HALF_MAX, HALF_MAX);
    Half3 out;
    out.x = float16_t(c.x);
    out.y = float16_t(c.y);
    out.z = float16_t(c.z);
    return out;
}

float3 loadCone(Ptr<Half3, Access.Read> cones, int index)
{
    Half3 v = cones[index];
    return float3(float(v.x), float(v.y), float(v.z));
}

// Both cascades are indexed (y, x, direction) with direction innermost, so threads adjacent in the
// dispatch touch adjacent addresses.
int cascadeIndex(int xi, int y, int d, int width, int dirs, int height)
{
    int cx = clamp(xi, 0, width - 1);
    int cy = clamp(y, 0, height - 1);
    int cd = clamp(d, 0, dirs - 1);
    return (cy * width + cx) * dirs + cd;
}

// Past the last stored column the segment starts outside the domain and points away from it, so
// nothing is in the way: no radiance, full transmittance.
float4 loadT(Ptr<Half4, Access.Read> t, int xi, int y, int k, int width, int dirs, int height)
{
    if (xi >= width)
        return float4(0.0, 0.0, 0.0, 1.0);
    return float4(t[cascadeIndex(xi, y, k, width, dirs, height)]);
}

float3 loadR(Ptr<Half3, Access.Read> r, int xi, int y, int ci, int width, int cones, int height)
{
    return loadCone(r, cascadeIndex(xi, y, ci, width, cones, height));
}

// The far cone of a merge at `level`.
//
// R_N is the far end of every merge and holds nothing but the sky, uniformly across the whole
// domain — so it is evaluated here rather than stored. That removes the top cascade's buffer, the
// pass that filled it, and the reads the level below made of it, which between them were the
// single largest block of traffic in the sweep.
float3 farCone(int levels, float3 sky, int level, Ptr<Half3, Access.Read> cones,
               int xi, int y, int ci, int width, int coneCount, int height)
{
    // Two cascades are uniform rather than stored: the top one, which is the far end of every
    // merge and holds only the sky, and the column past the domain edge, which sees only sky too.
    if (level >= levels || xi >= width)
        return sky * coneArc(level, ci);
    return loadR(cones, xi, y, ci, width, coneCount, height);
}

// Merge(<rn,tn>, <rf,tf>) = <rn + tn*rf, tn*tf>   (eq. 7)
float4 mergePair(float4 near, float4 far)
{
    return float4(near.rgb + near.a * far.rgb, near.a * far.a);
}

// Merge_r(<rn,tn>, rf) = rn + tn*rf, with the near radiance first converted to angular fluence
// over arc A   (eq. 8, and the A x <r,t> = <A r, t> convention)
float3 mergeFluence(float arc, float4 near, float3 farFluence)
{
    return arc * near.rgb + near.a * farFluence;
}

// Whether a segment provably misses every primitive, from one lookup rather than a ray.
//
// Only levels 0..DIRECT_TRACE_LEVELS-1 are traced, so a traced segment spans at most
// `sqrt(2) * 2^(DIRECT_TRACE_LEVELS-1)` pixels — four probes, by construction. Almost all of them
// sit in open space, and the scene is analytic, so the broadphase can carry a distance the BVH
// cannot: `cell_clear` bounds how far every point of a cell is from any surface, so a segment whose
// midpoint lands in a cell clearer than its own half-length cannot touch anything.
//
// This is exact, not an approximation. `primDist` is a true exterior distance and the tessellated
// outline is inscribed in it, so the polygon a ray would hit is never nearer than the bound. A
// segment that fails the test is traced as before.
bool segmentIsClear(Root* r, float2 mid, float halfLength)
{
    // Segments near the domain edge can have a midpoint outside the grid. Rather than widen the
    // bound to cover that, hand those few back to the ray query.
    int2 cell = int2(floor(float2(r.cells) * mid / float2(r.res)));
    if (any(cell < int2(0, 0)) || any(cell >= r.cells))
        return false;
    return r.cell_clear[cell.y * r.cells.x + cell.x] > halfLength;
}

// --- T: traced segments -----------------------------------------------------

// Levels 0..DIRECT_TRACE_LEVELS-1, traced directly. These segments span at most four probes.
[shader("compute")]
[numthreads(CASCADE_THREADS, 1, 1)]
void traceSegments(uint3 tid : SV_DispatchThreadID, uniform Root* r)
{
    int n = r.level;
    int dirs = (1 << n) + 1;
    int width = levelWidth(r.canon_res.x, n);
    int y = int(tid.y);
    if (int(tid.x) >= dirs * width || y >= r.canon_res.y)
        return;
    int k = int(tid.x) % dirs;
    int xi = int(tid.x) / dirs;

    float2 p = float2(int2(xi << n, y));
    float2 a = canonToWorld(r, p);
    float2 b = canonToWorld(r, p + float2(stepVector(n, k)));

    float2 seg = b - a;
    float len = length(seg);
    float4 result = segmentIsClear(r, 0.5 * (a + b), 0.5 * len)
        ? float4(0.0, 0.0, 0.0, 1.0)
        : traceIntervalRT(r, a, seg / max(len, 1e-6), 0.0, len);
    r.segments_out[(y * width + xi) * dirs + k] = storeHalf(result);
}

// Levels DIRECT_TRACE_LEVELS and up, built by merging two half-length segments (eq. 18 and 20).
[shader("compute")]
[numthreads(CASCADE_THREADS, 1, 1)]
void buildSegments(uint3 tid : SV_DispatchThreadID, uniform Root* r)
{
    int n = r.level;
    int2 canonRes = r.canon_res;
    Ptr<Half4, Access.Read> prev = r.segments_prev;

    int dirs = (1 << n) + 1;
    int width = levelWidth(canonRes.x, n);
    int y = int(tid.y);
    if (int(tid.x) >= dirs * width || y >= canonRes.y)
        return;
    int m = int(tid.x) % dirs;
    int xi = int(tid.x) / dirs;

    int pw = levelWidth(canonRes.x, n - 1);
    int pd = (1 << (n - 1)) + 1;
    int h = canonRes.y;
    int pxi = 2 * xi;  // same probe, indexed one level down

    float4 result;
    if ((m & 1) == 0)
    {
        // v_n(2k) = 2 v_{n-1}(k): exactly the near and far halves.
        int k = m >> 1;
        int2 v = stepVector(n - 1, k);
        result = mergePair(
            loadT(prev, pxi, y, k, pw, pd, h),
            loadT(prev, pxi + 1, y + v.y, k, pw, pd, h));
    }
    else
    {
        // Odd m has no integer direction one level down, so blend the two closest two-segment
        // paths: out along k+1/2 then back along k-1/2, and vice versa.
        int kp = (m + 1) >> 1;
        int km = (m - 1) >> 1;
        float4 fp = mergePair(
            loadT(prev, pxi, y, kp, pw, pd, h),
            loadT(prev, pxi + 1, y + stepVector(n - 1, kp).y, km, pw, pd, h));
        float4 fm = mergePair(
            loadT(prev, pxi, y, km, pw, pd, h),
            loadT(prev, pxi + 1, y + stepVector(n - 1, km).y, kp, pw, pd, h));
        result = 0.5 * (fp + fm);
    }
    r.segments_out[(y * width + xi) * dirs + m] = storeHalf(result);
}

// --- R: cone fluence --------------------------------------------------------

// R_n(p, i) = F+ + F-, each half of the cone approximated by a thinner cone one level up (eq. 14
// for odd x, eq. 15-17 for even x). Casts no rays.
[shader("compute")]
[numthreads(CASCADE_THREADS, 1, 1)]
void buildFluence(uint3 tid : SV_DispatchThreadID, uniform Root* r)
{
    int n = r.level;
    int levels = r.levels;
    int2 canonRes = r.canon_res;
    float3 sky = r.sky.rgb;
    Ptr<Half3, Access.Read> conesNext = r.cones_next;
    Ptr<Half4, Access.Read> segmentsCur = r.segments_cur;
    Ptr<Half4, Access.Read> segmentsNext = r.segments_next;

    int cones = 1 << n;
    int width = levelWidth(canonRes.x, n);
    int y = int(tid.y);
    if (int(tid.x) >= cones * width || y >= canonRes.y)
        return;
    int ci = int(tid.x) & (cones - 1);
    int xi = int(tid.x) >> n;

    int h = canonRes.y;

    // Cone ci at level n splits into cones 2ci and 2ci+1 at level n+1; its edges are directions ci
    // and ci+1.
    int cjm = 2 * ci;
    int cjp = 2 * ci + 1;
    float arcM = coneArc(n + 1, cjm);
    float arcP = coneArc(n + 1, cjp);

    int rw = levelWidth(canonRes.x, n + 1);
    int rc = 1 << (n + 1);

    float3 fluenceM, fluenceP;

    if ((xi & 1) != 0)
    {
        // No level n+1 probe here: trace out along each cone edge until one is reached, then merge
        // with the closest cone there.
        int tw = levelWidth(canonRes.x, n);
        int td = (1 << n) + 1;
        int nxi = (xi + 1) >> 1;
        float4 tm = loadT(segmentsCur, xi, y, ci, tw, td, h);
        float4 tp = loadT(segmentsCur, xi, y, ci + 1, tw, td, h);

        float3 rm = farCone(levels, sky, n + 1, conesNext, nxi, y + stepVector(n, ci).y, cjm, rw, rc, h);
        float3 rp = farCone(levels, sky, n + 1, conesNext, nxi, y + stepVector(n, ci + 1).y, cjp, rw, rc, h);

        fluenceM = mergeFluence(arcM, tm, rm);
        fluenceP = mergeFluence(arcP, tp, rp);
    }
    else
    {
        // A level n+1 probe sits here, but using it alone biases the result toward the cone centre,
        // since the odd-x branch biases toward the edges. Average it with a reprojection to the
        // next probe along.
        int tw = levelWidth(canonRes.x, n + 1);
        int td = (1 << (n + 1)) + 1;
        int nxi = xi >> 1;

        float3 rm0 = farCone(levels, sky, n + 1, conesNext, nxi, y, cjm, rw, rc, h);
        float3 rp0 = farCone(levels, sky, n + 1, conesNext, nxi, y, cjp, rw, rc, h);

        float4 tm = loadT(segmentsNext, nxi, y, cjm, tw, td, h);
        float4 tp = loadT(segmentsNext, nxi, y, cjp + 1, tw, td, h);
        float3 rm1 = farCone(levels, sky, n + 1, conesNext, nxi + 1, y + stepVector(n + 1, cjm).y, cjm, rw, rc, h);
        float3 rp1 = farCone(levels, sky, n + 1, conesNext, nxi + 1, y + stepVector(n + 1, cjp + 1).y, cjp, rw, rc, h);

        fluenceM = 0.5 * (rm0 + mergeFluence(arcM, tm, rm1));
        fluenceP = 0.5 * (rp0 + mergeFluence(arcP, tp, rp1));
    }

    r.cones_out[(y * width + xi) * cones + ci] = storeCone(fluenceM + fluenceP);
}

// --- resolve ----------------------------------------------------------------

int2 canonResFor(Root* r, int q)
{
    return (q & 1) == 0 ? r.res : int2(r.res.y, r.res.x);
}

// World position of the probe one step along this quadrant's gather axis.
int2 canonOffsetToWorld(int q, int2 world)
{
    if (q == 0) return world + int2(1, 0);
    if (q == 1) return world + int2(0, 1);
    if (q == 2) return world - int2(1, 0);
    return world - int2(0, 1);
}

// Inverse of canonToWorld, so the four quadrants can be gathered per pixel rather than scattered.
int2 worldToCanon(Root* r, int2 world, int q)
{
    int w = r.res.x - 1;
    int h = r.res.y - 1;
    if (q == 0) return world;
    if (q == 1) return int2(world.y, w - world.x);
    if (q == 2) return int2(w - world.x, h - world.y);
    return int2(h - world.y, world.x);
}

// R_0 has one cone per quadrant, so its width is canonRes.x + 1 and x clamps inclusively.
float3 sampleQuadrant(Root* r, int q, Ptr<Half3, Access.Read> r0, int2 uv)
{
    int2 cr = canonResFor(r, q);
    int y = clamp(uv.y, 0, cr.y - 1);
    // The offset lookup reaches one probe past the edge, which is the unstored column.
    if (uv.x >= cr.x)
        return r.sky.rgb * coneArc(0, 0);
    return loadCone(r0, y * cr.x + max(uv.x, 0));
}

// R_0([x + 1, y], 0), not R_0([x, y], 0). Without the offset the diagonal edge rays of adjacent
// quadrants overlap and every light gets a bright cross through it (paper, sec. 4.2). The offset
// probe is one pixel along the gather axis though, so next to a surface it reads through it; fall
// back to the probe itself wherever it would cross a boundary.
float3 quadrantFluence(Root* r, int q, Ptr<Half3, Access.Read> r0, int2 world, bool inside)
{
    int2 uv = worldToCanon(r, world, q);
    int2 offsetWorld = canonOffsetToWorld(q, world);

    bool offsetInside = sceneEvalGrid(r, float2(offsetWorld) + 0.5).d < 0.0;
    return sampleQuadrant(r, q, r0, offsetInside == inside ? uv + int2(1, 0) : uv);
}

// Level 0 holds one cone per quadrant at every pixel; the four sum to the full fluence. Alpha
// carries opacity so the blur below can avoid mixing across surfaces.
[shader("compute")]
[numthreads(FIELD_THREADS_X, FIELD_THREADS_Y, 1)]
void resolveFluence(uint3 tid : SV_DispatchThreadID, uniform Root* r)
{
    int2 probe = int2(tid.xy);
    if (any(probe >= r.res))
        return;

    bool inside = sceneEvalGrid(r, float2(probe) + 0.5).d < 0.0;
    float3 f = quadrantFluence(r, 0, r.cones0_q0, probe, inside)
             + quadrantFluence(r, 1, r.cones0_q1, probe, inside)
             + quadrantFluence(r, 2, r.cones0_q2, probe, inside)
             + quadrantFluence(r, 3, r.cones0_q3, probe, inside);

    r.fluence_out[probe.y * r.res.x + probe.x] = float4(f, inside ? 1.0 : 0.0);
}

// HRC leaves a checkerboard: v_n(k) is even for every n >= 1, so probes with odd and even y never
// interact (paper, eq. 21). Skips neighbours of differing opacity so it does not bleed across
// surfaces.
[shader("compute")]
[numthreads(FIELD_THREADS_X, FIELD_THREADS_Y, 1)]
void blurFluence(uint3 tid : SV_DispatchThreadID, uniform Root* r)
{
    int2 id = int2(tid.xy);
    if (any(id >= r.res))
        return;

    float4 centre = r.fluence_in[id.y * r.res.x + id.x];
    float3 sum = 4.0 * centre.rgb;
    float weight = 4.0;
    int2 offsets[4] = { int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1) };
    [ForceUnroll]
    for (int i = 0; i < 4; ++i)
    {
        int2 q = clamp(id + offsets[i], int2(0, 0), r.res - 1);
        float4 n = r.fluence_in[q.y * r.res.x + q.x];
        if (n.a == centre.a)
        {
            sum += n.rgb;
            weight += 1.0;
        }
    }
    r.fluence_out[id.y * r.res.x + id.x] = float4(sum / weight, centre.a);
}

// --- display ----------------------------------------------------------------

// Inverse of the sRGB transfer function, used to cancel the one the swapchain applies.
float3 srgbToLinear(float3 c)
{
    float3 low = c / 12.92;
    float3 high = pow(max(c + 0.055, 0.0) / 1.055, 2.4);
    return lerp(low, high, step(float3(0.04045), c));
}

// Exposure, Reinhard, then a plain 2.2 gamma.
//
// The gamma is deliberate, and so is undoing the swapchain's own curve to get it. sRGB's transfer
// function has a linear toe that pure 2.2 does not, and it crushes everything below about 1%
// luminance -- roughly 20% darker at 0.01 and over twice as dark at 0.002. In a renderer whose
// frame is mostly indirect light sitting in exactly that range, writing linear into an sRGB target
// visibly darkens the whole image, and this demo's exposure and falloff defaults were chosen
// against 2.2. So encode with 2.2 and hand the hardware the linear value that reproduces it.
float3 tonemap(Root* r, float3 c)
{
    c = max(c * r.exposure, 0.0);
    c = c / (1.0 + c);
    c = pow(c, 1.0 / 2.2);
    return r.target_is_srgb != 0 ? srgbToLinear(c) : c;
}

float2 toCascadeSpaceIn(int2 id, int2 res, int2 outRes)
{
    return (float2(id) + 0.5) * (float2(res) / float2(outRes));
}

float2 toCascadeSpace(int2 id, Root* r)
{
    return toCascadeSpaceIn(id, r.res, r.out_res);
}

// Bilinear lookup of the fluence field.
//
// `side` is the opacity of the point being shaded. Probes inside a solid hold almost no fluence, so
// blending them with lit probes outside darkens a band about one probe wide around every surface --
// the halo that makes geometry look like it sits in a pool of shadow, and the reason it shrinks
// when the probe grid is refined. Drop taps on the other side of a surface and renormalise, exactly
// as the cross blur already does.
float3 sampleField(Ptr<float4, Access.Read> fluence, int2 res, float2 pos, float side)
{
    float2 uv = pos - 0.5;
    float2 base = floor(uv);
    float2 f = uv - base;
    int2 b = int2(base);

    // Smoothstep the weights: plain bilinear is C0 but not C1, so its gradient jumps at every probe
    // boundary. Measured, this did not reduce the banding that motivated it (depth falloff did) --
    // kept only because a C1 interpolant costs one multiply-add and is strictly better behaved.
    f = f * f * (3.0 - 2.0 * f);

    float3 sum = float3(0.0);
    float3 all = float3(0.0);
    float weight = 0.0;
    for (int oy = 0; oy < 2; ++oy)
    {
        for (int ox = 0; ox < 2; ++ox)
        {
            float w = (ox == 0 ? 1.0 - f.x : f.x) * (oy == 0 ? 1.0 - f.y : f.y);
            int2 q = clamp(b + int2(ox, oy), int2(0, 0), res - 1);
            float4 v = fluence[q.y * res.x + q.x];
            all += w * v.rgb;
            if (v.a == side)
            {
                sum += w * v.rgb;
                weight += w;
            }
        }
    }
    // Features thinner than a probe can leave no tap on the right side.
    return weight > 1e-3 ? sum / weight : all;
}

float3 sampleFluence(Root* r, float2 pos, float side)
{
    return sampleField(r.fluence_in, r.res, pos, side);
}

struct VOut
{
    float4 pos : SV_Position;
};

[shader("vertex")]
VOut resolveVs(uint vid : SV_VertexID)
{
    float2 p = float2(float((vid << 1u) & 2u), float(vid & 2u));
    VOut o;
    o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

// The original wrote an RGBA16F texture from compute and blitted it. Kiln has no blit and the pass
// is one dependent read per pixel, so it resolves straight into the swapchain image instead.
[shader("fragment")]
float4 resolveFs(VOut i, uniform Root* r) : SV_Target
{
    float2 pixel = toCascadeSpace(int2(i.pos.xy), r);

    if (r.view_mode != 0)
        return float4(tonemap(r, sampleFluence(r, pixel, 0.0) / TWO_PI), 1.0);

    SceneHit h = sceneEvalGrid(r, pixel);
    float3 light = sampleFluence(r, pixel, h.d < 0.0 ? 1.0 : 0.0) / TWO_PI;
    float3 col = light;
    if (h.d < 0.0)
    {
        Prim prim = loadPrim(r.scene, h.id);
        float2 surface;
        float3 incident = light;
        // Light reaching a point inside an opaque body falls off with depth. Without this every
        // body pixel takes the value of its nearest surface point, smearing the field's residual
        // roughness into correlated bands across each face.
        //
        // surface_falloff is a fraction of the scene, not a pixel count: in probe units the lit rim
        // would shrink as the probe grid is refined, so more probes would make every body blacker.
        float falloff = max(r.surface_falloff * float(min(r.res.x, r.res.y)), 1e-3);
        if (r.surface_shading != 0 && surfaceSample(r, prim, pixel, h.d, surface))
            incident = exp(h.d / falloff) * sampleFluence(r, surface, 0.0) / TWO_PI;
        col = prim.emission + prim.albedo * incident;
    }
    return float4(tonemap(r, col), 1.0);
}

// --- GPU scene preparation --------------------------------------------------

// Tessellate every primitive into the extruded wall quads the BLAS is built over, one thread per
// outline edge. slangpy needed one kernel per output tensor; one entry point can write all three.
[shader("compute")]
[numthreads(CASCADE_THREADS, 1, 1)]
void tessellate(uint3 tid : SV_DispatchThreadID, uniform Root* r)
{
    int i = int(tid.x);
    if (i >= r.prim_count * EDGES_PER_PRIM)
        return;
    int prim = i / EDGES_PER_PRIM;
    int edge = i % EDGES_PER_PRIM;

    Prim p = loadPrimGeometry(r.scene, prim);
    float2 a = outlineVertex(p, edge);
    float2 b = outlineVertex(p, (edge + 1) % EDGES_PER_PRIM);

    // Four corners of the wall quad, as three tightly packed floats each: the BLAS reads the buffer
    // at a 12-byte stride, and a `float3` element would be padded to 16 on Metal.
    int base = i * 4;
    [ForceUnroll]
    for (int c = 0; c < 4; ++c)
    {
        float2 xy = (c == 1 || c == 2) ? b : a;
        r.vertices[(base + c) * 3 + 0] = xy.x;
        r.vertices[(base + c) * 3 + 1] = xy.y;
        r.vertices[(base + c) * 3 + 2] = c < 2 ? -1.0 : 1.0;
    }

    const int quad[6] = { 0, 1, 2, 0, 2, 3 };
    [ForceUnroll]
    for (int t = 0; t < 6; ++t)
        r.indices[i * 6 + t] = uint(base + quad[t]);

    // Both triangles of the quad belong to the primitive that produced it.
    r.tri_prim[i * 2 + 0] = uint(prim);
    r.tri_prim[i * 2 + 1] = uint(prim);
}

// One thread per cell: record which primitives reach it. Bucketed rather than CSR so no prefix sum
// is needed and the whole build stays on the GPU.
[shader("compute")]
[numthreads(FIELD_THREADS_X, FIELD_THREADS_Y, 1)]
void buildGrid(uint3 tid : SV_DispatchThreadID, uniform Root* r)
{
    int2 id = int2(tid.xy);
    if (any(id >= r.cells))
        return;

    float2 cellSize = float2(r.res) / float2(r.cells);
    float2 lo = float2(id) * cellSize;
    float2 centre = lo + 0.5 * cellSize;
    float radius = 0.5 * length(cellSize);

    uint n = 0;
    float nearest = 1e20;
    for (int i = 0; i < r.prim_count; ++i)
    {
        float d = primDist(loadPrimGeometry(r.scene, i), centre);
        nearest = min(nearest, d);
        // Conservative: a primitive matters to this cell if its surface can reach the cell's
        // bounding circle.
        if (d <= radius)
        {
            if (n < uint(GRID_CAPACITY))
                r.cell_prims[(id.y * r.cells.x + id.x) * GRID_CAPACITY + int(n)] = uint(i);
            ++n;
        }
    }
    // How far every point of this cell is guaranteed to be from any surface. `primDist` is an
    // exact exterior distance, and no point of the cell is more than `radius` from its centre, so
    // subtracting `radius` turns a centre distance into a bound that holds across the whole cell.
    r.cell_clear[id.y * r.cells.x + id.x] = nearest - radius;
    // The true overlap count, not the number stored, so a host-side check can detect a cell that
    // overflowed its bucket.
    r.cell_count[id.y * r.cells.x + id.x] = n;
}

// --- verification -----------------------------------------------------------
//
// The solver is an approximation: probes are spaced apart, directions are quantised, and intervals
// are stitched together with interpolation. These two passes are what that gets judged against —
// the solved field as linear mean radiance, and a dense per-pixel angular integral of the same
// scene. Ported from the original's `verify.py`, which is the only reason either exists.

// Linear mean radiance of the solved field, for comparison against the reference below.
[shader("compute")]
[numthreads(FIELD_THREADS_X, FIELD_THREADS_Y, 1)]
void lightFieldLinear(uint3 tid : SV_DispatchThreadID, uniform ReferenceRoot* r)
{
    int2 id = int2(tid.xy);
    if (any(id >= r.out_res))
        return;
    float2 pos = toCascadeSpaceIn(id, r.res, r.out_res);
    float3 mean = sampleField(r.fluence_in, r.res, pos, 0.0) / TWO_PI;
    r.fluence_out[id.y * r.out_res.x + id.x] = float4(mean, 1.0);
}

float2 refDir(int k, int count)
{
    float angle = TWO_PI * (float(k) + 0.5) / float(count);
    return float2(cos(angle), sin(angle));
}

// Dense per-pixel angular integral: the physical ground truth. Accumulates a slice of the direction
// set per dispatch, so a thousand directions per pixel does not become one command buffer long
// enough for the GPU to be reset out from under it. The host clears the target and divides by the
// total direction count.
[shader("compute")]
[numthreads(FIELD_THREADS_X, FIELD_THREADS_Y, 1)]
void referenceLinear(uint3 tid : SV_DispatchThreadID, uniform ReferenceRoot* r)
{
    int2 id = int2(tid.xy);
    if (any(id >= r.out_res))
        return;

    float2 pos = toCascadeSpaceIn(id, r.res, r.out_res);
    float3 acc = float3(0.0);
    for (uint k = 0; k < r.ref_dir_count; ++k)
    {
        float4 v = traceInterval(r.scene, r.prim_count, pos,
                                 refDir(int(r.ref_dir_begin + k), int(r.ref_dirs)),
                                 0.0, r.ref_max_dist, int(r.ref_max_steps), r.ref_eps);
        acc += v.rgb + v.a * r.sky.rgb;
    }
    int i = id.y * r.out_res.x + id.x;
    r.fluence_out[i] = float4(r.fluence_out[i].rgb + acc, 1.0);
}
