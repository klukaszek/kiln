# Holographic Radiance Cascades

Interactive 2D global illumination against an **analytic** signed-distance scene, traced with
**hardware ray query**. Implements Holographic Radiance Cascades (Freeman, Sannikov & Margel,
[arXiv:2505.02041](https://arxiv.org/abs/2505.02041)).

A port of my old SlangPy implementation onto Kiln. The solver is the same but what has changed is how the GPU
is addressed, which is most of what makes this worth reading — see [The port](#the-port).

```bash
cargo run --release -p hrc
cargo run --release -p hrc -- --scene slit --probes 600 --no-vsync
```

Drag to move a light, `1`–`4` scene, space pause, esc quit. Everything else is in the window.

## Construction

Standard radiance cascades halve probe resolution in *both* axes per level, so a distant thin
occluder is resolved on a grid coarser than the occluder itself. HRC reduces resolution only
*along* the gathered direction. Fluence splits into four quadrants, each solved in a canonical
frame whose +x axis is the gather direction, then summed.

Two structures on probes `p = (x·2ⁿ, y)`, with `vₙ(k) = (2ⁿ, 2k − 2ⁿ)`:

* `Tₙ(p, k)` — radiance/transmittance of the segment `p → p + vₙ(k)`
* `Rₙ(p, i)` — angular fluence of the cone between `vₙ(i−½)` and `vₙ(i+½)`

`T` is built bottom-up; only levels 0–2 are ray traced, and those segments span at most four
pixels. `R` is swept top-down and casts no rays. Total ~19 short rays per output pixel.

Details that matter, all of them load-bearing:

* The `R₀` lookup is offset one probe (`R₀([x+1, y], 0)`), or the diagonal edge rays of adjacent
  quadrants overlap and every light gets a bright cross.
* That offset needs an opacity guard. One pixel along the gather axis reads through a nearby
  surface, bleeding light into occluders and darkness out — a bright rim outlining every shape.
  Where it would cross a boundary, fall back to the probe itself.
* Surfaces are shaded from the light field just *outside* them, found by stepping along the SDF
  gradient. Sampling inside a solid gives zero and renders every occluder as a black silhouette.
* The cross blur (paper eq. 21) removes the checkerboard from `vₙ(k)` being even for `n ≥ 1`,
  skipping neighbours of differing opacity. Two passes is the default.
* Surface shading falls off with depth into the solid. Projecting every body pixel to its nearest
  surface point otherwise smears the field's residual roughness into perfectly correlated bands
  across each flat face.
* Half storage bounds the usable radiance window to about `[1e-4, 1e4]`. Stores saturate at 65000
  rather than producing `inf` — without that clamp, `inf` meets a zero transmittance in the merge
  and turns the entire field to `NaN`.

Surfaces re-emit what reached them last frame: a ray that hits a primitive returns
`emission + albedo · fluence/2π` read from the previous frame's field just outside the hit. The
loop is only stable for gain ≤ 1, so the slider is capped there.

## The port

The solver is a transcription. The interesting differences are all in how data reaches the GPU.

**Tensors became one root struct.** SlangPy passed `ITensor<T, N>` parameters and generated the
indexing; Kiln has no such thing, so every buffer is a device pointer on a single `Root` that
`gpu_struct!` declares once for both Rust and Slang. There are about a hundred dispatches per
frame and they overlap heavily in what they read, so one root shared by every entry point — with
three or four fields rewritten per dispatch, out of a per-frame bump arena — beat a struct per
pass. The tensor subscripts slangpy generated are written out as explicit index arithmetic; both
cascades stay (row, column, direction) with direction innermost, so threads adjacent in a
threadgroup touch adjacent addresses.

**`spy.call_id()` became explicit dispatch shapes.** Each pass unflattens its own index: cascade
passes flatten (direction, column) across a 64-wide threadgroup and take the row from the
dispatch's second axis, because a level's direction count runs from 2 to ~1000 and a 3D dispatch
would leave most of a threadgroup idle at both ends.

**No `float3` in the root.** Slang gives it 12 bytes on SPIR-V and 16 on Metal, so a `float3`
field would silently disagree across backends. Colours are `float4`; the BVH vertex buffer is
written as bare floats at a 12-byte stride, which is what the acceleration build reads.

**Half storage survived unchanged.** `float16_t4` through a raw device pointer compiles on both
targets with no extra capability, so `T` and `R` keep the precision the original measured as free.

**The resolve is a fragment shader.** The original wrote an RGBA16F texture from compute and
blitted it. Kiln has no blit, and the pass is one dependent read per pixel, so it draws a
fullscreen triangle straight into the swapchain image inside the harness's render pass.

**One allocation behind the cascades.** Every `T` and `R` level is an offset into a single
`GpuOnly` allocation, which makes the memory plan and the teardown one number and one call each.

**Acceleration structures interleave with compute.** Both are sized once for a full-capacity scene
and rebuilt every frame over buffers a compute pass just wrote. The original needed a separate
command buffer for this, because slangpy would not let an acceleration-structure pass interleave
with the compute passes it opened; `cmd.build_blas` ends whatever encoder is open and runs the
build in its own, so it goes in the frame's one command buffer.

Two things changed behaviour rather than plumbing:

* **The sky works.** The original exposed a sky slider that fed nothing: escaped rays carried no
  energy and `R_N` was left at zero, so the value never entered the solver. `R_N` is the far end of
  every merge, which is exactly where a uniform background belongs, so it is now seeded with
  `sky · arc` each frame and the sweep attenuates it by each cone's transmittance for free. It also
  removes the one buffer that had to be cleared at allocation.
* **Temporal supersampling is gone.** It defaulted off in the original and its README records the
  measurement: the cascade band-limits the field to the probe grid by construction, so sub-probe
  sites carry ~0.008× the variation of the probes themselves. Dropping it removes the jitter, the
  subpixel phase, the accumulation blend and one of the two field buffers.

**The display curve is a plain 2.2 gamma, and the swapchain's own is undone to get it.** The
original wrote `pow(c, 1/2.2)` into a non-sRGB target; Kiln's harness presents to an sRGB one,
which applies a transfer function with a linear toe that 2.2 does not have. It crushes everything
below about 1% luminance — 20% darker at 0.01, more than twice as dark at 0.002 — and a global
illumination frame is mostly indirect light sitting in exactly that range, so writing linear into
it visibly darkens the whole image. The exposure and falloff defaults were chosen against 2.2, so
the resolve encodes with 2.2 and hands the hardware the linear value that reproduces it.

Not ported: `verify.py`'s closed-form disc comparison. The brute-force reference it shares a harness
with *is* here, as `--verify`.

## Barriers

Every pass reads what the one before it wrote, so the frame is a chain of
`cmd.barrier(COMPUTE, COMPUTE)`. Three places need more than that:

* **Top of frame,** `ALL_COMMANDS → ALL_COMMANDS`, recorded before any encoder opens so it lands
  as a queue-scoped barrier at the head of the first one. Every cascade and field buffer is shared
  across frames-in-flight, so this is what stops frame *n* writing what frame *n−1* is still
  reading.
* **Around the acceleration build,** `COMPUTE → ALL_COMMANDS` then `ALL_COMMANDS → COMPUTE`. The
  build runs in its own encoder, and an encoder-scoped barrier does not reach across that.
* **Before the resolve,** `COMPUTE → PIXEL_SHADER`, which is queue-scoped for the same reason.

## Verification

The solver is an approximation, so the example carries the thing that says by how much: a dense
per-pixel angular integral of the same scene, sphere-marched against the signed-distance functions
directly. It shares neither the BVH nor the broadphase with the solver, so agreement between them
checks the tessellation too.

```bash
cargo run --release -p hrc -- --verify --scene penumbra --res 512 --dirs 1024 --save
```

Against a 1024-direction-per-pixel integral at 512²:

| scene | relative L1 | energy ratio |
|---|---:|---:|
| Slit | 0.0105 | 1.005 |
| Penumbra | 0.0337 | 0.992 |
| Many lights | 0.0388 | 0.979 |

`--save` writes `rc.png`, `reference.png` and `error.png`. Every optimisation below was landed only
once these numbers came back bit-identical.

## Performance

2.46 ms/frame on an M4 Pro at the default 150k probes (447×335) resolving 1600×1200, and 4.45 ms if
you push the slider to the original's 315k.

**The frame cost has nothing to do with the scene.** Geometry enters in exactly two places — the ray
queries, which measure 0.46 ms for all six million of them, and the resolve, at about 0.2 ms. The
other ~3.7 ms is the cascade construction, which is scene-independent by design: 315k probes × 11
levels × 4 quadrants is ~15 M stored entries, each merged from four to six dependent loads. A
scene of 26 primitives and a scene of 26,000 cost the same.

So probe count is the only real lever, and it is a slider and a `--probes` flag. What it buys is
measurable — solve the field coarsely, reconstruct it at a fixed output resolution, and compare
against the reference there:

| probes | 36k | 73k | 147k | 295k | 589k |
|---|---:|---:|---:|---:|---:|
| relative L1 | 0.100 | 0.091 | 0.054 | 0.043 | 0.029 |
| ms | ~1.0 | ~1.7 | 2.46 | 4.45 | ~8.5 |

The knee is around 150k, which is the default. The original shipped 315k, chosen to fill a 120 Hz
frame rather than because the solver needed it: it costs 75% more time to take relative L1 from
0.054 to 0.043, and most of what is left at that point is the method's own approximation, not probe
density. `--verify --probe-res N --res M` is what produced that table.

The budget is a probe *count*, not a fixed side: cost is set by how many probes exist, so anything
else makes frame time swing with the window's aspect. It backs off automatically when the cascades
would exceed 1 GiB.

### What moved it, and what did not

From 5.29 ms, in the order the measurements justified:

| change | gain |
|---|---:|
| Top cascade evaluated instead of stored | 5.5% |
| Clearance early-out on traced segments | 4.7% |
| Exterior probe column evaluated instead of stored | 4.2% |
| Six-byte cone storage (also −28% memory) | 1.8% |

The negative results are worth as much, because they say what this workload is *not*:

* **Barriers are not the cost.** Removing every one of the ~80 barriers in the cascade — producing
  garbage, but pricing them — saves 0.25 ms. They run about 3 µs each.
* **Nor is serialisation.** Quadrants 0/1 and 2/3 own different buffers and can be interleaved,
  halving the barrier count and doubling the work in flight per step: 1%. Reverted.
* **Nor bandwidth.** Cutting `R` from eight bytes to six removes a quarter of the most-read
  structure's traffic and buys 1.8% — the loads cost what they cost, largely regardless of width.
* **Nor coalescing.** Transposing both cascades to (direction, row, column), so a threadgroup's
  gathers stop straddling rows, measured slightly *slower*. Reverted.
* **Nor address arithmetic.** Replacing the per-invocation integer divide with a mask and a shift:
  nil.
* **Nor occupancy.** 32 threads per group is 18% slower than 64; 128 and 256 are slower again,
  because meeting them costs register spills.
* **Nor the BVH.** `PREFER_FAST_TRACE` and quartering the tessellation density both measured nil,
  which is what sent the search away from the ray queries in the first place.

What is left is dependent-load latency over those 15 M entries, and the way to move it is to want
fewer probes.

## Layout

```
src/main.rs                       CLI and entry point
src/app.rs                        harness lifecycle, clock, per-frame scene
src/ui.rs                         the control window, keys, and pointer drag
src/scene.rs                      analytic scenes, and packing them for the GPU
src/cascades/mod.rs               allocation, the pass schedule, and the per-dispatch roots
src/cascades/program.rs           the root struct, assembled sources, and pipelines
src/cascades/resources.rs         packed scene, BVH geometry, broadphase, BLAS/TLAS
src/cascades/shaders/scene.slang        SDF primitives, tessellation, ray query
src/cascades/shaders/holographic.slang  T and R construction, resolve, scene preparation
src/cascades/shaders/clear.slang        one-time zeroing of the fluence field
src/verify.rs                     brute-force reference, and the statistics against it
```
