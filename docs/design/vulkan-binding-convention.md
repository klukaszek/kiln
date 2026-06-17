# Binding Convention — bindless acceleration structures & non-BDA globals (Vulkan / Metal parity)

**Status:** implemented (rev. 3) — verified on Vulkan (RTX 2070 SUPER, validation-clean) and Metal (M4, slangc 2026.10.2)
**Scope:** how acceleration structures and other opaque/non-BDA resources are bound across
the RHI, and the shader-authoring rules that keep one shader source compiling cleanly for
both the Vulkan (SPIR-V) and Metal (metal source) backends.

**Change from rev. 2:** the prototype (see §3a) showed the cleanest model is not a descriptor
heap at all. `DescriptorHandle<RaytracingAccelerationStructure>` lowers on **both** backends to
an AS handle carried **inline in the root struct** — on SPIR-V a 64-bit AS *device address*
converted via `OpConvertUToAccelerationStructureKHR`, on Metal an `acceleration_structure`
member read from `buffer(0)`. This is pure buffer-device-address, identical to how every other
resource already flows through the RHI: **no descriptor-set binding, no argument-table slot, no
registration API, no new heap.** Arbitrary numbers of acceleration structures are supported by
storing handles in a buffer and indexing dynamically.

**Change from rev. 1:** AS is bindless (by handle), not a fixed descriptor slot.

---

## 1. Background — why we're here

The Vulkan backend is bindless via `VK_EXT_descriptor_buffer`. Today the descriptor model is:

| Resource | Mechanism | Location (Vulkan) | Location (Metal arg table) |
|---|---|---|---|
| Bindless sampled images | descriptor buffer heap | set 0, binding **0** | slot **1** (heap base addr) |
| Bindless samplers | descriptor buffer heap | set 0, binding **1** | slot **2** (heap base addr) |
| Bindless storage images | descriptor buffer heap | set 0, binding **2** | (in texture heap) |
| Root data (per-draw/dispatch args) | push constant / `buffer(0)` | push const offset 0 (8-byte BDA) | slot **0** |
| Vertex/index/SSBO data | BDA pointer inside the root struct | — | — |

So **set 0 is the RHI's bindless heap**, the root pointer rides push constants (Vulkan) /
`buffer(0)` (Metal), and every data buffer is reached by dereferencing a `T*` (buffer device
address) carried in the root struct.

This works for graphics, mesh, and compute. It breaks for things that can't be a BDA pointer:

1. **Acceleration structures.** `RaytracingAccelerationStructure` is an opaque handle — it
   cannot be a `uint64_t` you dereference; Slang must emit it as a descriptor.
2. **Module-scope `uniform` globals** (e.g. `instanced_grid`'s `uniform GridCfg* cfg` declared
   at module scope instead of as an entry-point parameter).

### What Slang actually does (confirmed against docs)

- *Unannotated resources* get "the next unused binding number in descriptor set #0." An
  entry-point `uniform RaytracingAccelerationStructure tlas` therefore lands at **set 0,
  binding 0** — directly on top of our bindless sampled-image array.
- *Module-scope `uniform` globals* are collected into an implicit **`$Globals` cbuffer**, also
  placed at **set 0, binding 0** unless relocated. Same collision.
- *Entry-point* `uniform T*` parameters are different: they lower to a **push-constant pointer
  at offset 0** (Vulkan) / `buffer(0)` (Metal). This is the collision-free path our root data
  already rides. **Entry-point parameter vs. module-scope global is the whole ballgame.**
- *Bindless heaps* via `DescriptorHandle<T>` lower to `layout(set=S, binding=B) uniform T
  heap[]` on SPIR-V, with `S` set by `-bindless-space-index`. But `B` is auto-assigned per type
  (slang #8063) and AS support in `DescriptorHandle` is unconfirmed — so for AS we declare the
  heap **explicitly** (see §3.2) rather than depend on that.

Relevant Slang knobs: `[[vk::binding(binding,set)]]` (pin a resource; ignored on non-Vulkan
targets), `-fvk-bind-globals N M` (relocate `$Globals`), `-fvk-{b|s|t|u}-shift`,
`-bindless-space-index N`.

### Why bindless (not a fixed slot) for AS

The Metal backend currently reuses **argument-table slot 1 for both the texture heap and a
bound TLAS** (`refresh_argument_table` puts the texture heap at slot 1;
`bind_acceleration_structure(1, …)` overwrites slot 1 with the AS). That works only because RT
compute kernels happen not to use bindless textures — it would collide the moment a shader
needs both. A fixed-slot scheme just moves that fragility around. Referencing acceleration
structures **by 64-bit handle carried in buffers** (see §3) removes the ceiling entirely and
needs no descriptor/slot at all — it rides the BDA model the RHI already uses everywhere.

---

## 2. Design goals

1. **Set 0 stays the bindless heap.** The texture/sampler/storage-image path is validated;
   this must not be perturbed.
2. **Arbitrary count of acceleration structures**, addressable dynamically — no per-shader slot
   limit. (The explicit requirement: "bindless AS array is most certainly something we want.")
3. **One shader source, two backends.** A binding is authored once and lowers correctly for
   both SPIR-V and Metal.
4. **Metal/Vulkan parity by construction.**
5. **Sane defaults, minimal annotation** — common case needs no per-resource annotation.

---

## 3a. Prototype results (the deciding evidence)

Three shader forms of the one-triangle ray-query kernel were compiled with `slangc 2026.8`
(Vulkan SDK 1.4.350.0) to **both** `-target spirv` and `-target metal`:

| Form | SPIR-V | Metal | Notes |
|---|---|---|---|
| **A.** `[[vk::binding(3,0)]] RaytracingAccelerationStructure asHeap[]` indexed with `asHeap[NonUniformResourceIndex(id)]` | ✅ | ❌ | `NonUniformResourceIndex` is **unavailable in the Metal compute stage** (`error E36107`). |
| **A2.** same array, plain index `asHeap[id]` (no `NonUniformResourceIndex`) | ✅ (set 0, binding 3) | ✅ | Metal emits a *separate* `acceleration_structure asHeap[]` **kernel argument** → needs its own arg-table slot + a heap buffer of resource IDs. Vulkan needs a real AS descriptor in the heap (`vkGetDescriptorEXT`). Most plumbing. |
| **B.** `DescriptorHandle<RaytracingAccelerationStructure>` stored in the root struct | ✅ | ✅ | **Winner.** See below. |
| **C.** `DescriptorHandle<RaytracingAccelerationStructure>*` (buffer of handles) indexed dynamically | ✅ | ✅ | The bindless-array case — compiles cleanly on both. |

What form **B** lowers to:

- **SPIR-V:** the handle is a 64-bit value (`uint2`) inside the root struct; the shader does
  `%t = OpConvertUToAccelerationStructureKHR %u64`. The "handle" is literally the acceleration
  structure's **device address**. No descriptor, no binding decoration. Capabilities used:
  `RayQueryKHR`, `PhysicalStorageBufferAddresses`, `Int64` (exts `SPV_KHR_ray_query`,
  `SPV_KHR_physical_storage_buffer`, `SPV_KHR_ray_tracing`). The address→AS conversion is
  available through `SPV_KHR_ray_query`, which we already enable.
- **Metal:** the root struct gains a `metal::raytracing::acceleration_structure<instancing>`
  member, read straight from `buffer(0)` (where the root already lives). No extra kernel
  argument, no arg-table slot.

So **B is pure buffer-device-address** — the AS handle flows through the root struct exactly like
every other resource the RHI already passes by BDA. It needs **no descriptor heap, no
argument-table slot, no registration API**. Form C shows the same mechanism scales to dynamic
arrays of acceleration structures (the non-gimped requirement) with nothing more than a buffer
of handles.

This supersedes rev. 2's "binding 3 descriptor heap" plan, which form A2 proved would be the
heaviest option.

---

## 3. Proposed convention

### 3.1 The model

Acceleration structures are referenced by a **64-bit handle carried in normal buffers/root
structs**, identical to how the RHI already passes data by buffer device address. There is **no
descriptor set, no argument-table slot, and no heap** for acceleration structures.

- **Vulkan:** the handle is the AS **device address** (`vkGetAccelerationStructureDeviceAddressKHR`
  — already stored on `VulkanAccelerationStructure.device_address` and returned by `accel.gpu()`).
- **Metal:** the handle is the AS **`gpuResourceID`** (already stored, returned by `accel.gpu()`).
- The RHI therefore already exposes the correct per-backend handle: **`accel.gpu()`**. Callers
  write it into their root struct (or a handle buffer) like any other 64-bit GPU value.

### 3.2 Shader-authoring rules

1. **Root data is always an entry-point `uniform` parameter, never a module-scope global.**
   ```slang
   void traceMain(uint3 tid : SV_DispatchThreadID, uniform TraceRoot* r) { … }
   ```
   This is what makes it a push-constant BDA pointer (Vulkan) / `buffer(0)` (Metal). A
   module-scope global instead triggers the `$Globals` → set 0 collision.
   → **`instanced_grid` fix is purely a shader edit:** move `uniform GridCfg* cfg` from module
   scope into the entry point's parameter list. No RHI change for that case.

2. **Acceleration structures are `DescriptorHandle<RaytracingAccelerationStructure>` fields**, in
   the root struct (single) or in a buffer (bindless array). Single TLAS:
   ```slang
   struct TraceRoot {
       uint* output;
       DescriptorHandle<RaytracingAccelerationStructure> tlas;   // 64-bit handle = accel.gpu()
   };
   void traceMain(uint3 tid : SV_DispatchThreadID, uniform TraceRoot* r) {
       RaytracingAccelerationStructure tlas = r.tlas;            // implicit handle → AS
       RayQuery<RAY_FLAG_NONE> q;
       q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);
       …
   }
   ```
   Bindless array of acceleration structures (verified — form C):
   ```slang
   struct TraceRoot {
       DescriptorHandle<RaytracingAccelerationStructure>* tlasTable;  // buffer of handles
       uint tlasIndex;
   };
   RaytracingAccelerationStructure tlas = r.tlasTable[r.tlasIndex];
   ```
   No `[[vk::binding]]`, no `NonUniformResourceIndex` (which is Metal-incompatible in compute).

3. **Bindless textures/samplers/storage images** keep using the existing set-0 index path,
   unchanged.

4. **Defensive default in the compile harness:** pass `-fvk-bind-globals 0 1` so any *stray*
   module-scope `uniform` (a rule-1 violation) lands harmlessly on reserved set 1 instead of
   corrupting set 0 — turning a silent aliasing bug into an obvious "set 1 not bound" failure.

### 3.3 RHI API

`bind_acceleration_structure(slot, accel)` is **removed** (no-op on Vulkan, fragile slot
overwrite on Metal). No replacement registration call is needed: callers obtain the handle from
the existing **`accel.gpu()`** and place it in their root/handle buffer. `gpu_struct!` gains a
Slang spelling for the handle field (e.g. a `GpuAddress`/`u64` field annotated
`as "DescriptorHandle<RaytracingAccelerationStructure>"`); the field is 8 bytes on both backends,
matching `GpuAddress`.

### 3.4 Metal parity

Already aligned by construction: §3a confirmed Metal stores the `acceleration_structure` handle
inline in the root struct read from `buffer(0)`. The only Metal-side obligation is **residency** —
the TLAS (and its BLAS dependencies) must be in the residency set for any command that traces
against a handle. (Today `bind_acceleration_structure` implied that via `setResource`; with the
inline-handle model the RHI must add the AS to the residency set when the caller registers/uses
its handle — e.g. a lightweight `use_acceleration_structure(accel)` that only manages residency,
or automatic residency of all live acceleration structures.)

---

## 4. Implementation sketch (follow-up change — not part of this review)

**Vulkan** — ✅ landed & verified validation-clean on RTX 2070 SUPER (test `ray_query_triangle_hit`).
The address→AS path needed more device features than first expected; running with validation on
surfaced each one (the original feature chaining was silently broken — see below):
1. **Features the `DescriptorHandle` lowering requires**, all now enabled when present:
   `VK_KHR_ray_tracing_maintenance1` (`OpConvertUToAccelerationStructureKHR`),
   `VK_KHR_ray_tracing_pipeline` (the lowering emits the `SPV_KHR_ray_tracing` SPIR-V extension,
   which validation maps to this requirement even though we only do inline ray query),
   `shaderInt64` (handle is a 64-bit value), and `descriptorBindingPartiallyBound` (the bindless
   heap declared it but never enabled it).
2. **Pre-existing bug fixed:** the optional feature structs were chained with
   `let _ = features2.push_next(..)`. `PhysicalDeviceFeatures2` is `Copy`, so that linked each
   node into a discarded copy — accel/ray-query/etc. features were never actually enabled (NVIDIA
   tolerated it). Now reassigned (`features2 = features2.push_next(..)`).
3. **Pre-existing bug fixed:** `build_blas` dropped the triangle geometry's `OPAQUE` flag, so
   `RayQuery::Proceed` never auto-committed the hit → every trace missed. Now carried through.
4. **Pre-existing leak fixed:** `VulkanComputePso` had no `Drop`; its pipeline + layout leaked.
5. **No heap, no descriptor set, no pipeline-layout change**, and `accel.gpu()` already returns
   `device_address` — the core bindless-AS claim held exactly as designed.

**Metal**
1. **Residency** for traced acceleration structures (see §3.4).
2. **`accel.gpu()`** already returns the `gpuResourceID`; nothing new to plumb.

**Frontend**
1. Remove `bind_acceleration_structure` from the device/command API.
2. `gpu_struct!` Slang spelling for `DescriptorHandle<RaytracingAccelerationStructure>` fields.

### Open questions — resolved

- **A. Confirm on-device (RTX 2070).** ✅ `raytracing.rs` traces correctly, validation-clean.
- **B. Residency strategy on Metal.** ✅ Explicit: `add_accel_to_residency()` called at BLAS/TLAS
  build time in `metal/command.rs`. Mirrors how other resources opt into residency.
- **C. `gpu_struct!` ergonomics.** ✅ `as "DescriptorHandle<RaytracingAccelerationStructure>"`
  string override on a `GpuAddress` field. No dedicated type alias needed — the override is
  self-documenting at the field site and the comment convention in roots.rs is sufficient.

---

## 5. What changes, concretely

> **All items complete.** Verified on Vulkan (RTX 2070 SUPER, validation-clean) and Metal (M4, slangc 2026.10.2).

| Item | Change | Kind |
|---|---|---|
| `instanced_grid` shader | ✅ moved `uniform GridCfg* cfg` module scope → entry-point param | shader edit |
| `raytracing.rs` (`RQ_BODY` + `Root`) | ✅ `tlas` as `DescriptorHandle<…>` in `Root`; `tlas.gpu()` on CPU side | shader + test edit |
| spectral RT shaders (`roots.rs`, `integrator.rs`) | ✅ `tlas: GpuAddress as "DescriptorHandle<…>"` in `TraceRoot`; `accel.tlas.gpu()` on CPU side | shader edit |
| Vulkan `device.rs` | ✅ `VK_KHR_ray_tracing_maintenance1`, `shaderInt64`, `descriptorBindingPartiallyBound` enabled | RHI |
| `command.rs` | ✅ `bind_acceleration_structure` removed | RHI |
| Metal `command.rs`/`device.rs` | ✅ `add_accel_to_residency()` at BLAS/TLAS build time | RHI (Metal) |
| compile harness | ✅ `-fvk-bind-globals 0 1` in `src/compiler/mod.rs` (canonical compile path) | harness |
| frontend `device.rs` + `macros.rs` | ✅ `bind_acceleration_structure` removed; `as "…"` override used at field site | RHI API |
| `mesh.rs`/`textures.rs` test leaks | ✅ `destroy_texture` + `free` added for all transient render targets | test cleanup |

---

## 6. Sources

- Slang — default unannotated binding (set 0), `$Globals` cbuffer, `-fvk-bind-globals`,
  `-fvk-{b|s|t|u}-shift`, `[[vk::binding]]`:
  <https://shader-slang.org/slang/user-guide/> (SPIR-V target-specific functionality).
- DescriptorHandle / bindless heap lowering & `-bindless-space-index`, single-set limitation:
  <https://github.com/shader-slang/slang/discussions/8610>,
  <https://github.com/shader-slang/slang/issues/8063>.
- HLSL-for-Vulkan resource binding background:
  <https://www.lei.chat/posts/hlsl-for-vulkan-resources/>.
