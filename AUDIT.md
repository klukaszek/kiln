# RHI Aaltonen Parity Audit

Complete comparison of `crates/rhi` against the **Prototype API** in `NoGraphicsApi.md`
(the 150-line C++ spec, lines 963–1226 including the Appendix bump allocator).

Every item is one of:
- ✅ **Match** — functionally identical, possibly with Rust naming conventions
- ⚠️ **Divergence** — exists but differs; note explains the delta
- ❌ **Missing** — not present; action required for full parity
- 🔵 **Extension** — we have it, Aaltonen doesn't; intentional addition, not a gap

---

## 1. Enums

### `MEMORY { MEMORY_DEFAULT, MEMORY_GPU, MEMORY_READBACK }`
✅ `MemoryType { Default, GpuOnly, Readback }` in `memory.rs`

### `CULL { CULL_CCW, CULL_CW, CULL_ALL, CULL_NONE }`
✅ `Cull { None, Cw, Ccw, All }` in `types.rs`

### `DEPTH_FLAGS { DEPTH_READ = 0x1, DEPTH_WRITE = 0x2 }`
✅ `DepthFlags { READ = 0x1, WRITE = 0x2 }` in `types.rs`

### `OP { OP_NEVER, OP_LESS, OP_EQUAL, OP_LESS_EQUAL, OP_GREATER, OP_NOT_EQUAL, OP_GREATER_EQUAL, OP_ALWAYS }`
✅ `CompareOp { Never, Less, Equal, LessOrEqual, Greater, NotEqual, GreaterOrEqual, Always }` in `types.rs`
> Aaltonen reuses `OP` for both compare and stencil ops. We split into `CompareOp` + `StencilOp` — cleaner, not a deficit.

### `BLEND { BLEND_ADD, BLEND_SUBTRACT, BLEND_REV_SUBTRACT, BLEND_MIN, BLEND_MAX }`
✅ `BlendOp { Add, Subtract, ReverseSubtract, Min, Max }` in `types.rs`

### `FACTOR { FACTOR_ZERO, FACTOR_ONE, FACTOR_SRC_COLOR, FACTOR_DST_COLOR, FACTOR_SRC_ALPHA, ... }`
✅ `BlendFactor { Zero, One, SrcColor, OneMinusSrcColor, DstColor, OneMinusDstColor, SrcAlpha, OneMinusSrcAlpha, DstAlpha, OneMinusDstAlpha }` in `types.rs`

### `TOPOLOGY { TOPOLOGY_TRIANGLE_LIST, TOPOLOGY_TRIANGLE_STRIP, TOPOLOGY_TRIANGLE_FAN }`
✅ `Topology { TriangleList, TriangleStrip, TriangleFan }` in `types.rs`.
Metal rejects `TriangleFan` at PSO creation because Metal has no native triangle-fan topology; callers must rewrite to indexed triangles before submission.

### `TEXTURE { TEXTURE_1D, TEXTURE_2D, TEXTURE_3D, TEXTURE_CUBE, TEXTURE_2D_ARRAY, TEXTURE_CUBE_ARRAY }`
✅ `TextureDimension { D1, D2, D2Array, D3, Cube, CubeArray }` in `types.rs`.
Both backends map these to native Vulkan image view types / Metal texture types.

### `FORMAT { FORMAT_NONE, FORMAT_RGBA8_UNORM, FORMAT_D32_FLOAT, FORMAT_RG11B10_FLOAT, FORMAT_RGB10_A2_UNORM, ... }`
✅ `Format` enum in `types.rs` covers all listed formats and more (🔵 extended).
⚠️ Aaltonen uses `FORMAT_NONE` as an explicit zero sentinel. We use `Option<Format>` in `GraphicsPsoDesc` fields and `TextureDesc` doesn't have a None variant. This is idiomatic Rust and not a problem.

### `USAGE_FLAGS { USAGE_SAMPLED, USAGE_STORAGE, USAGE_COLOR_ATTACHMENT, USAGE_DEPTH_STENCIL_ATTACHMENT, ... }`
✅ `TextureUsage::{SAMPLED, STORAGE, COLOR_ATTACHMENT, DEPTH_STENCIL_ATTACHMENT, ...}` in `texture.rs`.

### `STAGE { STAGE_TRANSFER, STAGE_COMPUTE, STAGE_RASTER_COLOR_OUT, STAGE_PIXEL_SHADER, STAGE_VERTEX_SHADER, ... }`
✅ `StageFlags` in `barrier.rs` covers all listed stages.
🔵 We add `RASTER_DEPTH_OUT`, `ALL_GRAPHICS`, `ALL_COMMANDS`.

### `HAZARD_FLAGS { HAZARD_DRAW_ARGUMENTS = 0x1, HAZARD_DESCRIPTORS = 0x2, HAZARD_DEPTH_STENCIL = 0x4 }`
✅ `HazardFlags { DRAW_ARGUMENTS = 0x1, DESCRIPTORS = 0x2, DEPTH_STENCIL = 0x4 }` in `barrier.rs` — exact match.

### `SIGNAL { SIGNAL_ATOMIC_SET, SIGNAL_ATOMIC_MAX, SIGNAL_ATOMIC_OR, ... }`
✅ `SignalOp { AtomicSet, AtomicMax, AtomicOr }` in `command.rs`

---

## 2. Structs

### `Stencil { OP test, OP failOp, OP passOp, OP depthFailOp, uint8 reference }`
✅ `StencilDesc { test: CompareOp, fail_op: StencilOp, pass_op: StencilOp, depth_fail_op: StencilOp, reference: u8 }` in `pipeline.rs` — exact match.

### `GpuDepthStencilDesc`
✅ `DepthStencilState` in `pipeline.rs` — all fields match including defaults:
- `depthMode = 0` → `depth_mode: DepthFlags::empty()` ✅
- `depthTest = OP_ALWAYS` → `depth_test: CompareOp::Always` ✅
- `depthBias/SlopeFactor/Clamp = 0` → all `0.0` ✅
- `stencilReadMask = 0xff` → `stencil_read_mask: 0xff` ✅
- `stencilWriteMask = 0xff` → `stencil_write_mask: 0xff` ✅

### `GpuBlendDesc { BLEND colorOp, FACTOR srcColorFactor, FACTOR dstColorFactor, BLEND alphaOp, FACTOR srcAlphaFactor, FACTOR dstAlphaFactor, uint8 colorWriteMask }`
⚠️ Our `BlendAttachment` adds a `blend_enable: bool` field not present in Aaltonen's struct.
Aaltonen implies blending is always active when a `GpuBlendDesc` is bound (the object itself signals intent). Our `blend_enable = false` default replicates "no blending" without a separate concept.
**Assessment:** Not a gap — `blend_enable` is needed for the Metal MTL4BlendState enum and the Vulkan `blendEnable` field. Keeping it.

### `ColorTarget { FORMAT format, uint8 writeMask }`
✅ `ColorTarget { format: Format, write_mask: ColorWriteMask }` in `pipeline.rs` — exact match.

### `GpuRasterDesc { TOPOLOGY, CULL, alphaToCoverage, supportDualSourceBlending, uint8 sampleCount, FORMAT depthFormat, FORMAT stencilFormat, Span<ColorTarget>, GpuBlendDesc* blendstate }`
⚠️ **Three divergences:**
1. ✅ `blendstate: Option<BlendState>` is present. When `Some`, the backend pre-bakes that blend variant; when `None`, `cmd.set_blend_state(...)` selects a flyweight variant.
2. ✅ `SampleCount` covers `S1/S2/S4/S8/S16`.
3. **Extra fields** (`vertex_shader: usize`, `pixel_shader: usize`, `root_constant_size`, `label`) — 🔵 our implementation detail; Aaltonen passes IR blobs as separate function arguments, not in the desc.

### `GpuTextureDesc { TEXTURE type, uint32x3 dimensions, uint32 mipCount, uint32 layerCount, uint32 sampleCount, FORMAT format, USAGE_FLAGS usage }`
⚠️ Dimensions are split as `width/height/depth` vs `uint32x3` — functionally identical, idiomatic Rust.

### `GpuViewDesc { FORMAT format, uint8 baseMip, uint8 mipCount=ALL_MIPS, uint16 baseLayer, uint16 layerCount=ALL_LAYERS }`
✅ `GpuViewDesc`, `ALL_MIPS`, `ALL_LAYERS`, `device.texture_view_descriptor(...)`, and `device.rw_texture_view_descriptor(...)` are implemented.

### `GpuTextureSizeAlign { size_t size; size_t align; }` + placement-alloc texture creation
✅ `device.texture_size_align(desc)` + `device.create_texture(desc, texture_gpu)`.
Aaltonen: `gpuTextureSizeAlign(desc)` returns the size/alignment, then `gpuCreateTexture(desc, ptrGpu)` places the texture into a caller-managed allocation.
Ours: caller allocates with `device.malloc_aligned(size, align, MemoryType::GpuOnly)`, passes `allocation.gpu_address()` to `create_texture`, and `Texture::gpu_address()` stores that placement address.
Vulkan binds the image to the allocation behind `texture_gpu`; Metal creates textures from placement heaps.

### `GpuTextureDescriptor { uint64[4] data }`
⚠️ **Different texture handle concept.**
Aaltonen exposes raw 4×u64 GPU-side descriptors written directly into GPU memory by the user.
We use `TextureId(u32)` — a 32-bit index into the backend-managed bindless heap. Shaders index the heap with this value.
**Assessment:** Functionally equivalent for current usage. Our model is simpler and sufficient for Metal 4 argument tables and Vulkan descriptor buffer. No action needed.

### `GPUBumpAllocator` (Appendix)
✅ `BumpAllocator` in `memory.rs` — matches: `alloc(size, align)`, `reset()`, dual CPU/GPU pointers.

---

## 3. Functions / Commands

### Memory
| Aaltonen | Ours | Status |
|---|---|---|
| `gpuMalloc(size, memory)` | `device.malloc(size, memory)` | ✅ |
| `gpuMalloc(size, align, memory)` | `device.malloc_aligned(size, align, memory)` | ✅ |
| `gpuFree(ptr)` | `device.free(alloc)` | ✅ |
| `gpuHostToDevicePointer(ptr)` | `device.host_to_device_pointer(ptr)` | ✅ |
| `gpuTextureSizeAlign(desc)` | `device.texture_size_align(desc)` | ✅ |
| `gpuCreateTexture(desc, ptrGpu)` | `device.create_texture(desc, texture_gpu)` | ✅ |
| `gpuTextureViewDescriptor(tex, view)` | `device.texture_view_descriptor(tex, view)` | ✅ |
| `gpuRWTextureViewDescriptor(tex, view)` | `device.rw_texture_view_descriptor(tex, view)` | ✅ |

### Pipelines & State
| Aaltonen | Ours | Status |
|---|---|---|
| `gpuCreateComputePipeline(ir)` | `device.create_compute_pso(desc)` | ✅ |
| `gpuCreateGraphicsPipeline(vIR, pIR, desc)` | `device.create_graphics_pso(desc)` | ✅ |
| `gpuCreateGraphicsMeshletPipeline(...)` | `device.create_meshlet_pso(desc)` | ✅ |
| `gpuFreePipeline(pipeline)` | `Drop` impl | ✅ |
| `gpuCreateDepthStencilState(desc)` | inline struct, no opaque handle | ⚠️ see below |
| `gpuCreateBlendState(desc)` | inline struct, no opaque handle | ⚠️ see below |
| `gpuFreeDepthStencilState` / `gpuFreeBlendState` | `Drop` | ✅ |

> **State object handles:** Aaltonen has opaque `GpuDepthStencilState` / `GpuBlendState` handles created via factory functions. We use plain structs passed by reference to `cmd.set_depth_stencil_state` / `cmd.set_blend_state`. This is idiomatic Rust and has zero overhead — the opaque handles are just a C API convention. No action needed.

### Queue & Submission
| Aaltonen | Ours | Status |
|---|---|---|
| `gpuCreateQueue(...)` | `Device::new()` — queue is integrated | ✅ |
| `gpuStartCommandRecording(queue)` | `device.create_command_buffer()` | ✅ |
| `gpuSubmit(queue, cmds)` | `queue.submit(cmd)` | ✅ |
| `gpuCreateSemaphore(initValue)` | `device.create_timeline_semaphore(init)` | ✅ |
| `gpuWaitSemaphore(sema, value)` | `sema.wait(value, timeout_ns)` | ✅ |
| `gpuDestroySemaphore(sema)` | `Drop` | ✅ |

### Commands
| Aaltonen | Ours | Status |
|---|---|---|
| `gpuMemCpy(cb, dst, src)` | `cmd.memcpy(dst, src, size)` | ✅ |
| `gpuCopyToTexture(cb, dst, src, tex)` | `cmd.copy_to_texture(texture_gpu, src, tex)` | ✅ |
| `gpuCopyFromTexture(cb, dst, src, tex)` | `cmd.copy_from_texture(dst, texture_gpu, tex)` | ✅ |
| `gpuSetActiveTextureHeapPtr(cb, ptr)` | `cmd.set_active_texture_heap_ptr(ptr)` | ✅ |
| `gpuBarrier(cb, before, after, hazards=0)` | `cmd.barrier(src, dst)` / `cmd.barrier_with_hazard(...)` | ✅ |
| `gpuSignalAfter(cb, before, ptr, val, sig)` | `cmd.signal_after(desc)` | ✅ |
| `gpuWaitBefore(cb, after, ptr, val, op, hazards, mask)` | `cmd.wait_before(desc)` | ✅ |
| `gpuSetPipeline(cb, pipeline)` | `cmd.set_graphics_pipeline(pso)` / `cmd.set_compute_pipeline(pso)` | ✅ |
| `gpuSetDepthStencilState(cb, state)` | `cmd.set_depth_stencil_state(state)` | ✅ |
| `gpuSetBlendState(cb, state)` | `cmd.set_blend_state(state)` | ✅ |
| `gpuDispatch(cb, dataGpu, gridDims)` | `cmd.dispatch(root, x, y, z)` | ✅ |
| `gpuDispatchIndirect(cb, dataGpu, dimGpu)` | `cmd.dispatch_indirect(root, args)` | ✅ |
| `gpuBeginRenderPass(cb, desc)` | `cmd.begin_render_pass(desc)` | ✅ |
| `gpuEndRenderPass(cb)` | `cmd.end_render_pass()` | ✅ |
| `gpuDrawIndexedInstanced(cb, vData, pData, idx, count, instances)` | `cmd.draw_indexed(vR, pR, idx, cnt, inst)` | ✅ |
| `gpuDrawIndexedInstancedIndirect(cb, vData, pData, idx, argsGpu)` | `cmd.draw_indexed_indirect(vR, pR, idx, args)` | ✅ |
| `gpuDrawIndexedInstancedIndirectMulti(cb, vData, vStride, pData, pStride, args, drawCount)` | `cmd.draw_indexed_indirect_multi(...)` | ⚠️ API shape now preserves per-draw index-buffer GPU pointers in `DrawIndexedIndirectMultiArgs`; Vulkan needs `VK_EXT_device_generated_commands`, Metal4 needs GPU ICB generation |
| `gpuDrawMeshlets(cb, meshData, pixData, dim)` | `cmd.draw_meshlets(mesh_root, pixel_root, x, y, z)` | ✅ |
| `gpuDrawMeshletsIndirect(cb, meshData, pixData, dimGpu)` | `cmd.draw_meshlets_indirect(mesh_root, pixel_root, args)` | ✅ |

---

## 4. Prioritised Action List

### P0 — Correctness
| # | Item | Files | Status |
|---|------|-------|--------|
| 0 | Finish indexed MDI backends. `NoGraphicsApi.md` requires `gpuDrawIndexedInstancedIndirectMulti(cb, vData, vStride, pData, pStride, args, drawCount)` with root data arrays, per-draw index-buffer GPU pointers, indirect draw args, and a GPU draw count. Vulkan should use `VK_EXT_device_generated_commands` with an index-buffer token plus draw-indexed-count token; core `vkCmdDrawIndexedIndirectCount` alone is only valid for a pre-bound index atlas and violates the pointer-first contract. Metal4 must generate an `MTLIndirectCommandBuffer` and `MTLIndirectCommandBufferExecutionRange` on the GPU, then execute it through the Metal4 render encoder. | `command.rs`, `backend/vulkan/command.rs`, `backend/metal/command.rs`, `backend/metal/device.rs` | ⚠️ Open |

### P1 — Direct Prototype API gaps (should fix before shipping any feature that touches them)

| # | Item | Files | Status |
|---|------|-------|--------|
| 1 | Add `Topology::TriangleFan` | `types.rs`, `vulkan/pipeline.rs`, `metal/device.rs` (panic + note) | ✅ Done 2026-05-25 |
| 2 | Add `TextureDimension::D2Array` and `CubeArray` | `types.rs`, `vulkan/device.rs`, `metal/device.rs` | ✅ Done 2026-05-25 |
| 3 | Add `GpuViewDesc` + `device.texture_view_descriptor` / `device.rw_texture_view_descriptor` | `texture.rs`, `device.rs`, `vulkan/device.rs`, `vulkan/texture.rs`, `metal/device.rs` | ✅ Done 2026-05-25 |
| 4 | Use `TextureUsage::DEPTH_STENCIL_ATTACHMENT` | `texture.rs`, `vulkan/device.rs`, `metal/device.rs` | ✅ Done 2026-05-25 |

### P2 — API shape differences (improve parity, not urgent)

| # | Item | Files | Status |
|---|------|-------|--------|
| 5 | Add `blendstate: Option<BlendState>` to `GraphicsPsoDesc`; bake when `Some` | `pipeline.rs`, `vulkan/device.rs`, `metal/device.rs` | ✅ Done 2026-05-25 |
| 6 | Add `S16` to `SampleCount` enum | `types.rs`, `vulkan/pipeline.rs`, `vulkan/device.rs`, `metal/device.rs` | ✅ Done 2026-05-25 |

### P3 — Out of scope / deferred

| # | Item | Reason |
|---|------|--------|
| 7 | ~~Mesh shader support~~ | ✅ Done 2026-05-25 — `create_meshlet_pso`, `draw_meshlets`, `draw_meshlets_indirect` |
| 8 | ~~Placement-alloc textures~~ (`gpuTextureSizeAlign` + `ptrGpu` in `gpuCreateTexture`) | ✅ Done 2026-05-26 |
| 9 | Raw `GpuTextureDescriptor` (4×u64) | `TextureId(u32)` is equivalent for our bindless model |

---

## 5. Extensions We Add That Aaltonen Doesn't Have

> Note: Aaltonen defers ray-tracing and the full shader framework to a followup post. We implement it as an extension using the same "everything is GPU memory" philosophy — SBT is a user-managed GPU buffer, BLAS/TLAS handles are `u64` GPU addresses.

These are intentional additions, not gaps:

| Item | Location | Rationale |
|------|----------|-----------|
| `StageFlags::RASTER_DEPTH_OUT, ALL_GRAPHICS, ALL_COMMANDS` | `barrier.rs` | Needed for depth prepass barriers and flush-all |
| `TextureUsage::TRANSFER_SRC / TRANSFER_DST` | `texture.rs` | Required for staging uploads |
| `BlendAttachment::blend_enable` | `pipeline.rs` | Required for Metal MTL4BlendState enum |
| `ClipSpaceY` enum | `types.rs` | Tracks Y-flip convention per backend |
| `SamplerId(u32)` + sampler heap | `types.rs`, `sampler.rs` | Aaltonen samples inline in shaders; we bindless-index samplers |
| `draw` (non-indexed) | `command.rs` | Non-indexed draw path |
| Split-barrier `signal_after` / `wait_before` (non-value) | `command.rs` | Simpler barrier form without value sync |
| `transition_to_present` / `end` | `command.rs` | Swapchain lifecycle (Vulkan needs explicit layout transition) |
| `vulkan_handles` / `vulkan_command_buffer` escape hatches | `device.rs`, `command.rs` | ImGui integration |
| `Backend`, `BindlessMode`, `DeviceDesc` | `device.rs` | Runtime backend selection |
| `SubmitDesc` with timeline wait/signal | `queue.rs` | Fine-grained queue sync |
| `BumpAllocator::alloc_typed<T>` / `upload_slice` | `memory.rs` | Ergonomic typed allocation |
| `create_command_buffer_for_swapchain` | `device.rs` | Swapchain-aware command buffer |
| `create_meshlet_pso` / `draw_meshlets` / `draw_meshlets_indirect` | `device.rs`, `command.rs` | Mesh shader pipeline + draws; root pointers in the draw call per spec (Vulkan `VK_EXT_mesh_shader`, Metal 4 mesh render pipeline path) |
| `create_ray_tracing_pso` / `trace_rays` | `device.rs`, `command.rs` | RT pipeline (Vulkan `VK_KHR_ray_tracing_pipeline`, Metal 4 compute dispatch with function-table resource IDs in the GPU root/SBT block) |
| `create_blas` / `create_tlas` / `build_blas` / `build_tlas` | `device.rs`, `command.rs` | Acceleration structure lifecycle (Vulkan `VK_KHR_acceleration_structure`, Metal 4 `MTL4AccelerationStructureDescriptor`; Metal BLAS supports triangle and AABB geometry) |

---

## 6. Files to Edit per Action Item

### Action 1 — `Topology::TriangleFan`
- `crates/rhi/src/types.rs` — add variant
- `crates/rhi/src/backend/vulkan/pipeline.rs` — `create_pipeline` topology match arm: `Topology::TriangleFan => vk::PrimitiveTopology::TRIANGLE_FAN`
- `crates/rhi/src/backend/metal/pipeline.rs` — no native Metal equivalent; `TriangleFan => panic!("TriangleFan not supported on Metal; rewrite indices to TriangleList before submission")` or silent remap to `Triangle` with a warning. Document in a comment.
- `crates/rhi/src/backend/metal/command.rs` — same topology dispatch if stored on encoder

### Action 2 — `TextureDimension::D2Array` / `CubeArray`
- `crates/rhi/src/types.rs` — add two variants
- `crates/rhi/src/backend/vulkan/device.rs` — image type and view type match arms:
  - `D2Array` → `vk::ImageType::TYPE_2D` + `vk::ImageViewType::TYPE_2D_ARRAY`
  - `CubeArray` → `vk::ImageType::TYPE_2D` + `vk::ImageViewType::CUBE_ARRAY`
- `crates/rhi/src/backend/metal/device.rs` — MTLTexture creation:
  - `D2Array` → `MTLTextureType2DArray`
  - `CubeArray` → `MTLTextureTypeCubeArray`

### Action 3 — `GpuViewDesc` + view descriptor API (done)
- `crates/rhi/src/texture.rs` — add:
  ```rust
  pub const ALL_MIPS: u8 = 0xFF;
  pub const ALL_LAYERS: u16 = 0xFFFF;
  pub struct GpuViewDesc {
      pub format: Option<Format>,   // None = same as texture
      pub base_mip: u8,
      pub mip_count: u8,            // ALL_MIPS = all remaining
      pub base_layer: u16,
      pub layer_count: u16,         // ALL_LAYERS = all remaining
  }
  ```
- `crates/rhi/src/device.rs` — add:
  ```rust
  pub fn texture_view_descriptor(&self, texture: &Texture, view: &GpuViewDesc) -> RhiResult<TextureId>
  pub fn rw_texture_view_descriptor(&self, texture: &Texture, view: &GpuViewDesc) -> RhiResult<TextureId>
  ```
- `crates/rhi/src/backend/vulkan/device.rs` — create `VkImageView` with the given sub-range, write into the bindless heap, return the slot index
- `crates/rhi/src/backend/metal/device.rs` — create a `MTLTextureView` with the given sub-range, write into the argument table, return the slot index

### Action 4 — Rename `DEPTH_STENCIL_ATTACHMENT` flag (done)
- `crates/rhi/src/texture.rs` — rename the bitflag constant
- `crates/rhi/src/backend/vulkan/device.rs` — update the flag check
- `crates/rhi/src/backend/metal/device.rs` — update the flag check
- `src/render/scene_renderer.rs` — update call site (if used)
- `src/scene/loaders/pbrt.rs` — grep and update all uses

### Action 5 — `GraphicsPsoDesc::blendstate: Option<BlendState>`
- `crates/rhi/src/pipeline.rs` — add `pub blendstate: Option<BlendState>` to `GraphicsPsoDesc`; update `Default` to `None`
- `crates/rhi/src/backend/vulkan/device.rs` — when `desc.blendstate.is_some()`, call `pipeline_for_blend` immediately at PSO creation and store as the "default" pipeline; otherwise use `BlendState::default()` as now
- `crates/rhi/src/backend/metal/device.rs` — same: if embedded blend, pre-bake that variant

### Action 6 — `SampleCount::S16`
- `crates/rhi/src/types.rs` — add `S16` variant
- `crates/rhi/src/backend/vulkan/pipeline.rs` — add match arm: `SampleCount::S16 => vk::SampleCountFlags::TYPE_16`
- `crates/rhi/src/backend/metal/device.rs` — add match arm: `SampleCount::S16 => 16`

---

*Last updated: 2026-05-25. All prototype API surface implemented. RT + mesh shader added as 🔵 extensions. Re-run this audit whenever `NoGraphicsApi.md` is updated or a new feature is added to the RHI.*
