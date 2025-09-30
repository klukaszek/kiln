# Kiln Metal Facade: Feature Checklist

This checklist tracks the safe, zero-copy Metal coverage exposed via `kiln::metal`.
The app runtime now implicitly begins/ends frames and provides a `RenderEncoder`
to `KilnApp::draw`, making multi‑PSO rendering natural while maintaining zero‑copy usage.
Goal: provide an idiomatic, safe Rust API without leaking unsafe `objc2_metal` items.

## Safety & Zero-Copy Principles
- [x] Zero-copy buffer APIs use `IntoBytes + FromBytes + Copy + Immutable`
- [x] Avoid exposing unsafe constructors/protocols in public API
- [ ] Document lifetime/ownership guarantees for wrappers

## Core Platform
- [x] Device wrapper (`metal::Device::from_surface`) from `RenderSurface`
- [x] Command queue wrapper (`metal::Queue`): wait, commit, signal
- [x] Command allocator wrapper (`metal::CommandAllocator`): reset
- [x] Command buffer wrapper (`metal::CommandBuffer`): begin/end, render encoder
- [ ] GPU selection/options (default/system device abstraction)
- [ ] Feature sets / family capabilities query
- [ ] Capture manager integration (debugging tools)

## Shaders & Pipeline
- [x] MSL compile via `MTL4Compiler` into `metal::Library`
- [x] Render pipeline builder (vertex/fragment/color format)
  - [x] Multi‑PSO per frame via encoder API (app‑managed frames)
- [ ] Compute pipeline creation
- [ ] Function specialization constants
- [ ] Linked functions / dynamic libraries
- [ ] Binary archives (pipeline caching)

## Render Pass & Encoder (Graphics)
- [x] Render pass descriptor wrapper: acquire current, set clear/load
- [x] Render encoder wrapper: set pipeline, set argument table at stages
  - [x] Encoder passed into `KilnApp::draw` by app runtime
- [x] Non-indexed draw (`draw_primitives`)
- [ ] Indexed draws (u16/u32)
- [x] Instanced draws
- [ ] Indirect draws
- [ ] Graphics state
  - [ ] Viewport, scissor, triangle fill, cull, front face
  - [ ] Blend / color write masks
  - [ ] Depth/stencil state, depth bias, clip mode
  - [ ] Vertex buffer bindings / attributes layout
  - [ ] Fragment buffers/textures/samplers

## Compute Pass
- [ ] Compute pipeline wrapper
- [ ] Compute encoder wrapper (set pipeline, bind resources)
- [ ] Dispatch threads / threadgroups
- [ ] Indirect dispatch

## Blit Pass
- [ ] Blit encoder wrapper
- [ ] Buffer/texture copy/fill
- [ ] Resource state/synchronization blits

## Resources & Memory
- [x] Buffer wrappers: vertex, uniform (zero-copy create/from slice/with len)
- [x] Argument table wrapper: bind buffers by GPU address
- [x] Index buffer helpers (u16/u32) + binding prep
- [ ] Texture wrapper + descriptor/creation
- [ ] Texture views (2D/3D/array/cube), usage flags
- [ ] Sampler state wrapper + descriptor
- [ ] Heaps and placement allocation
- [ ] Residency sets
- [ ] Resource fences / hazard tracking controls

## Synchronization & Presentation
- [x] Drawable wrapper: acquire/present (via app‑managed `RenderFrame` RAII)
- [x] Queue signals/waits for drawable
- [ ] Events/shared events
- [ ] Frame pacing/present at time

## Ray Tracing / Acceleration Structures
- [ ] Acceleration structure types and descriptors
- [ ] Geometry/instance descriptors
- [ ] Acceleration structure builder/encoder
- [ ] Intersection/visible function tables
- [ ] Ray tracing pipeline state

## Indirect Command Buffers (ICB)
- [ ] ICB descriptors + creation
- [ ] Record indirect draw/compute commands safely

## Counters/Queries/Statistics
- [ ] Counter sampling point support
- [ ] Performance statistics (if exposed)

## Enums & Structs (Safe Re-exports)
- [x] MTLClearColor, MTLLoadAction, MTLStoreAction
- [x] MTLPixelFormat, MTLPrimitiveType, MTLRenderStages, MTLResourceOptions
- [ ] Storage/CPU cache/ hazard/ usage enums as needed
- [ ] All texture/sampler descriptor structs
- [ ] All render/compute/blit descriptor structs

## Error Handling & Diagnostics
- [ ] Map Metal NSError to kiln error types with context
- [ ] Log/trace hooks (optional)

## Feature Flags & Platform
- [ ] cfg-gate APIs that require GPU family/OS versions
- [ ] Ensure macOS/iOS surface support split is clear

## Testing & Examples
- [x] Triangle example via `kiln::metal` facade
- [x] Instanced triangle example
- [ ] Examples for compute + blit
- [ ] Example for argument tables with multiple resources
- [ ] Example for textures/samplers
- [ ] Example for acceleration structures (when implemented)

---

Status key
- [x] Implemented
- [ ] Planned
- [~] Partial
