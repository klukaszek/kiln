# Kiln RHI

A render hardware interface for Vulkan and Metal, built around a single idea: the GPU is a
machine that reads pointers. Instead of modelling the API as descriptor sets, bind groups, and
per-resource state machines, Kiln exposes raw GPU virtual addresses and a small set of verbs for
turning them into work. The design follows Sebastian Aaltonen's "No Graphics API" model, adapted
to two modern backends with one portable surface.

The crate targets Vulkan 1.3+ (with buffer device address and descriptor buffers) and Metal 4 (with
argument tables). The same application code runs on both. Shaders are authored once in Slang and
compiled per backend.

![Path-traced Cornell box](public/cornell_box.png)

*Cornell box path-traced headless at 1024x1024, 1024 spp, through the ray-tracing path. Reproduce
with `cargo run --release --example cornell_box -- --headless 1024x1024`.*

## Philosophy

Modern GPU APIs spend a large fraction of their surface area on resource binding: descriptor set
layouts, pipeline layouts, root signatures, bind groups, and the state tracking that keeps them
coherent. Most of that machinery exists to hide the fact that the hardware already speaks a much
simpler language. A shader wants an address; the driver wants to know when a write must be visible
to a later read. Everything else is ceremony.

Kiln removes the ceremony and exposes the two things that actually matter:

1. **Memory is addresses.** Every allocation hands back a CPU pointer and a GPU virtual address.
   You write data through the CPU pointer and you hand the GPU address to a shader. There is no
   binding step in between.
2. **Synchronization is stages, not states.** A barrier names a producer stage and a consumer
   stage. There is no per-resource layout tracking, no "transition this texture from X to Y." If a
   compute pass wrote a buffer that a vertex shader will read, you say so in one line.

What you give up is the API holding your hand. What you get is a model where the cost of a draw is
visible, the data the GPU reads is explicit, and the abstraction does not fight you when you want to
drive the GPU from the GPU.

## Core concepts

### Dual-pointer memory

Every allocation is a `{ cpu, gpu }` pair. The CPU pointer is a mapped host pointer (null for
device-only memory); the GPU pointer is a virtual address the shader can dereference.

```rust
let alloc = device.malloc(size, MemoryType::Default)?;
alloc.upload(&my_uniforms)?;      // write through the CPU pointer
let addr = alloc.gpu();           // hand this address to a shader
```

Three residency classes cover the common cases:

| `MemoryType` | Residency | Use |
|---|---|---|
| `Default`  | CPU-mapped, write-combined | Uniforms, staging, draw args, descriptors |
| `GpuOnly`  | Device-local, not mapped | Textures, large persistent buffers |
| `Readback` | GPU-writable, CPU-cached read | Screenshots, feedback, GPGPU output |

For per-frame transient data, the preferred path is a `BumpAllocator` over one large buffer. Each
sub-allocation returns a dual-pointer `TransientAllocation`; the whole arena is reset once per
frame rather than freed piecewise.

### Root data: one pointer per draw

There are no descriptor sets and no bind groups. A draw or dispatch carries a single root pointer
per shader stage. That pointer is the address of a struct you laid out yourself; inside the shader,
the struct is dereferenced and its fields (themselves often pointers to vertex data, instance data,
material tables) are followed from there.

```rust
cmd.set_graphics_pipeline(&pso);
cmd.draw(vertex_root, pixel_root, vertex_count, 1, 0, 0);
```

`vertex_root` and `pixel_root` are `Option<GpuAddress>`; passing `None` binds a never-dereferenced
null. Compute is the same shape with one root:

```rust
cmd.dispatch(root, groups_x, groups_y, groups_z);
```

This is why indirect and multi-draw fall out naturally. For multi-draw, each draw's root is
`base + draw_id * stride`, so a single recorded command can fan out across thousands of objects
whose parameters live entirely in GPU memory.

### Bindless texture heap

Textures are not bound to slots. A texture view is registered once into a global heap and returns
a `TextureId(u32)`. Shaders index the heap by that integer. The active heap pointer is set once per
command buffer:

```rust
let tex_id = device.texture_view_descriptor(&texture, &view)?;   // -> TextureId
cmd.set_active_texture_heap_ptr(heap_addr);
// shaders read the heap by tex_id.0
```

Sampled views and storage (read-write) views go through `texture_view_descriptor` and
`rw_texture_view_descriptor` respectively. On Vulkan this is backed by the descriptor buffer
extension; on Metal by `MTL4ArgumentTable`. The application sees one model.

### Stage-only barriers

A barrier names a source stage and a destination stage. That is the whole API for the common case:

```rust
cmd.barrier(StageFlags::COMPUTE, StageFlags::VERTEX_SHADER);
```

There are six stages (vertex, pixel, compute, color-out, depth-out, transfer) plus convenience
masks. A small set of `HazardFlags` covers the cases where a stage barrier alone is not enough and
a specific cache needs invalidating: GPU-written indirect arguments, a freshly written descriptor
heap, or depth written by compute.

```rust
cmd.barrier_with_hazard(
    StageFlags::COMPUTE,
    StageFlags::ALL_GRAPHICS,
    HazardFlags::DRAW_ARGUMENTS,
);
```

There is no per-resource state tracking and no layout enum to keep coherent across passes.

### Minimal PSO

A graphics pipeline bakes only what genuinely changes the compiled shader: topology, attachment
formats, sample count, cull mode, and static color write masks. Depth-stencil and blend are
separate flyweight states set on the command buffer, so they do not multiply pipeline permutations.

```rust
let pso = device.create_graphics_pso(
    &GraphicsPsoDesc {
        topology: Topology::TriangleList,
        color_targets: vec![ColorTarget::new(color_format)],
        depth_format: None,
        sample_count: SampleCount::S1,
        root_constant_size: 16,
        cull: Cull::None,
        ..Default::default()
    },
    &vertex_shader,
    &pixel_shader,
)?;
```

Shaders are arguments to pipeline creation, not fields of the descriptor.

### Transient command buffers

A command buffer is created, recorded, submitted, and reclaimed automatically. There is no
explicit pool management at the API surface.

```rust
let mut cmd = device.create_command_buffer()?;
cmd.begin_render_pass(&pass_desc);
cmd.set_graphics_pipeline(&pso);
cmd.draw(root, None, 3, 1, 0, 0);
cmd.end_render_pass();
queue.submit(cmd)?;
```

Rendering is dynamic: there are no `VkRenderPass` objects to author. Attachments are described
inline at `begin_render_pass`.

### Timeline synchronization

Frame and cross-queue synchronization use timeline semaphores. For GPU-driven workflows, the
command buffer also exposes split signal and wait against a memory value (`signal_after` /
`wait_before`), so producers and consumers can rendezvous on a counter the GPU writes.

### One clip-space convention

Kiln normalizes NDC to Y-up on every backend, so a single Y-up projection matrix and the same
shader output render identically on Vulkan and Metal. The Vulkan backend achieves this with a
negative-height viewport while keeping counter-clockwise front faces; Metal is Y-up natively.
`device.clip_space_y()` always reports `ClipSpaceY::Up`.

## Architecture

```
src/
  lib.rs            public API re-exports
  device.rs         Device: resource creation, backend selection
  memory.rs         GpuAllocation, GpuBuffer, BumpAllocator, MemoryType, GpuPod
  command.rs        CommandBuffer: draws, dispatches, barriers, copies
  queue.rs          submit / acquire / present / submit_frame
  pipeline.rs       Graphics / Compute / Meshlet PSOs, depth-stencil + blend states
  shader.rs         ShaderModule (SPIR-V or MSL)
  texture.rs        textures + bindless views (TextureId)
  sampler.rs        samplers
  barrier.rs        StageFlags, HazardFlags
  sync.rs           TimelineSemaphore
  swapchain.rs      surface + swapchain
  accel.rs          BLAS / TLAS acceleration structures
  types.rs          GpuAddress, TextureId, Format, Topology, Cull, ...
  backend/
    vulkan/         Vulkan 1.3 backend (ash)
    metal/          Metal 4 backend (objc2)
```

The public types are thin handles. Each holds a backend enum (`Vulkan(..)` or `Metal(..)`) and
dispatches through a `backend_dispatch!` macro. This is static enum dispatch rather than `dyn Trait`,
so backend selection costs a branch the optimizer can usually fold away, with no vtable per call.
The two backends are gated behind the `vulkan` and `metal` Cargo features; only what you compile is
linked.

### Backends

- **Vulkan 1.3+**: dynamic rendering, buffer device address for the pointer model, and the
  descriptor buffer extension for the bindless heap. Validation layers are on by default in debug
  builds.
- **Metal 4**: argument tables for bindless, `MTL4` command queues and compilers, native mesh
  shaders and acceleration structures.

## Feature support

- Graphics, compute, and mesh-shader pipelines
- Bindless textures through a global heap indexed by `TextureId`
- Indirect and multi-draw, including GPU-written argument buffers
- Indexed and non-indexed draws with programmable index fetch
- Ray tracing: BLAS/TLAS build plus inline ray query in compute
- Timeline semaphores and GPU-side split signal/wait
- Dynamic rendering with inline attachment description
- MSAA, depth-stencil, and separate blend state

## A complete triangle

```rust
let device = Device::new(&DeviceDesc::default())?;

let vs = compile(&device, SHADER_SRC, "vsMain", ShaderStage::Vertex);
let fs = compile(&device, SHADER_SRC, "fsMain", ShaderStage::Pixel);

let pso = device.create_graphics_pso(
    &GraphicsPsoDesc {
        topology: Topology::TriangleList,
        color_targets: vec![ColorTarget::new(color_format)],
        cull: Cull::None,
        ..Default::default()
    },
    &vs,
    &fs,
)?;

// in the frame loop:
cmd.set_graphics_pipeline(&pso);
cmd.draw(None, None, 3, 1, 0, 0);
```

The vertex and pixel roots are `None` here because this shader generates its positions and colors
from `SV_VertexID`. A real workload passes the address of a per-draw struct instead.

## Building

```bash
# Metal (default on Apple platforms)
cargo build

# Vulkan
cargo build --no-default-features --features vulkan
```

At least one backend feature must be enabled. The examples need `slangc` on `PATH` to compile their
Slang shaders.

### Examples

```bash
cargo run --example triangle_graphics   # windowed raster triangle
cargo run --example triangle_mesh       # mesh-shader pipeline
cargo run --example cornell_box         # USD scene + ray tracing
```

### Tests

The `tests/` directory exercises each subsystem headlessly (graphics, compute, mesh, ray tracing,
textures, transfer, memory, device) by rendering to offscreen targets and reading the result back.

```bash
cargo test
```

## Status

Kiln RHI is version 0.1 and under active development. The API is not yet stable. It is a personal
project for experimenting with the pointer-first GPU model across Vulkan and Metal, not a
production rendering library.
