# Kiln

Kiln is a Rust render hardware interface for Vulkan and Metal that skips the descriptor-set model.
You write a struct, upload it, and hand its GPU address to the draw call:

```rust
cmd.set_pipeline(&pipeline);
cmd.draw(root.gpu(), vertex_count, 1, 0, 0);
```

That struct is the root. It carries every address and handle the shader needs. No bind groups, no
descriptor set layouts, no per-resource state tracking in the application.

Both APIs have supported this for years: buffer device address on Vulkan, argument tables on
Metal. Most portable RHIs still put bind groups on top anyway, because that's the common
denominator across everything they target. Kiln only targets two, and both are modern, so I wanted
to see what the interface looks like if you just assume addresses from the start.

This is a personal research project. Expect the API to move.

![Path-traced Cornell box](public/cornell_box.png)

## Workspace

| Package | Purpose |
| --- | --- |
| `kiln-rhi` | The RHI itself: memory, commands, pipelines, shaders, sync, presentation, ray tracing. |
| `kiln-app` | winit window and present loop shared by the examples. |
| `kiln-egui` | egui painter built on Kiln. |
| `triangle-graphics` | Smallest windowed graphics pipeline. |
| `triangle-mesh` | Same, through a mesh shader. |
| `egui-demo` | egui overlay. |
| `spectra` | USD importer with progressive spectral path tracing. |

## Requirements

Kiln targets two modern backends and nothing else. There is no fallback path and no capability
branching, by design, so the floor is high and stated exactly:

| | Vulkan | Metal |
| --- | --- | --- |
| **GPU** | NVIDIA Turing or newer (GeForce RTX 20-series and up), or any GPU whose driver exposes the extensions below | Apple silicon, M1 or newer |
| **OS** | Windows or Linux with a current driver | macOS 26.0 or newer |
| **API** | Vulkan 1.4 | Metal 4 |

The Vulkan backend additionally requires these device extensions. A GPU missing any of them is
skipped during adapter selection, so the failure surfaces as `RhiError::NoSuitableGpu` rather than
as a later crash:

| Extension | Why |
| --- | --- |
| `VK_EXT_descriptor_heap` | app-owned heaps, layout-free pipelines, `vkCmdPushDataEXT` |
| `VK_KHR_device_address_commands` | every command takes an address range |
| `VK_KHR_shader_untyped_pointers` | the descriptor-heap SPIR-V path needs it |
| `VK_KHR_unified_image_layouts` | every image stays in `GENERAL`, so nothing tracks layouts |
| `VK_EXT_mesh_shader` | mesh pipelines |
| `VK_KHR_acceleration_structure`, `VK_KHR_ray_query`, `VK_KHR_ray_tracing_maintenance1`, `VK_KHR_deferred_host_operations` | BLAS/TLAS and inline ray queries |

These are recent extensions, so a current driver matters as much as the hardware.
`vulkaninfo | grep descriptor_heap` is the quickest way to check a Vulkan machine.

## Quick start

You need Rust with the 2024 edition, and `slangc` new enough to lower `DescriptorHandle<T>` onto
`SPV_EXT_descriptor_heap` (2026.14.1 or later) on `PATH`.

Pick a backend explicitly:

```bash
cargo build --no-default-features --features vulkan
cargo build --no-default-features --features metal
```

Then run something:

```bash
cargo run -p triangle-graphics
cargo run -p triangle-mesh
cargo run -p egui-demo
cargo run -p spectra
```

Spectra is the interesting one. It opens the bundled Cornell box and path traces it progressively.
Camera is `WASD` plus left-drag to look. It also renders headless to a PNG:

```bash
cargo run --release -p spectra -- --scene cornell-box --spp 64 --headless 1024x1024
```

Scenes, light spectra, and the analysis dumps are covered in
[`examples/spectra/README.md`](examples/spectra/README.md). Every example takes `--help`.

Shaders compile through `slangc` and cache in your temp directory under `kiln-shader-cache/`. The
key covers the source and everything about how it got compiled, `slangc` version included, so
upgrading the compiler doesn't hand you a stale binary.

## Design

### Roots and addresses

An allocation hands you a GPU virtual address, plus a CPU pointer when the memory is mapped. The
`gpu_struct!` macro declares a root layout once and emits the `#[repr(C)]` Rust type alongside a
`DrawRoot::SLANG` string you prepend to the shader source, so host and device layouts can't drift.
Structs must be padding-free, hence the explicit tail padding:

```rust
gpu_struct! {
    pub struct DrawRoot {
        vertices: GpuPtr<Vertex>,
        count: u32,
        pad: u32,
    }
}
```

That emits the Rust type and `DrawRoot::SLANG`:

```slang
struct DrawRoot {
    Vertex* vertices;
    uint count;
    uint pad;
};
```

A root is just an upload-visible allocation, so the direct path is the whole story:

```rust
let mut root = device.create_allocation(&AllocationDesc {
    size: size_of::<DrawRoot>() as u64,
    memory: MemoryType::Upload,
    label: Some("draw-root".into()),
    ..Default::default()
})?;
root.upload(&DrawRoot {
    vertices: vertex_buffer.gpu().cast(),
    count: vertex_count,
    pad: 0,
})?;

cmd.draw(root.gpu(), vertex_count, 1, 0, 0);
```

Padding is written out like any other field. A zeroed base would spare the keystrokes and also
silently fill any real field you left out, so the bytes are named instead and the compiler keeps
checking omissions.

An allocation per root is fine for anything long-lived. For per-frame roots a mapped
`BumpAllocator` amortises it. Allocating counts as a pointer bump, and the whole arena is reclaimed at
once:

```rust
let mut frame_arena = BumpAllocator::new(device.create_allocation(&AllocationDesc {
    size: 64 * 1024,
    memory: MemoryType::Upload,
    label: Some("frame-roots".into()),
    ..Default::default()
})?);

frame_arena.reset();
let root = frame_arena
    .alloc(size_of::<DrawRoot>() as u64, 16)
    .expect("frame arena exhausted")
    .cast::<DrawRoot>();
root.write(&DrawRoot {
    vertices: vertex_buffer.gpu().cast(),
    count: vertex_count,
    pad: 0,
})?;

cmd.draw(root.gpu(), vertex_count, 1, 0, 0);
```

Keep one arena per in-flight frame slot. `reset()` is only safe once that slot's previous GPU work
has retired. The arena has no idea what the GPU is still reading, so waiting on that fence is on
you.

### Bindless resources

Sampled and storage views go into a global heap and travel through roots as small handles like
`TextureHandle` and `SamplerHandle`, which `gpu_struct!` spells as Slang `DescriptorHandle<T>`.
Vulkan backs the heap with `VK_EXT_descriptor_heap`, Metal with argument tables, and neither shows
through. The heaps are bound once per command buffer and never rebound.

`AccelHandle` is the one exception. Metal reaches an acceleration structure through the same
bindless table as everything else, while Vulkan passes its device address and converts, so
`gpu_struct!` emits a `RaytracingAccelerationStructure` property instead. Shader code still just
reads the field.

### Barriers name stages, not resources

```rust
cmd.barrier(StageFlags::COMPUTE, StageFlags::VERTEX_SHADER);
```

A producer stage and a consumer stage. Hazard flags cover the cases that need an extra cache or
argument-buffer dependency. Nothing on the application side tracks per-resource layout.

Timeline semaphores order work across submissions; frame pacing waits on the swapchain's
per-frame fence. Command buffers are transient and go back to the pool after submission.

### Shaders

Everything is authored in Slang and compiled to SPIR-V or metallib. Root data arrives as a pointer
parameter to the entry point. There are no descriptor sets to collide with: pipelines are created
without a layout on both backends. Declaring root data as a module-scope `uniform` is the one thing
to avoid, since Slang collapses those into a `$Globals` cbuffer that has nowhere to bind.

Clip space is Y-up on both backends. Projection matrices and shader code don't need a
backend-specific vertical flip.

### Coverage

The public types dispatch over whichever backend is compiled in. What works today:

- graphics, compute, and mesh shader pipelines
- bindless textures, dynamic rendering, depth buffering
- indirect dispatch, indexed draws, and meshlet draws
- BLAS/TLAS ray tracing with inline ray queries in compute

`SampleCount` reaches the pipeline's multisample state, but nothing resolves a multisampled target
yet and nothing exercises it, so treat anything above `S1` as unimplemented.

The backends are private. Everything goes through the public types, and there is no escape hatch to
a raw `VkDevice` or `MTLDevice`.

## Development

```bash
cargo fmt --all -- --check
cargo check --workspace
cargo test --workspace
```

The integration tests render to offscreen targets and cover the RHI end to end. They need a GPU
and `slangc`, and fail without them — this is a render hardware interface, so a machine that
cannot run them is broken, not exempt. Tests enable validation by default; run with
`-- --nocapture` to see the layer output. Select a backend explicitly with
`--no-default-features --features metal` or `--no-default-features --features vulkan`.

Pass `--validation` to a windowed example for Vulkan validation layers.

Metal API validation is configured before process startup through Xcode or `MTL_DEBUG_LAYER=1`,
as described in [Apple's validation guide](https://developer.apple.com/documentation/xcode/validating-your-apps-metal-api-usage).
`DeviceDesc.validation` controls Vulkan validation; it does not toggle Metal's process-wide layer.

Writable `Allocation::mapped` handles and `Device::write_tlas_instance` require a mutable
allocation borrow. Mapped handles remain copyable for pointer arithmetic and checked writes;
their slice accessors are `unsafe` because copies can overlap. Prefer `Allocation::as_slice` /
`as_mut_slice` for safe borrowed slices, and `Mapped::read` / `write` for individual values.

## Resource lifetime

`Device::destroy` is safe to call the moment you are done with a resource, including mid-frame and
immediately after submitting work that reads it. The handle is consumed at once, but the storage
and any bindless slot are held until every submission issued so far has retired, and reclaimed by
the next submit, `acquire_image`, or `wait_idle`. You never need a fence of your own for this.

`wait_idle` and `wait_for_frame` remain available for the cases that genuinely need a drain, such
as resizing a swapchain.

## Threading

The RHI is single-threaded by design: `Device` is `Rc`-backed and neither it nor a `CommandBuffer`
is `Send`. Both backends use `Rc`/`RefCell` internally to match, so there is no lock traffic on the
record path. Parallel command recording would be a deliberate future change, not something the
current types quietly allow.

## Shader compilation

`kiln_rhi::compiler` compiles Slang source at runtime by shelling out to a `slangc` binary on
`PATH`, caching artifacts in the temp dir. That suits tests, examples and iteration.
