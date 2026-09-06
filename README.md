# Kiln

Kiln is a Rust render hardware interface for Vulkan and Metal that skips the descriptor-set model.
You write a struct, upload it, and hand its GPU address to the draw call:

```rust
cmd.set_graphics_pipeline(&pipeline);
cmd.draw(root.gpu(), vertex_count, 1, 0, 0);
```

That struct is the root. It carries every address and handle the shader needs. No bind groups, no
descriptor set layouts, no per-resource state tracking in the application.

Both APIs have supported this for years: buffer device address on Vulkan, argument buffers on
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
| `spectra` | USD importer with progressive spectral path tracing and a raster fallback. |

## Quick start

You need Rust with the 2024 edition, `slangc` on `PATH`, and either a Vulkan 1.3 driver or an Apple
platform with Metal 4.

Metal is the default feature. Vulkan is opt-in:

```bash
cargo build
cargo build --no-default-features --features vulkan
```

Then run something:

```bash
cargo run -p triangle-graphics
cargo run -p triangle-mesh
cargo run -p egui-demo
cargo run -p spectra
```

Spectra is the interesting one. It opens the bundled Cornell box and path traces it progressively,
falling back to raster if the spectral backend won't initialize. Camera is `WASD` plus left-drag to
look. It also renders headless to a PNG:

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
        vertices: GpuAddress as "Vertex*",
        count: u32,
        _pad: u32,
    }
}
```

Per-frame roots come from a mapped `BumpAllocator` instead of individual allocations. Allocating is
a pointer bump, and the whole arena gets reclaimed at once:

```rust
let buffer = device.create_buffer(&BufferDesc {
    size: 64 * 1024,
    memory: MemoryType::Default,
    label: Some("frame-roots".into()),
})?;
let mut frame_arena = BumpAllocator::new(buffer);

frame_arena.reset();
let root = frame_arena
    .alloc(std::mem::size_of::<DrawRoot>() as u64, 16)
    .expect("frame arena exhausted");
root.upload(&DrawRoot {
    vertices: vertex_buffer.gpu(),
    count: vertex_count,
    _pad: 0,
})?;

cmd.draw(root.gpu, vertex_count, 1, 0, 0);
```

Keep one arena per in-flight frame slot. `reset()` is only safe once that slot's previous GPU work
has retired. The arena has no idea what the GPU is still reading, so waiting on that fence is on
you.

### Bindless resources

Sampled and storage views go into a global heap and travel through roots as small handles like
`TextureHandle`, `SamplerHandle`, and `AccelHandle`, which `gpu_struct!` spells as Slang
`DescriptorHandle<T>`. Vulkan backs the heap with descriptor buffers, Metal with argument tables,
and neither shows through.

### Barriers name stages, not resources

```rust
cmd.barrier(StageFlags::COMPUTE, StageFlags::VERTEX_SHADER);
```

A producer stage and a consumer stage. Hazard flags cover the cases that need an extra cache or
argument-buffer dependency. Nothing on the application side tracks per-resource layout.

Timeline semaphores handle frame pacing and cross-queue work. Command buffers are transient and go
back to the pool after submission.

### Shaders

Everything is authored in Slang and compiled to SPIR-V or metallib. Root data arrives as a pointer
parameter to the entry point. Set 0 belongs to the RHI's bindless heap, so application shaders
can't claim it.

Clip space is Y-up on both backends. Projection matrices and shader code don't need a
backend-specific vertical flip.

### Coverage

The public types dispatch over whichever backend is compiled in. What works today:

- graphics, compute, and mesh shader pipelines
- bindless textures, dynamic rendering, MSAA, depth/stencil
- indirect dispatch, indexed draws, and meshlet draws
- BLAS/TLAS ray tracing with inline ray queries in compute

The backend-specific handles are still reachable if you need to drop through to them.

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
`--no-default-features --features metal` or `--features vulkan`.

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

## Validation

The RHI does not re-implement validation. Formats, usage flags, layouts and subresource ranges are
checked by the Vulkan validation layers and Metal's API validation, which report them better than a
hand-rolled layer could. What the RHI does check is the small set those validators structurally
cannot see — chiefly mixing resources between devices — and only under `debug_assertions`.

## Threading

The RHI is single-threaded by design: `Device` is `Rc`-backed and neither it nor a `CommandBuffer`
is `Send`. Both backends use `Rc`/`RefCell` internally to match, so there is no lock traffic on the
record path. Parallel command recording would be a deliberate future change, not something the
current types quietly allow.

## Shader compilation

The `slangc` feature (on by default) compiles Slang source at runtime by shelling out to a `slangc`
binary on `PATH`, caching artifacts in the temp dir. That suits tests, examples and iteration. A
shipping build should turn the feature off, compile shaders offline, and hand the bytes to
`Device::create_shader_module`.
