# Kiln Metal Facade: Usage Scaffolding

This guide shows how to use `kiln::metal` without the higher‑level `kiln::app` runtime. You bring a source of `CAMetalDrawable`s (e.g., an `MTKView` or `CAMetalLayer`) and implement `MTLDrawableSource`. The rest is safe, zero‑copy Metal via Kiln.

## 1) Implement `MTLDrawableSource`

```rust
use kiln::metal::{MTLDrawableSource, MTLPixelFormat};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;

struct MyViewSurface { /* your fields: MTKView/CAMetalLayer + device */ }

impl MTLDrawableSource for MyViewSurface {
    fn current_mtl4_render_pass_descriptor(&self) -> Option<Retained<objc2_metal::MTL4RenderPassDescriptor>> { /* from view/layer */ }
    fn current_drawable(&self) -> Option<Retained<ProtocolObject<dyn objc2_quartz_core::CAMetalDrawable>>> { /* next drawable */ }
    fn device(&self) -> Retained<ProtocolObject<dyn objc2_metal::MTLDevice>> { /* your device */ }
    fn color_pixel_format(&self) -> kiln::metal::MTLPixelFormat { /* view/layer pixel format */ }
}
```

## 2) Create device and resources (zero‑copy)

```rust
use kiln::metal::{Device, MTLResourceOptions};

let dev = Device::from_surface(&my_surface);
let lib = dev.compile_library_from_source("lib", include_str!("shader.metal"))?;
let pso = dev.pipeline_builder(&lib)
    .vertex("vertex_main")
    .fragment("fragment_main")
    .color_format(my_surface.color_pixel_format())
    .build()?;

let verts: [kiln::metal::VertexInput; 3] = [ /* ... */ ];
let vbuf = dev.vertex_buffer_from_slice(&verts, MTLResourceOptions::CPUCacheModeDefaultCache);
let scene = dev.uniform_buffer_with_len::<kiln::metal::SceneProperties>(1, MTLResourceOptions::CPUCacheModeDefaultCache);
let args = dev.new_argument_buffer(2, 0);
```

## 3) Render loop

```rust
use kiln::metal::{Queue, CommandAllocator, CommandBuffer, RenderPass, RenderEncoder, MTLRenderStages, MTLPrimitiveType};

let queue = dev.new_queue();
let alloc = dev.new_command_allocator();

loop {
    // Update scene (zero‑copy)
    scene.write_one(0, &kiln::metal::SceneProperties { time: t })?;
    args.bind2(0, &scene, 1, &vbuf);

    // Acquire pass + drawable
    let Some(rp) = RenderPass::from_surface_current(&my_surface) else { continue; };
    let Some(drawable) = kiln::metal::Drawable::from_surface_current(&my_surface) else { continue; };

    // Command buffer + encoder
    alloc.reset();
    let Some(cmd) = CommandBuffer::begin_with_allocator(&dev, &alloc) else { continue; };
    rp.set_clear(kiln::metal::MTLClearColor { red: 0.1, green: 0.1, blue: 0.12, alpha: 1.0 }, kiln::metal::MTLLoadAction::Clear);
    if let Some(enc) = cmd.render_encoder(&rp) {
        enc.set_pipeline(&pso);
        enc.set_argument_table_at_stages(&args, MTLRenderStages::Vertex);
        enc.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
        enc.end();
    }
    cmd.end();

    // Present
    queue.wait_for_drawable(&drawable);
    queue.commit_one(&cmd);
    queue.signal_drawable(&drawable);
    drawable.present();
}
```

Everything here is safe and zero‑copy by design.

## Choosing API style: `kiln::metal` vs `kiln::mtl`

Kiln exposes the same safe Metal facade under two module paths:

- `kiln::metal` — explicit, closer to Rust naming, full type names like `MTLDrawableSource`, `ArgumentBuffer`, etc.
- `kiln::mtl` — concise alias mirroring Metal naming closely. Examples:
  - `kiln::mtl::DrawableSource` (alias of `metal::MTLDrawableSource`)
  - `kiln::mtl::{Device, Queue, CommandBuffer, RenderPass, RenderEncoder}`
  - Enums re-exported as short names: `PixelFormat`, `PrimitiveType`, `RenderStages`, `ResourceOptions`, `ClearColor`, `LoadAction`, `StoreAction`.

Pick the style that fits your codebase; both are the same safe, zero‑copy API.
