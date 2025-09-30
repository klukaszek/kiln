#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

pub mod kiln {
    pub mod app; // now a module tree: app::{events, renderer, swapchain, windowing}
    pub mod gfx;
    pub mod metal;
}

// Example binaries live under `examples/` and are not part of the library API.

// Top-level re-exports for cleaner imports: `use kiln::{app, gfx, renderer, swapchain};`
pub use crate::kiln::{app, gfx, metal};

// Short, Metal-style alias with simplified names for discoverability.
pub mod mtl {
    // Core objects
    pub use crate::kiln::metal::{
        Device, Library, PipelineState, RenderPipelineBuilder, Queue, CommandAllocator, CommandBuffer,
        RenderPass, RenderEncoder, Drawable, ArgumentBuffer, Bindable, Uniform, Vertex, IndexBuffer,
        HasGpuAddress, SwapchainConfig, PresentMode, ColorSpace,
    };
    // Surface source
    pub use crate::kiln::metal::MTLDrawableSource as DrawableSource;
    // Data-only enums/structs (drop the MTL* prefix at the callsite)
    pub use crate::kiln::metal::{
        MTLClearColor as ClearColor,
        MTLLoadAction as LoadAction,
        MTLStoreAction as StoreAction,
        MTLPixelFormat as PixelFormat,
        MTLPrimitiveType as PrimitiveType,
        MTLRenderStages as RenderStages,
        MTLResourceOptions as ResourceOptions,
    };
    // Shader-friendly PODs
    pub use crate::kiln::metal::{PackedFloat3, SceneProperties, VertexInput};
}
