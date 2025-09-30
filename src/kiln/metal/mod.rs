//! Safe, curated Metal facade for Kiln.
//!
//! - Zero-copy by design using `IntoBytes + FromBytes + Copy + Immutable` for buffers.
//! - No public exposure of unsafe protocol types; only safe wrappers.
//! - Organized into submodules: device, pipeline, command, pass, encoder, drawable, argument.

pub use crate::kiln::gfx::{PackedFloat3, SceneProperties, VertexInput};

// Safe data-only Metal types (no constructors that require `unsafe`).
pub use objc2_metal::{
    MTLClearColor, MTLLoadAction, MTLPixelFormat, MTLPrimitiveType, MTLRenderStages,
    MTLResourceOptions, MTLStoreAction,
};

mod device;
mod pipeline;
mod command;
mod pass;
mod encoder;
mod drawable;
mod argument;
mod buffer;
mod surface;

pub use device::Device;
pub use pipeline::{Library, PipelineState, RenderPipelineBuilder};
pub use command::{Queue, CommandAllocator, CommandBuffer};
pub use pass::RenderPass;
pub use encoder::RenderEncoder;
pub use drawable::Drawable;
pub use argument::{ArgumentBuffer, Bindable};
pub use surface::{MTLDrawableSource, SwapchainConfig, PresentMode, ColorSpace};
pub use buffer::{IndexBuffer, Uniform, Vertex, HasGpuAddress};

// Re-export shader-friendly POD types that examples use
// Already re-exported above; remove duplication
