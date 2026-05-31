//! Kiln RHI - Render Hardware Interface
//!
//! An Aaltonen "No Graphics API"-inspired abstraction over Vulkan and Metal.
//!
//! Core design principles:
//! - Dual-pointer memory model: every GPU allocation returns (CPU ptr, GPU address)
//! - Single root data pointer per draw/dispatch (no descriptor sets, no bind groups)
//! - Global texture heap indexed by TextureId(u32)
//! - Stage-only barriers (no per-resource state tracking)
//! - Minimal PSO (topology + formats + MSAA baked; separate DepthStencil/Blend)
//! - Transient command buffers (create, record, submit, auto-reclaim)
//! - Timeline semaphores for frame sync
//! - Enum dispatch for zero-cost backend selection
//! - Mesh shader pipelines (gpuCreateGraphicsMeshletPipeline / gpuDrawMeshlets)
//! - Ray tracing (BLAS/TLAS + inline ray query in compute)

#[macro_use]
mod macros;

pub mod accel;
pub mod backend;
pub mod barrier;
pub mod command;
pub mod device;
pub mod error;
pub mod memory;
pub mod pipeline;
pub mod queue;
pub mod sampler;
pub mod shader;
pub mod surface;
pub mod swapchain;
pub mod sync;
pub mod texture;
pub mod types;

// The RHI is built around zerocopy for its GPU data contract (`GpuPod`, `gpu_struct!`,
// the indirect-args structs). Re-export it so the `gpu_struct!` macro and downstream
// crates share one zerocopy instance.
pub use zerocopy;

// Re-export core types at crate root for convenience
pub use accel::AccelerationStructure;
pub use barrier::{HazardFlags, StageFlags};
pub use command::{
    ColorAttachment, CommandBuffer, DepthAttachment, DispatchIndirectArgs, DrawIndexedIndirectArgs,
    DrawIndirectArgs, DrawIndirectMultiArgs, LoadOp, RenderPassDesc, RenderTarget, SignalOp,
    SignalValueDesc, StoreOp, WaitOp, WaitValueDesc,
};
pub use device::{Backend, BindlessMode, Device, DeviceDesc};
pub use error::{RhiError, RhiResult};
pub use memory::{
    BufferDesc, BumpAllocator, GpuAllocation, GpuBuffer, GpuPod, MemoryType, TransientAllocation,
};
pub use pipeline::*;
pub use queue::Queue;
pub use sampler::{Sampler, SamplerDesc};
pub use shader::{ShaderModule, ShaderModuleDesc, ShaderStage};
pub use surface::{Surface, SurfaceDesc};
pub use swapchain::{AcquiredImage, Swapchain, SwapchainDesc};
pub use sync::TimelineSemaphore;
pub use texture::{GpuViewDesc, Texture, TextureDesc, TextureUsage, ALL_LAYERS, ALL_MIPS};
pub use types::*;
pub use types::{
    AccelerationStructureId, BlasDesc, BlasMeshDesc, BuildAccelFlags, GeometryFlags, GeometryType,
    InstanceFlags, TlasDesc, TlasInstance,
};
