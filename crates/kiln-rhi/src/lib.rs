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
//!
//! # Binding layout
//!
//! Set 0 is owned by the RHI. Shaders must not claim anything on set 0; all
//! per-draw/dispatch data arrives via the root BDA pointer.
//!
//! | Resource                | Vulkan                  | Metal            |
//! |-------------------------|-------------------------|------------------|
//! | Bindless sampled images | set 0, binding 0        | arg-table slot 1 |
//! | Bindless samplers       | set 0, binding 1        | arg-table slot 2 |
//! | Bindless storage images | set 0, binding 2        | (texture heap)   |
//! | Root data pointer       | push constant, offset 0 | buffer(0)        |
//! | All other buffers/SSBO  | BDA inside root struct  | BDA              |
//! | Acceleration structures | BDA inside root struct  | inline in root   |
//!
//! # Shader authoring rules
//!
//! ## Root structs
//!
//! All per-draw/dispatch data flows through a single root struct. Define it with
//! [`gpu_struct!`], which generates both the `#[repr(C)]` Rust type and a `SLANG` string
//! constant to prepend to shader source:
//!
//! ```ignore
//! gpu_struct! {
//!     pub struct DrawRoot {
//!         verts:  GpuAddress as "Vertex*",
//!         count:  u32,
//!         _pad:   u32,
//!     }
//! }
//! ```
//!
//! Accept the struct as an entry-point `uniform` pointer parameter in the shader:
//!
//! ```text
//! [shader("compute")]
//! [numthreads(64, 1, 1)]
//! void csMain(uint3 tid : SV_DispatchThreadID, uniform DrawRoot* r)
//! {
//!     if (tid.x >= r.count) return;
//!     Vertex v = r.verts[tid.x];
//! }
//! ```
//!
//! On Vulkan the root pointer is a push constant BDA; on Metal it is `buffer(0)`.
//! Structs must be padding-free -- add explicit `_pad` fields where alignment requires it.
//!
//! **Never declare root data as a module-scope `uniform` global.** Module-scope uniforms
//! collapse into a Slang `$Globals` cbuffer placed at set 0, binding 0, colliding with
//! the bindless heap. The compile harness passes `-fvk-bind-globals 0 1` so stray globals
//! land on set 1 and surface as a missing-binding error.
//!
//! ## Acceleration structures
//!
//! Use [`AccelHandle`] fields in the root struct. `AccelHandle` maps to
//! `DescriptorHandle<RaytracingAccelerationStructure>` on the Slang side with no
//! annotation required:
//!
//! ```ignore
//! gpu_struct! {
//!     pub struct TraceRoot {
//!         tlas: AccelHandle,
//!     }
//! }
//! // root.tlas = tlas_accel.gpu();
//! ```
//!
//! Slang lowers this to a 64-bit device address + `OpConvertUToAccelerationStructureKHR`
//! on Vulkan, and an inline `acceleration_structure` member in the Metal root buffer.
//! No descriptor set, no argument-table slot. For a bindless array, use
//! `GpuAddress as "DescriptorHandle<RaytracingAccelerationStructure>*"` and index
//! dynamically. Do not use `NonUniformResourceIndex` -- it is unavailable in Metal compute.

#[macro_use]
mod macros;

pub mod accel;
pub mod compiler;
pub mod backend;
pub mod barrier;
pub mod command;
pub mod device;
pub mod error;
pub mod memory;
pub mod pipeline;
pub mod query;
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
pub use query::QueryPool;
pub use queue::Queue;
pub use sampler::{Sampler, SamplerDesc};
pub use shader::{ShaderModule, ShaderModuleDesc, ShaderStage};
pub use surface::{Surface, SurfaceDesc};
pub use swapchain::{AcquiredImage, Swapchain, SwapchainDesc};
pub use sync::TimelineSemaphore;
pub use texture::{ALL_LAYERS, ALL_MIPS, GpuViewDesc, Texture, TextureDesc, TextureUsage};
pub use types::*;
pub use types::{
    AccelerationStructureId, BlasDesc, BlasMeshDesc, BuildAccelFlags, GeometryFlags, GeometryType,
    InstanceFlags, TlasDesc, TlasInstance,
};
