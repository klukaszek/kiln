//! Kiln RHI - Render Hardware Interface
//!
//! An Aaltonen "No Graphics API"-inspired abstraction over Vulkan and Metal.
//!
//! Core design principles:
//! - One allocation model: optional CPU mapping + typed GPU pointer + byte length
//! - Single root data pointer per draw/dispatch (no descriptor sets, no bind groups)
//! - Opaque shader handles for textures, samplers, and acceleration structures
//! - Stage-only barriers (no per-resource state tracking)
//! - Minimal PSO (topology + formats + MSAA + blend baked; separate DepthStencil)
//! - Transient command buffers (create, record, submit, auto-reclaim)
//! - Timeline semaphores for frame sync
//! - Enum dispatch for zero-cost backend selection
//! - Mesh shader pipelines (gpuCreateGraphicsMeshletPipeline / gpuDrawMeshlets)
//! - Ray tracing (BLAS/TLAS + inline ray query in compute)
//!
//! # Binding layout
//!
//! Set 0 is owned by the RHI. Shaders must not claim anything on set 0; all
//! per-draw/dispatch data arrives via the root pointer.
//!
//! | Resource                | Vulkan                  | Metal            |
//! |-------------------------|-------------------------|------------------|
//! | Bindless sampled images | set 0, binding 2        | arg-table slot 1 |
//! | Bindless samplers       | set 0, binding 0        | arg-table slot 2 |
//! | Bindless storage images | set 0, binding 2        | inline in root   |
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
//! type VertexPtr = GpuPtr<Vertex>;
//!
//! gpu_struct! {
//!     pub struct DrawRoot {
//!         verts:  VertexPtr as "Vertex*",
//!         count:  u32,
//!         _pad:   u32,
//!     }
//! }
//! ```
//! where `VertexPtr` is a local alias for `GpuPtr<Vertex>`. The pointer is still exactly one
//! 64-bit GPU address; the Rust type only supplies element arithmetic and documents the ABI.
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
//! `GpuPtr<AccelHandle> as "DescriptorHandle<RaytracingAccelerationStructure>*"` and index
//! dynamically. Do not use `NonUniformResourceIndex` -- it is unavailable in Metal compute.

#[macro_use]
mod macros;

pub mod accel;
pub(crate) mod backend;
pub mod barrier;
pub mod command;
pub mod compiler;
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

/// Native handles for narrowly-scoped backend interop.
pub mod raw {
    #[cfg(feature = "vulkan")]
    pub use crate::backend::vulkan::device::VulkanHandles;
}

// The RHI is built around zerocopy for its GPU data contract (`GpuPod`, `gpu_struct!`,
// the indirect-args structs). Re-export it so the `gpu_struct!` macro and downstream
// crates share one zerocopy instance.
pub use zerocopy;

/// Run one frame's work inside whatever scope the backend needs for its temporaries.
///
/// On Metal this is an autorelease pool, and it is not optional: `nextDrawable`, its texture and
/// every encoder are autoreleased, so without a per-frame scope their pending releases pile up
/// until the event loop drains them all at once — periodic CPU spikes with flat GPU time. On
/// Vulkan this compiles to a direct call.
pub fn frame_scope<R>(f: impl FnOnce() -> R) -> R {
    #[cfg(feature = "metal")]
    {
        objc2::rc::autoreleasepool(|_| f())
    }
    #[cfg(not(feature = "metal"))]
    {
        f()
    }
}

pub use accel::AccelerationStructure;
pub use barrier::{HazardFlags, StageFlags};
pub use command::{
    ColorAttachment, CommandBuffer, DepthAttachment, DispatchIndirectArgs, DrawIndexedIndirectArgs,
    DrawIndirectArgs, LoadOp, RenderPassDesc, RenderTarget, StoreOp,
};
pub use device::{Backend, BindlessMode, Device, DeviceDesc};
pub use error::{RhiError, RhiResult};
pub use memory::{
    Allocation, AllocationDesc, BumpAllocator, GpuPod, MemoryType, TransientAllocation,
};
pub use pipeline::*;
pub use query::QueryPool;
pub use queue::Queue;
pub use sampler::{Sampler, SamplerDesc};
pub use shader::{ShaderModule, ShaderModuleDesc, ShaderStage};
pub use surface::{Surface, SurfaceDesc};
pub use swapchain::{AcquiredImage, Swapchain, SwapchainDesc};
pub use sync::TimelineSemaphore;
pub use texture::{ALL_LAYERS, ALL_MIPS, Texture, TextureDesc, TextureUsage, TextureViewDesc};
pub use types::*;
pub use types::{
    AccelerationStructureId, BlasDesc, BlasMeshDesc, BuildAccelFlags, GeometryFlags, GeometryType,
    InstanceFlags, TlasDesc, TlasInstance,
};
