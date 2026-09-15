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
//! - `destroy` whenever you are done: releases are held until the work referencing them retires
//! - Single-threaded: `Device` and `CommandBuffer` are `Rc`-backed and deliberately not `Send`
//! - Validation is left to the backends; the RHI only checks what they cannot see
//! - Timeline semaphores for cross-submit ordering; frame pacing uses the swapchain fence
//! - Enum dispatch for zero-cost backend selection
//! - Mesh shader pipelines ([`Device::create_meshlet_pso`], [`CommandBuffer::draw_meshlets`])
//! - Ray tracing (BLAS/TLAS + inline ray query in compute)
//!
//! # Binding layout
//!
//! There is nothing to bind. Neither backend uses descriptor sets or pipeline layouts, and every
//! pipeline is created without one; all per-draw/dispatch data arrives through the root pointer.
//!
//! | Resource                | Vulkan                     | Metal            |
//! |-------------------------|----------------------------|------------------|
//! | Bindless textures       | resource heap slot         | arg-table slot 1 |
//! | Bindless samplers       | sampler heap slot          | arg-table slot 2 |
//! | Root data pointer       | `vkCmdPushDataEXT`         | buffer(0)        |
//! | All other buffers       | address inside root struct | address          |
//! | Acceleration structures | address inside root struct | arg-table slot   |
//!
//! Both heaps are bound once per command buffer and never rebound. A texture or sampler handle is
//! its slot index on Vulkan and its `gpuResourceID` on Metal; either way the shader just indexes
//! what `gpu_struct!` declared.
//!
//! # Shader authoring
//!
//! Per-draw data flows through one root struct. [`gpu_struct!`] emits the `#[repr(C)]` Rust type
//! plus a `SLANG` string to prepend to the shader source; take it as an entry-point `uniform`
//! pointer. [`TextureHandle`] and [`SamplerHandle`] fields become Slang `DescriptorHandle<..>`
//! with no annotation; [`AccelHandle`] becomes a `RaytracingAccelerationStructure` property, since
//! that is the one resource the two backends genuinely reach differently. Either way the shader
//! reads the field and gets the resource.
//!
//! ```text
//! [shader("compute")] [numthreads(64, 1, 1)]
//! void csMain(uint3 tid : SV_DispatchThreadID, uniform DrawRoot* r)
//! {
//!     if (tid.x >= r.count) return;
//!     Vertex v = r.verts[tid.x];
//! }
//! ```
//!
//! Two things bite:
//!
//! - **Never declare root data as a module-scope `uniform`.** Slang collapses those into a
//!   `$Globals` cbuffer, which the descriptor-heap path has nowhere to bind. Declare it as an
//!   entry-point `uniform Root*` parameter instead.
//! - **No `NonUniformResourceIndex`.** It is unavailable in Metal compute.

#[macro_use]
mod macros;

/// Seals [`DeviceResource`] and [`Pipeline`]. Implemented once per type here, since a type may
/// satisfy both.
mod sealed {
    pub trait Sealed {}

    macro_rules! impl_sealed {
        ($($ty:ty),+ $(,)?) => { $( impl Sealed for $ty {} )+ };
    }

    impl_sealed!(
        crate::memory::Allocation,
        crate::pipeline::ComputePso,
        crate::pipeline::GraphicsPso,
        crate::pipeline::MeshletPso,
        crate::query::QueryPool,
        crate::sampler::Sampler,
        crate::texture::Texture,
    );
}

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
    DrawIndirectArgs, LoadOp, Pipeline, RenderPassDesc, RenderTarget, StoreOp,
};
pub use device::{Backend, Device, DeviceDesc, DeviceResource};
pub use error::{RhiError, RhiResult};
pub use memory::{
    Allocation, AllocationDesc, BumpAllocator, DEFAULT_ALIGN, GpuPod, Mapped, MemoryType,
};
pub use pipeline::*;
pub use query::QueryPool;
pub use queue::{Queue, SubmitDesc};
pub use sampler::{Sampler, SamplerDesc};
pub use shader::{ShaderModule, ShaderModuleDesc, ShaderStage};
pub use surface::{Surface, SurfaceDesc};
pub use swapchain::{AcquiredImage, Swapchain, SwapchainDesc};
pub use sync::TimelineSemaphore;
pub use texture::{
    ALL_LAYERS, ALL_MIPS, Texture, TextureDesc, TextureRegion, TextureUsage, TextureViewDesc,
    ViewKind, bytes_per_pixel, formats_are_view_compatible,
};
pub use types::*;
