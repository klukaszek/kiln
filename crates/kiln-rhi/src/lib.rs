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
//! - Timeline semaphores for cross-submit ordering; frame pacing uses the swapchain fence
//! - Enum dispatch for zero-cost backend selection
//! - Mesh shader pipelines ([`Device::create_meshlet_pso`], [`CommandBuffer::draw_meshlets`])
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
//! # Shader authoring
//!
//! Per-draw data flows through one root struct. [`gpu_struct!`] emits the `#[repr(C)]` Rust type
//! plus a `SLANG` string to prepend to the shader source; take it as an entry-point `uniform`
//! pointer. [`AccelHandle`], [`TextureHandle`] and [`SamplerHandle`] fields map to Slang
//! `DescriptorHandle<..>` with no annotation.
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
//!   `$Globals` cbuffer at set 0, binding 0, colliding with the bindless heap. The compiler
//!   passes `-fvk-bind-globals 0 1` so a stray global becomes a missing-binding error instead.
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
        crate::accel::AccelerationStructure,
        crate::command::CommandBuffer,
        crate::memory::Allocation,
        crate::pipeline::ComputePso,
        crate::pipeline::GraphicsPso,
        crate::pipeline::MeshletPso,
        crate::query::QueryPool,
        crate::sampler::Sampler,
        crate::shader::ShaderModule,
        crate::surface::Surface,
        crate::swapchain::Swapchain,
        crate::sync::TimelineSemaphore,
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
pub use device::{Backend, BindlessMode, Device, DeviceDesc, DeviceResource};
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
    ALL_LAYERS, ALL_MIPS, Texture, TextureDesc, TextureUsage, TextureViewDesc, ViewKind,
    bytes_per_pixel, formats_are_view_compatible,
};
pub use types::*;
