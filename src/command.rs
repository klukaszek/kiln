use crate::accel::AccelerationStructure;
use crate::barrier::{HazardFlags, StageFlags};
use crate::pipeline::{
    BlendState, ComputePso, DepthStencilState, GraphicsPso, MeshletPso,
};
use crate::types::*;
use crate::types::{BlasDesc, TlasDesc};

/// Color attachment for dynamic rendering.
#[derive(Clone, Debug)]
pub struct ColorAttachment {
    /// Index into swapchain images or a TextureId for offscreen.
    pub target: RenderTarget,
    pub load_op: LoadOp,
    pub store_op: StoreOp,
    pub clear_color: [f32; 4],
}

/// Depth attachment for dynamic rendering.
#[derive(Clone, Debug)]
pub struct DepthAttachment {
    pub target: RenderTarget,
    pub load_op: LoadOp,
    pub store_op: StoreOp,
    pub clear_depth: f32,
    pub clear_stencil: u8,
}

/// Render target reference.
#[derive(Clone, Debug)]
pub enum RenderTarget {
    /// Swapchain image by index.
    SwapchainImage(u32),
    /// Off-screen texture by TextureId.
    Texture(TextureId),
}

/// Load operation for attachments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadOp {
    Load,
    Clear,
    DontCare,
}

/// Store operation for attachments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreOp {
    Store,
    DontCare,
}

/// Description for beginning dynamic rendering.
#[derive(Clone, Debug)]
pub struct RenderPassDesc {
    pub color_attachments: Vec<ColorAttachment>,
    pub depth_attachment: Option<DepthAttachment>,
    pub render_area: [u32; 4], // x, y, width, height
}

/// Arguments for non-indexed indirect draws.
#[repr(C)]
#[derive(Clone, Copy, Debug, zerocopy::IntoBytes, zerocopy::FromBytes, zerocopy::Immutable)]
pub struct DrawIndirectArgs {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

/// Arguments for indexed indirect draws (matches VkDrawIndexedIndirectCommand layout).
#[repr(C)]
#[derive(Clone, Copy, Debug, zerocopy::IntoBytes, zerocopy::FromBytes, zerocopy::Immutable)]
pub struct DrawIndexedIndirectArgs {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
}

/// Arguments for multi-draw indirect (layout matches `VkDrawIndirectCommand`).
///
/// The hardware draw is non-indexed: the vertex shader reads `draw_id`, finds its per-draw
/// root, and does programmable index fetch from a pointer in that root.
#[repr(C)]
#[derive(Clone, Copy, Debug, zerocopy::IntoBytes, zerocopy::FromBytes, zerocopy::Immutable)]
pub struct DrawIndirectMultiArgs {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

/// Arguments for indirect dispatch (matches VkDispatchIndirectCommand layout).
#[repr(C)]
#[derive(Clone, Copy, Debug, zerocopy::IntoBytes, zerocopy::FromBytes, zerocopy::Immutable)]
pub struct DispatchIndirectArgs {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// Atomic signal operation for split synchronization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignalOp {
    /// Write the value unconditionally.
    AtomicSet,
    /// Atomically update the counter to max(current, value). Used for timeline semaphores.
    AtomicMax,
    /// Atomically OR the value into the counter. Used for bitmask completion patterns.
    AtomicOr,
}

/// Value comparison operation for split synchronization waits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitOp {
    Equal,
    GreaterOrEqual,
    MaskedEqual,
}

/// Producer-side value signal descriptor.
#[derive(Clone, Copy, Debug)]
pub struct SignalValueDesc {
    pub src_stage: StageFlags,
    pub value_ptr: GpuAddress,
    pub value: u64,
    pub signal_op: SignalOp,
}

/// Consumer-side value wait descriptor.
#[derive(Clone, Copy, Debug)]
pub struct WaitValueDesc {
    pub dst_stage: StageFlags,
    pub value_ptr: GpuAddress,
    pub value: u64,
    pub wait_op: WaitOp,
    pub hazard: HazardFlags,
    pub mask: u64,
}

/// Transient command buffer. Created, recorded, submitted, auto-reclaimed.
pub struct CommandBuffer {
    pub(crate) inner: CommandBufferInner,
}

/// Resolve an optional root pointer: `None` (a draw that carries no root data) maps to
/// the null GPU address, which the backends bind as a never-dereferenced `buffer(0)`.
#[inline]
fn root_or_null(root: impl Into<Option<GpuAddress>>) -> GpuAddress {
    root.into().unwrap_or(GpuAddress::NULL)
}

pub(crate) enum CommandBufferInner {
    #[cfg(feature = "vulkan")]
    Vulkan(Box<crate::backend::vulkan::command::VulkanCommandBuffer>),
    #[cfg(feature = "metal")]
    Metal(Box<crate::backend::metal::command::MetalCommandBuffer>),
}

impl CommandBuffer {
    // -- Dynamic rendering --

    /// Begin dynamic rendering (no VkRenderPass objects).
    pub fn begin_render_pass(&mut self, desc: &RenderPassDesc) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.begin_render_pass(desc))
    }

    /// End dynamic rendering.
    pub fn end_render_pass(&mut self) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.end_render_pass())
    }

    // -- Pipeline state --

    /// Set the active graphics pipeline.
    pub fn set_graphics_pipeline(&mut self, pso: &GraphicsPso) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.set_graphics_pipeline(pso))
    }

    /// Set the active compute pipeline.
    pub fn set_compute_pipeline(&mut self, pso: &ComputePso) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.set_compute_pipeline(pso))
    }

    /// `gpuSetPipeline` for mesh pipelines — set before `draw_meshlets`/`draw_meshlets_indirect`.
    pub fn set_meshlet_pipeline(&mut self, pso: &MeshletPso) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.set_meshlet_pipeline(pso))
    }

    /// Set depth-stencil state.
    pub fn set_depth_stencil_state(&mut self, state: &DepthStencilState) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.set_depth_stencil_state(state))
    }

    /// Set blend state.
    pub fn set_blend_state(&mut self, state: &BlendState) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.set_blend_state(state))
    }

    // -- Root data (internal — callers never set these separately) --

    fn set_root_data(&mut self, vertex_root: GpuAddress, pixel_root: GpuAddress) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.set_root_data(vertex_root, pixel_root))
    }

    fn set_compute_root(&mut self, root: GpuAddress) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.set_compute_root(root))
    }

    /// Set the active bindless texture heap pointer for subsequent pipeline binds.
    /// Relevant for descriptor-buffer style backends.
    pub fn set_active_texture_heap_ptr(&mut self, heap_ptr: GpuAddress) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.set_active_texture_heap_ptr(heap_ptr))
    }

    // -- Draw / Dispatch --
    // Every call includes its root data pointer(s) — no pre-call dance, no separate set_root_data.

    /// Non-indexed draw. 🔵 Extension (spec only has indexed draws).
    ///
    /// Pass `None` for a root the shader doesn't read; otherwise a `GpuAddress` (bare,
    /// no `Some`). Vertex and pixel share one root table, so pass the same pointer for both.
    pub fn draw(
        &mut self,
        vertex_root: impl Into<Option<GpuAddress>>,
        pixel_root: impl Into<Option<GpuAddress>>,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        self.set_root_data(root_or_null(vertex_root), root_or_null(pixel_root));
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd =>
            cmd.draw(vertex_count, instance_count, first_vertex, first_instance))
    }

    /// `gpuDrawIndexedInstanced(cb, vertexDataGpu, pixelDataGpu, indicesGpu, indexCount, instanceCount)`
    pub fn draw_indexed(
        &mut self,
        vertex_root: impl Into<Option<GpuAddress>>,
        pixel_root: impl Into<Option<GpuAddress>>,
        indices: GpuAddress,
        index_count: u32,
        instance_count: u32,
    ) {
        self.set_root_data(root_or_null(vertex_root), root_or_null(pixel_root));
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd =>
            cmd.draw_indexed(indices, index_count, instance_count))
    }

    /// `gpuDispatch(cb, dataGpu, gridDimensions)`
    pub fn dispatch(&mut self, root: impl Into<Option<GpuAddress>>, x: u32, y: u32, z: u32) {
        self.set_compute_root(root_or_null(root));
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.dispatch(x, y, z))
    }

    /// `gpuDispatchIndirect(cb, dataGpu, gridDimensionsGpu)`
    pub fn dispatch_indirect(&mut self, root: impl Into<Option<GpuAddress>>, args: GpuAddress) {
        self.set_compute_root(root_or_null(root));
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.dispatch_indirect(args))
    }

    /// `gpuDrawIndexedInstancedIndirect(cb, vertexDataGpu, pixelDataGpu, indicesGpu, argsGpu)`
    pub fn draw_indexed_indirect(
        &mut self,
        vertex_root: impl Into<Option<GpuAddress>>,
        pixel_root: impl Into<Option<GpuAddress>>,
        indices: GpuAddress,
        args: GpuAddress,
    ) {
        self.set_root_data(root_or_null(vertex_root), root_or_null(pixel_root));
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.draw_indexed_indirect(indices, args))
    }

    /// Multi-draw indirect. `args` is an array of `DrawIndirectMultiArgs`; per-draw root data
    /// is `root + draw_id * stride` (stride 0 broadcasts one block). Non-indexed — the vertex
    /// shader does its own index fetch (see [`DrawIndirectMultiArgs`]).
    pub fn draw_indirect_multi(
        &mut self,
        vertex_root: impl Into<Option<GpuAddress>>,
        vertex_stride: u32,
        pixel_root: impl Into<Option<GpuAddress>>,
        pixel_stride: u32,
        args: GpuAddress,
        draw_count: GpuAddress,
    ) {
        let (vertex_root, pixel_root) = (root_or_null(vertex_root), root_or_null(pixel_root));
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.draw_indirect_multi(
            vertex_root,
            vertex_stride,
            pixel_root,
            pixel_stride,
            args,
            draw_count,
        ))
    }

    // -- Transfer --

    /// Copy bytes between two GPU pointers.
    pub fn memcpy(&mut self, dst: GpuAddress, src: GpuAddress, size: u64) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.memcpy(dst, src, size))
    }

    /// Copy a staging buffer into a texture. `texture_gpu` is the texture's backing
    /// allocation address; `src` is the source buffer address.
    pub fn copy_to_texture(
        &mut self,
        texture_gpu: GpuAddress,
        src: GpuAddress,
        texture: &crate::texture::Texture,
    ) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.copy_to_texture(texture_gpu, src, texture))
    }

    /// Copy a texture into a buffer. `dst` is the destination buffer address; `texture_gpu`
    /// is the texture's backing allocation address.
    pub fn copy_from_texture(
        &mut self,
        dst: GpuAddress,
        texture_gpu: GpuAddress,
        texture: &crate::texture::Texture,
    ) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.copy_from_texture(dst, texture_gpu, texture))
    }

    // -- Barriers --

    /// Stage-only global barrier.
    pub fn barrier(&mut self, src: StageFlags, dst: StageFlags) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.barrier(src, dst))
    }

    /// Stage barrier with hazard flags.
    pub fn barrier_with_hazard(&mut self, src: StageFlags, dst: StageFlags, hazard: HazardFlags) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.barrier_with_hazard(src, dst, hazard))
    }

    /// `gpuSignalAfter(cb, STAGE before, ptrGpu, value, SIGNAL signal)`
    ///
    /// Split-barrier producer: writes `desc.value` to `desc.value_ptr` after `desc.src_stage`
    /// completes, using the specified atomic operation.
    pub fn signal_after(&mut self, desc: &SignalValueDesc) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.signal_after_value(desc))
    }

    /// `gpuWaitBefore(cb, STAGE after, ptrGpu, value, OP op, hazards=0, mask=~0)`
    ///
    /// Split-barrier consumer: stalls `desc.dst_stage` until the value at `desc.value_ptr`
    /// satisfies `desc.wait_op` against `desc.value`, then enforces `desc.hazard` visibility.
    pub fn wait_before(&mut self, desc: &WaitValueDesc) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.wait_before_value(desc))
    }

    // -- Viewport / Scissor --

    /// Set viewport.
    pub fn set_viewport(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd =>
            cmd.set_viewport(x, y, width, height, min_depth, max_depth))
    }

    /// Set scissor rect.
    pub fn set_scissor(&mut self, x: i32, y: i32, width: u32, height: u32) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.set_scissor(x, y, width, height))
    }

    // -- Presentation --

    /// Transition a swapchain image to present-ready layout.
    pub fn transition_to_present(&mut self, _swapchain_image_index: u32) {
        // Underscored because only the Vulkan arm consumes it; Metal-only builds would
        // otherwise see it as unused. (`_`-prefixed params are still referenceable.)
        match &mut self.inner {
            #[cfg(feature = "vulkan")]
            CommandBufferInner::Vulkan(cmd) => cmd.transition_to_present(_swapchain_image_index),
            #[cfg(feature = "metal")]
            CommandBufferInner::Metal(_cmd) => {
                // Metal handles presentation transitions automatically.
            }
        }
    }

    /// Finalize command buffer recording.
    /// On Vulkan, calls vkEndCommandBuffer. On Metal, this is a no-op.
    pub fn end(&mut self) {
        match &mut self.inner {
            #[cfg(feature = "vulkan")]
            CommandBufferInner::Vulkan(cmd) => unsafe {
                cmd.device
                    .end_command_buffer(cmd.command_buffer)
                    .expect("Failed to end command buffer");
            },
            #[cfg(feature = "metal")]
            CommandBufferInner::Metal(cmd) => {
                cmd.end_active_encoders();
            }
        }
    }

    // -- Mesh shader (meshlet) draws --

    /// Draw using the bound mesh-shader pipeline.
    /// `gpuDrawMeshlets(cb, meshletDataGpu, pixelDataGpu, uvec3 dim)`
    ///
    /// Pipeline must be set first via `set_meshlet_pipeline`.
    pub fn draw_meshlets(
        &mut self,
        mesh_root: impl Into<Option<GpuAddress>>,
        pixel_root: impl Into<Option<GpuAddress>>,
        x: u32,
        y: u32,
        z: u32,
    ) {
        self.set_root_data(root_or_null(mesh_root), root_or_null(pixel_root));
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.draw_meshlets(x, y, z))
    }

    /// `gpuDrawMeshletsIndirect(cb, meshletDataGpu, pixelDataGpu, dimGpu)`
    ///
    /// Pipeline must be set first via `set_meshlet_pipeline`.
    /// `args` is a GPU address pointing to one `VkDrawMeshTasksIndirectCommandEXT` (12 bytes: x,y,z).
    pub fn draw_meshlets_indirect(
        &mut self,
        mesh_root: impl Into<Option<GpuAddress>>,
        pixel_root: impl Into<Option<GpuAddress>>,
        args: GpuAddress,
    ) {
        self.set_root_data(root_or_null(mesh_root), root_or_null(pixel_root));
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.draw_meshlets_indirect(args))
    }

    // -- Acceleration structure builds --

    /// Bind a TLAS at `slot` for ray queries (kernels use slot 1: Slang places the trailing
    /// `RaytracingAccelerationStructure` after the root). Call after the pipeline, before dispatch.
    pub fn bind_acceleration_structure(&mut self, slot: u32, accel: &AccelerationStructure) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.bind_acceleration_structure(slot, accel))
    }

    /// Build a BLAS. `accel` must come from `device.create_blas(desc)` with the same `desc`.
    pub fn build_blas(&mut self, accel: &AccelerationStructure, desc: &BlasDesc) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.build_blas(accel, desc))
    }

    /// Build a TLAS.
    pub fn build_tlas(&mut self, accel: &AccelerationStructure, desc: &TlasDesc) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.build_tlas(accel, desc))
    }
}
