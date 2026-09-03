//! Command buffer recording: passes, draws, dispatches, barriers, copies, and AS builds.

use crate::accel::AccelerationStructure;
use crate::barrier::{HazardFlags, StageFlags};
use crate::pipeline::{ComputePso, DepthStencilState, GraphicsPso, MeshletPso};
use crate::query::{QueryPool, QueryPoolInner};
use crate::types::*;
use crate::types::{BlasDesc, TlasDesc};

/// Color attachment for dynamic rendering.
#[derive(Clone, Debug)]
pub struct ColorAttachment {
    /// Swapchain image or offscreen texture.
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
#[derive(Clone, Copy, Debug)]
pub struct RenderTarget(RenderTargetKind);

#[derive(Clone, Copy, Debug)]
pub(crate) enum RenderTargetKind {
    SwapchainImage(u32),
    Texture(TextureId),
}

impl RenderTarget {
    /// Select a swapchain image by index.
    pub const fn swapchain_image(index: u32) -> Self {
        Self(RenderTargetKind::SwapchainImage(index))
    }

    pub(crate) const fn texture(id: TextureId) -> Self {
        Self(RenderTargetKind::Texture(id))
    }

    pub(crate) const fn kind(self) -> RenderTargetKind {
        self.0
    }
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
#[derive(Clone, Debug, Default)]
pub struct RenderPassDesc {
    pub color_attachments: Vec<ColorAttachment>,
    pub depth_attachment: Option<DepthAttachment>,
    pub render_area: [u32; 4], // x, y, width, height
    /// Debug name for the pass, shown against the encoder in a GPU capture. A `&'static str`
    /// rather than the `String` the creation-time descriptors use: this struct is rebuilt every
    /// frame, so the name is expected to be a literal.
    ///
    /// Applied on Metal today. Vulkan needs a device-level `VK_EXT_debug_utils` loader that the
    /// backend does not create yet (it only builds the instance-level messenger), so the name is
    /// ignored there for now.
    pub label: Option<&'static str>,
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

/// Arguments for indirect dispatch (matches VkDispatchIndirectCommand layout).
#[repr(C)]
#[derive(Clone, Copy, Debug, zerocopy::IntoBytes, zerocopy::FromBytes, zerocopy::Immutable)]
pub struct DispatchIndirectArgs {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// Transient command buffer. Created, recorded, submitted, auto-reclaimed.
pub struct CommandBuffer {
    pub(crate) inner: CommandBufferInner,
    pub(crate) _owner: Option<std::rc::Rc<crate::device::DeviceInner>>,
}

pub(crate) enum CommandBufferInner {
    #[cfg(feature = "vulkan")]
    Vulkan(Box<crate::backend::vulkan::command::VulkanCommandBuffer>),
    #[cfg(feature = "metal")]
    Metal(Box<crate::backend::metal::command::MetalCommandBuffer>),
}

impl CommandBuffer {
    fn assert_same_device(
        &self,
        other: &Option<std::rc::Rc<crate::device::DeviceInner>>,
        resource: &str,
    ) {
        let same = self
            ._owner
            .as_ref()
            .zip(other.as_ref())
            .is_some_and(|(command, resource)| std::rc::Rc::ptr_eq(command, resource));
        assert!(same, "{resource} belongs to a different device");
    }

    /// Begin a render pass.
    pub fn begin_render_pass(&mut self, desc: &RenderPassDesc) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.begin_render_pass(desc))
    }

    /// End dynamic rendering.
    pub fn end_render_pass(&mut self) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.end_render_pass())
    }

    /// Set the active graphics pipeline.
    pub fn set_graphics_pipeline(&mut self, pso: &GraphicsPso) {
        self.assert_same_device(&pso._owner, "graphics pipeline");
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.set_graphics_pipeline(pso))
    }

    /// Set the active compute pipeline.
    pub fn set_compute_pipeline(&mut self, pso: &ComputePso) {
        self.assert_same_device(&pso._owner, "compute pipeline");
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.set_compute_pipeline(pso))
    }

    /// Set the active mesh pipeline.
    pub fn set_meshlet_pipeline(&mut self, pso: &MeshletPso) {
        self.assert_same_device(&pso._owner, "meshlet pipeline");
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.set_meshlet_pipeline(pso))
    }

    /// Set depth-stencil state.
    pub fn set_depth_stencil_state(&mut self, state: &DepthStencilState) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.set_depth_stencil_state(state))
    }

    fn set_root_data<T: ?Sized>(&mut self, root: GpuPtr<T>) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.set_root_data(root.cast()))
    }

    fn set_compute_root<T: ?Sized>(&mut self, root: GpuPtr<T>) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.set_compute_root(root.cast()))
    }

    /// Draw non-indexed geometry using `root` as the shared vertex/fragment root.
    pub fn draw<R: ?Sized>(
        &mut self,
        root: GpuPtr<R>,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        self.set_root_data(root);
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd =>
            cmd.draw(vertex_count, instance_count, first_vertex, first_instance))
    }

    /// Draw indexed geometry.
    pub fn draw_indexed<R: ?Sized>(
        &mut self,
        root: GpuPtr<R>,
        indices: GpuPtr<u32>,
        index_count: u32,
        instance_count: u32,
    ) {
        self.set_root_data(root);
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd =>
            cmd.draw_indexed(indices.cast(), index_count, instance_count))
    }

    /// Dispatch compute work.
    pub fn dispatch<R: ?Sized>(&mut self, root: GpuPtr<R>, x: u32, y: u32, z: u32) {
        self.set_compute_root(root);
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.dispatch(x, y, z))
    }

    /// Dispatch compute work from GPU arguments.
    pub fn dispatch_indirect<R: ?Sized>(
        &mut self,
        root: GpuPtr<R>,
        args: GpuPtr<DispatchIndirectArgs>,
    ) {
        self.set_compute_root(root);
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.dispatch_indirect(args.cast()))
    }

    /// Draw indexed geometry from GPU arguments.
    pub fn draw_indexed_indirect<R: ?Sized>(
        &mut self,
        root: GpuPtr<R>,
        indices: GpuPtr<u32>,
        args: GpuPtr<DrawIndexedIndirectArgs>,
    ) {
        self.set_root_data(root);
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.draw_indexed_indirect(indices.cast(), args.cast()))
    }

    /// Copy bytes between two GPU pointers.
    pub fn memcpy<D: ?Sized, S: ?Sized>(&mut self, dst: GpuPtr<D>, src: GpuPtr<S>, size: u64) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.memcpy(dst.cast(), src.cast(), size))
    }

    /// Copy a tightly-packed buffer into the base mip and first layer of a texture.
    pub fn copy_buffer_to_texture<S: ?Sized>(
        &mut self,
        src: GpuPtr<S>,
        texture: &crate::texture::Texture,
    ) {
        self.assert_same_device(&texture._owner, "texture");
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.copy_buffer_to_texture(texture.gpu_address, src.cast(), texture))
    }

    /// Copy the base mip and first layer of a texture into a tightly-packed buffer.
    ///
    /// This is the common-case spelling and derives the placement address from `texture`.
    pub fn copy_texture_to_buffer<D: ?Sized>(
        &mut self,
        texture: &crate::texture::Texture,
        dst: GpuPtr<D>,
    ) {
        self.assert_same_device(&texture._owner, "texture");
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.copy_texture_to_buffer(dst.cast(), texture.gpu_address, texture))
    }

    /// Stage-only global barrier.
    pub fn barrier(&mut self, src: StageFlags, dst: StageFlags) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.barrier(src, dst))
    }

    /// Stage barrier with hazard flags.
    pub fn barrier_with_hazard(&mut self, src: StageFlags, dst: StageFlags, hazard: HazardFlags) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.barrier_with_hazard(src, dst, hazard))
    }

    /// Open a split barrier: record that work up to here in `src` must complete before the
    /// matching [`wait_before`](Self::wait_before). Nothing is encoded until that call, so
    /// unrelated work recorded in between overlaps freely.
    pub fn signal_after(&mut self, src: StageFlags, hazard: HazardFlags) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.signal_after(src, hazard))
    }

    /// Close a split barrier opened by [`signal_after`](Self::signal_after), making `dst` wait on
    /// it. Panics if no split barrier is open.
    pub fn wait_before(&mut self, dst: StageFlags, hazard: HazardFlags) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.wait_before(dst, hazard))
    }

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

    /// Reset a timestamp pool outside a render pass.
    pub fn reset_queries(&mut self, pool: &QueryPool) {
        self.assert_same_device(&pool._owner, "query pool");
        match (&mut self.inner, &pool.inner) {
            #[cfg(feature = "vulkan")]
            (CommandBufferInner::Vulkan(cmd), QueryPoolInner::Vulkan(p)) => {
                cmd.reset_queries(p.pool, 0, pool.count)
            }
            #[cfg(feature = "metal")]
            (CommandBufferInner::Metal(_), QueryPoolInner::Metal(_)) => {}
            #[allow(unreachable_patterns)]
            _ => unreachable!("query pool backend does not match command buffer backend"),
        }
    }

    /// Write a timestamp outside a render pass.
    pub fn write_timestamp(&mut self, pool: &QueryPool, query: u32) {
        self.assert_same_device(&pool._owner, "query pool");
        match (&mut self.inner, &pool.inner) {
            #[cfg(feature = "vulkan")]
            (CommandBufferInner::Vulkan(cmd), QueryPoolInner::Vulkan(p)) => {
                cmd.write_timestamp(p.pool, query)
            }
            #[cfg(feature = "metal")]
            (CommandBufferInner::Metal(cmd), QueryPoolInner::Metal(p)) => {
                cmd.write_timestamp(&p.heap, query as usize)
            }
            #[allow(unreachable_patterns)]
            _ => unreachable!("query pool backend does not match command buffer backend"),
        }
    }

    /// Transition a swapchain image to present-ready layout.
    pub fn transition_to_present(&mut self, _swapchain_image_index: u32) {
        match &mut self.inner {
            #[cfg(feature = "vulkan")]
            CommandBufferInner::Vulkan(cmd) => cmd.transition_to_present(_swapchain_image_index),
            #[cfg(feature = "metal")]
            CommandBufferInner::Metal(_cmd) => {
                // Metal handles presentation transitions automatically.
            }
        }
    }

    /// Finalize command recording early. This is idempotent and optional because queue submission
    /// finalizes command buffers on both backends.
    pub fn end(&mut self) {
        match &mut self.inner {
            #[cfg(feature = "vulkan")]
            CommandBufferInner::Vulkan(cmd) => {
                cmd.finish().expect("Failed to end command buffer");
            }
            #[cfg(feature = "metal")]
            CommandBufferInner::Metal(cmd) => {
                cmd.finish();
            }
        }
    }

    /// Draw mesh tasks using the active mesh pipeline.
    pub fn draw_meshlets<R: ?Sized>(&mut self, root: GpuPtr<R>, x: u32, y: u32, z: u32) {
        self.set_root_data(root);
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.draw_meshlets(x, y, z))
    }

    /// Draw mesh tasks from GPU arguments.
    pub fn draw_meshlets_indirect<R: ?Sized>(
        &mut self,
        root: GpuPtr<R>,
        args: GpuPtr<DispatchIndirectArgs>,
    ) {
        self.set_root_data(root);
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.draw_meshlets_indirect(args.cast()))
    }

    /// Build a BLAS. `accel` must come from `device.create_blas(desc)` with the same `desc`.
    pub fn build_blas(&mut self, accel: &AccelerationStructure, desc: &BlasDesc) {
        self.assert_same_device(&accel._owner, "acceleration structure");
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.build_blas(accel, desc))
    }

    /// Build a TLAS.
    pub fn build_tlas(&mut self, accel: &AccelerationStructure, desc: &TlasDesc) {
        self.assert_same_device(&accel._owner, "acceleration structure");
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.build_tlas(accel, desc))
    }
}
