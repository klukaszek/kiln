//! Command buffer recording: passes, draws, dispatches, barriers, copies, and AS builds.

use crate::accel::AccelerationStructure;
use crate::barrier::{HazardFlags, StageFlags};
use crate::pipeline::{ComputePso, GraphicsPso, MeshletPso};
use crate::query::{QueryPool, QueryPoolInner};
use crate::texture::{ResolvedRegion, Texture, TextureRegion};
use crate::types::{BlasDesc, GpuPtr, TextureId, TlasDesc};

/// Fill in a caller's optional region against the texture it addresses.
fn resolve_region(region: impl Into<Option<TextureRegion>>, texture: &Texture) -> ResolvedRegion {
    region.into().unwrap_or_default().resolve(texture.desc())
}

/// Color attachment for dynamic rendering.
#[derive(Clone, Copy, Debug)]
pub struct ColorAttachment {
    pub target: RenderTarget,
    pub load_op: LoadOp,
    pub store_op: StoreOp,
    pub clear_color: [f32; 4],
}

/// Depth attachment for dynamic rendering.
#[derive(Clone, Copy, Debug)]
pub struct DepthAttachment {
    pub target: RenderTarget,
    pub load_op: LoadOp,
    pub store_op: StoreOp,
    pub clear_depth: f32,
}

/// Render target reference.
#[derive(Clone, Copy, Debug)]
pub struct RenderTarget(RenderTargetKind);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
///
/// Borrowed rather than owned: a render pass is described once per pass per frame, so the
/// attachments come straight off the caller's stack with no allocation.
#[derive(Clone, Copy, Debug, Default)]
pub struct RenderPassDesc<'a> {
    pub color_attachments: &'a [ColorAttachment],
    pub depth_attachment: Option<DepthAttachment>,
    pub render_area: [u32; 4], // x, y, width, height
    /// Pass name in a GPU capture. `&'static str` because this struct is rebuilt every frame.
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

/// A pipeline state object bindable with [`CommandBuffer::set_pipeline`].
pub trait Pipeline: crate::sealed::Sealed {
    #[doc(hidden)]
    fn bind_to(&self, cmd: &mut CommandBuffer);
}

macro_rules! impl_pipeline {
    ($($ty:ty => $bind:ident),+ $(,)?) => {
        $(
            impl Pipeline for $ty {
                fn bind_to(&self, cmd: &mut CommandBuffer) {
                    backend_dispatch!(&mut cmd.inner, CommandBufferInner, c => c.$bind(self))
                }
            }
        )+
    };
}

impl_pipeline!(
    GraphicsPso => set_graphics_pipeline,
    ComputePso => set_compute_pipeline,
    MeshletPso => set_meshlet_pipeline,
);

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
    pub fn begin_render_pass(&mut self, desc: &RenderPassDesc<'_>) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.begin_render_pass(desc))
    }

    pub fn end_render_pass(&mut self) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.end_render_pass())
    }

    /// Bind a pipeline. Its type selects which draw or dispatch calls are then valid.
    pub fn set_pipeline<P: Pipeline>(&mut self, pso: &P) {
        pso.bind_to(self);
    }

    fn set_root_data<T: ?Sized>(&mut self, root: GpuPtr<T>) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.set_root_data(root.cast()))
    }

    /// `root` is shared by the vertex and pixel stages.
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

    pub fn dispatch<R: ?Sized>(&mut self, root: GpuPtr<R>, x: u32, y: u32, z: u32) {
        self.set_root_data(root);
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.dispatch(x, y, z))
    }

    /// Indirect dispatch.
    pub fn dispatch_indirect<R: ?Sized>(
        &mut self,
        root: GpuPtr<R>,
        args: GpuPtr<DispatchIndirectArgs>,
    ) {
        self.set_root_data(root);
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.dispatch_indirect(args.cast()))
    }

    /// Indirect draw.
    pub fn draw_indirect<R: ?Sized>(&mut self, root: GpuPtr<R>, args: GpuPtr<DrawIndirectArgs>) {
        self.set_root_data(root);
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.draw_indirect(args.cast()))
    }

    /// Indirect indexed draw.
    /// `max_index_count` bounds the index range the GPU may read. The draw count itself comes
    /// from `args`; this is the upper bound Metal needs to size the index buffer binding, and the
    /// caller already knows it — deriving it instead would mean an address lookup per draw.
    pub fn draw_indexed_indirect<R: ?Sized>(
        &mut self,
        root: GpuPtr<R>,
        indices: GpuPtr<u32>,
        max_index_count: u32,
        args: GpuPtr<DrawIndexedIndirectArgs>,
    ) {
        self.set_root_data(root);
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.draw_indexed_indirect(indices.cast(), max_index_count, args.cast()))
    }

    pub fn memcpy<D: ?Sized, S: ?Sized>(&mut self, dst: GpuPtr<D>, src: GpuPtr<S>, size: u64) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.memcpy(dst.cast(), src.cast(), size))
    }

    /// Upload tightly packed texels into one mip/layer of `texture`.
    ///
    /// `region` is `None` for the whole of mip 0, layer 0, or a [`TextureRegion`] to target a
    /// specific subresource or sub-box. The source layout is tightly packed to the region's
    /// extent: `region.extent[0] * bytes_per_pixel` per row, no padding between rows or slices.
    pub fn copy_buffer_to_texture<S: ?Sized>(
        &mut self,
        src: GpuPtr<S>,
        texture: &Texture,
        region: impl Into<Option<TextureRegion>>,
    ) {
        let region = resolve_region(region, texture);
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.copy_buffer_to_texture(src.cast(), texture, region))
    }

    /// Read one mip/layer of `texture` back into a tightly packed buffer. See
    /// [`copy_buffer_to_texture`](Self::copy_buffer_to_texture) for the `region` and layout rules.
    pub fn copy_texture_to_buffer<D: ?Sized>(
        &mut self,
        texture: &Texture,
        dst: GpuPtr<D>,
        region: impl Into<Option<TextureRegion>>,
    ) {
        let region = resolve_region(region, texture);
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.copy_texture_to_buffer(dst.cast(), texture, region))
    }

    /// Copy one mip/layer of `src` into one mip/layer of `dst`. The copied extent comes from
    /// `src_region`; `dst_region`'s extent is ignored, only its mip, layer and origin apply.
    pub fn copy_texture_to_texture(
        &mut self,
        src: &Texture,
        src_region: impl Into<Option<TextureRegion>>,
        dst: &Texture,
        dst_region: impl Into<Option<TextureRegion>>,
    ) {
        let src_region = resolve_region(src_region, src);
        let dst_region = resolve_region(dst_region, dst);
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.copy_texture_to_texture(src, src_region, dst, dst_region))
    }

    pub fn barrier(&mut self, src: StageFlags, dst: StageFlags) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.barrier(src, dst))
    }

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

    pub fn set_scissor(&mut self, x: i32, y: i32, width: u32, height: u32) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.set_scissor(x, y, width, height))
    }

    /// Must be outside a render pass, after the previous GPU use of this pool has completed.
    /// On Metal the reset happens immediately on the CPU, rather than at submission.
    pub fn reset_queries(&mut self, pool: &QueryPool) {
        match (&mut self.inner, &pool.inner) {
            #[cfg(feature = "vulkan")]
            (CommandBufferInner::Vulkan(cmd), QueryPoolInner::Vulkan(p)) => {
                cmd.reset_queries(p.pool, 0, pool.count)
            }
            #[cfg(feature = "metal")]
            (CommandBufferInner::Metal(_), QueryPoolInner::Metal(p)) => {
                use objc2_metal::MTL4CounterHeap;
                // Metal invalidation runs immediately on the CPU. The caller must
                // have waited for the previous use of this pool before resetting it.
                unsafe {
                    p.heap.invalidateCounterRange(objc2_foundation::NSRange {
                        location: 0,
                        length: pool.count as usize,
                    });
                }
            }
            #[allow(unreachable_patterns)]
            _ => unreachable!("query pool backend does not match command buffer backend"),
        }
    }

    /// Must be outside a render pass.
    pub fn write_timestamp(&mut self, pool: &QueryPool, query: u32) {
        assert!(query < pool.count, "timestamp query index out of range");
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

    pub fn draw_meshlets<R: ?Sized>(&mut self, root: GpuPtr<R>, x: u32, y: u32, z: u32) {
        self.set_root_data(root);
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.draw_meshlets(x, y, z))
    }

    /// Indirect mesh draw.
    pub fn draw_meshlets_indirect<R: ?Sized>(
        &mut self,
        root: GpuPtr<R>,
        args: GpuPtr<DispatchIndirectArgs>,
    ) {
        self.set_root_data(root);
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.draw_meshlets_indirect(args.cast()))
    }

    /// `accel` must come from `device.create_blas` with this same `desc`.
    pub fn build_blas(&mut self, accel: &AccelerationStructure, desc: &BlasDesc) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.build_blas(accel, desc))
    }

    pub fn build_tlas(&mut self, accel: &AccelerationStructure, desc: &TlasDesc) {
        backend_dispatch!(&mut self.inner, CommandBufferInner, cmd => cmd.build_tlas(accel, desc))
    }
}
