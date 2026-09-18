use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer,
    MTL4CommandEncoder, MTL4ComputeCommandEncoder, MTL4RenderCommandEncoder,
    MTL4RenderPassDescriptor, MTL4VisibilityOptions, MTLBuffer, MTLDevice, MTLGPUAddress,
    MTLIndexType, MTLLoadAction, MTLOrigin, MTLPrimitiveType, MTLResidencySet, MTLResourceOptions,
    MTLScissorRect, MTLSize, MTLStages, MTLStoreAction, MTLTexture, MTLViewport,
};
use smallvec::SmallVec;
use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::barrier::{HazardFlags, StageFlags};
use crate::command::{LoadOp, RenderPassDesc, RenderTarget, RenderTargetKind, StoreOp};
use crate::pipeline::{ComputePso, GraphicsPso, MeshletPso};
use crate::texture::{ResolvedRegion, Texture, bytes_per_pixel};
use crate::types::{BlasDesc, GpuPtr, TextureId, TlasDesc};

use super::accel::downcast_base;
use super::as_allocation;
use super::barrier::{ALL_RENDER_STAGES, to_mtl_stages, visibility_from_hazard};
use super::device::MetalDevice;
use super::device::MetalShared;
use super::swapchain::MetalSwapchain;
use super::swapchain::SharedDrawableSlot;
use crate::command::CommandBuffer;
use crate::error::{RhiError, RhiResult};

// Metal root-table entries are 16 bytes.
fn to_mtl_origin(origin: [u32; 3]) -> MTLOrigin {
    MTLOrigin {
        x: origin[0] as usize,
        y: origin[1] as usize,
        z: origin[2] as usize,
    }
}

/// Metal argument-table entries are 16 bytes.
const ROOT_TABLE_SLOT_BYTES: usize = 16;

/// Threads per object/mesh threadgroup for a mesh draw. Fixed: no amplification stage is exposed,
/// so the object group is one thread and the mesh group one SIMD width.
const MESH_THREADS_PER_OBJECT_GROUP: MTLSize = MTLSize {
    width: 1,
    height: 1,
    depth: 1,
};
const MESH_THREADS_PER_MESH_GROUP: MTLSize = MTLSize {
    width: 32,
    height: 1,
    depth: 1,
};

/// Bump arena over the command buffer's root-pointer ring. `cpu`/`gpu` are cached because
/// `contents()`/`gpuAddress()` are message sends and `set_root_data` runs once per draw.
struct RootArena {
    cpu: *mut u8,
    gpu: MTLGPUAddress,
    cursor: usize,
}

impl RootArena {
    fn new(buffer: &ProtocolObject<dyn MTLBuffer>) -> Self {
        Self {
            cpu: buffer.contents().as_ptr().cast::<u8>(),
            gpu: buffer.gpuAddress(),
            cursor: 0,
        }
    }

    /// Carve `size` bytes, rounded up to a whole argument-table slot.
    fn alloc(&mut self, size: usize) -> (MTLGPUAddress, *mut u8) {
        let size = size.next_multiple_of(ROOT_TABLE_SLOT_BYTES);
        let offset = self.cursor;
        let end = offset + size;
        assert!(
            end <= ROOT_TABLE_ARENA_BYTES,
            "this command buffer has recorded {ROOT_TABLE_SLOTS} draws or dispatches, the most \
             one can hold on Metal. Split the work across several command buffers."
        );
        self.cursor = end;
        // SAFETY: `end <= ROOT_TABLE_ARENA_BYTES`, the length the buffer was allocated with.
        (self.gpu + offset as u64, unsafe { self.cpu.add(offset) })
    }
}

/// Root-pointer slots one command buffer can hand out, and so the maximum number of draws,
/// dispatches and mesh draws it can record: each one publishes its root pointer through a fresh
/// slot.
///
/// This is an arena, not a ring — it only ever moves forward, because a slot's address is live
/// until the command buffer retires. Recording past it is a panic rather than a wrap, which would
/// silently repoint an earlier draw's root data.
pub(crate) const ROOT_TABLE_SLOTS: usize = 65_536;
const ROOT_TABLE_ARENA_BYTES: usize = ROOT_TABLE_SLOT_BYTES * ROOT_TABLE_SLOTS;

/// Everything one buffer↔texture copy needs, resolved from a [`ResolvedRegion`].
struct TextureCopy {
    texture: Retained<ProtocolObject<dyn MTLTexture>>,
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Byte offset of the linear data within `buffer`.
    offset: u64,
    size: MTLSize,
    origin: MTLOrigin,
    bytes_per_row: usize,
    bytes_per_image: usize,
}

#[derive(Clone, Copy)]
struct PendingQueueBarrier {
    after_queue_stages: MTLStages,
    before_stages: MTLStages,
    visibility: MTL4VisibilityOptions,
}

fn add_buffer_to_residency(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    residency_set: &ProtocolObject<dyn MTLResidencySet>,
    residency_dirty: &Cell<bool>,
) {
    residency_set.addAllocation(as_allocation(buffer));
    residency_dirty.set(true);
}

pub(crate) fn create_root_table_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    residency_set: &ProtocolObject<dyn MTLResidencySet>,
    residency_dirty: &Cell<bool>,
) -> crate::error::RhiResult<Retained<ProtocolObject<dyn MTLBuffer>>> {
    let root_table_buffer = device
        .newBufferWithLength_options(
            ROOT_TABLE_ARENA_BYTES,
            MTLResourceOptions::StorageModeShared,
        )
        .ok_or_else(|| {
            crate::error::RhiError::CommandBuffer(
                "Failed to allocate Metal root table arena".into(),
            )
        })?;
    add_buffer_to_residency(root_table_buffer.as_ref(), residency_set, residency_dirty);
    Ok(root_table_buffer)
}

/// One command buffer's worth of reusable native state, recycled rather than rebuilt: the root
/// arena alone is a 1 MiB allocation that has to join the residency set. Only handed out again
/// once its fence has retired.
pub(crate) struct MetalFrameTableSlot {
    pub(crate) root_table_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) command_allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    pub(crate) command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    pub(crate) argument_table: RefCell<Option<Retained<ProtocolObject<dyn MTL4ArgumentTable>>>>,
    pub(crate) in_use: Cell<bool>,
    /// Free list to return to on drop. `None` for swapchain slots, which live in `frame_table_slots`.
    pub(crate) pool: Option<SharedTableSlotPool>,
}

pub(crate) type SharedFrameTableSlots =
    Rc<RefCell<[Option<Rc<MetalFrameTableSlot>>; crate::types::MAX_FRAMES_IN_FLIGHT]>>;
/// Free list of generic (non-swapchain) table slots.
pub(crate) type SharedTableSlotPool = Rc<RefCell<Vec<Rc<MetalFrameTableSlot>>>>;

pub struct MetalCommandBuffer {
    pub(crate) command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    pub(crate) render_encoder: Option<Retained<ProtocolObject<dyn MTL4RenderCommandEncoder>>>,
    pub(crate) compute_encoder: Option<Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>>,
    /// Set when this command buffer renders to the swapchain; the drawable is taken on first use.
    pub(crate) drawable_slot: Option<SharedDrawableSlot>,
    current_topology: MTLPrimitiveType,
    pub(crate) shared: Rc<MetalShared>,
    argument_table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    frame_table_slot: Rc<MetalFrameTableSlot>,
    root_arena: RootArena,
    current_threads_per_threadgroup: [u32; 3],
    pending_queue_barrier: Option<PendingQueueBarrier>,
    /// Render targets already written by an earlier pass in this command buffer. Metal 4 tracks no
    /// hazards of its own, so a second pass on the same attachment needs an explicit dependency.
    written_targets: SmallVec<[RenderTargetKind; 8]>,
    ended: bool,
}

impl MetalCommandBuffer {
    fn create_argument_table(
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> crate::error::RhiResult<Retained<ProtocolObject<dyn MTL4ArgumentTable>>> {
        let desc = MTL4ArgumentTableDescriptor::new();
        desc.setMaxBufferBindCount(6);
        desc.setMaxTextureBindCount(0);
        desc.setMaxSamplerStateBindCount(0);
        desc.setInitializeBindings(true);
        desc.setSupportAttributeStrides(false);

        device
            .newArgumentTableWithDescriptor_error(&desc)
            .map_err(|e| {
                crate::error::RhiError::CommandBuffer(
                    format!("Failed to create Metal 4 argument table: {e}").into(),
                )
            })
    }

    fn resolve_buffer(
        &self,
        addr: GpuPtr<u8>,
        size: u64,
    ) -> (Retained<ProtocolObject<dyn MTLBuffer>>, u64) {
        let addr_u64 = addr.address;
        let allocations = self.shared.allocations.borrow();
        if let Some((&base, alloc)) = allocations.range(..=addr_u64).next_back() {
            let offset = addr_u64 - base;
            if offset < alloc.size && size <= alloc.size - offset {
                return (alloc.buffer.clone(), offset);
            }
        }
        panic!("GPU address {addr_u64:#x} not found in allocation registry");
    }

    fn resolve_texture(&self, id: TextureId) -> Retained<ProtocolObject<dyn MTLTexture>> {
        self.shared
            .textures
            .with(id.0, |t| t.texture.clone())
            .expect("texture id has no live slot")
    }

    /// Reuses the open compute encoder so a run of copies shares one. Consecutive copies are
    /// unordered either way — Metal 4 orders nothing implicitly.
    fn begin_copy_encoder(
        &mut self,
        label: &str,
    ) -> Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>> {
        if let Some(encoder) = self.compute_encoder.take() {
            return encoder;
        }
        if let Some(encoder) = self.render_encoder.take() {
            encoder.endEncoding();
        }
        self.begin_compute_encoder(label)
    }

    /// Open a named compute encoder and flush any pending queue barrier into it.
    fn begin_compute_encoder(
        &mut self,
        label: &str,
    ) -> Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>> {
        let encoder = self
            .command_buffer
            .computeCommandEncoder()
            .expect("Failed to create Metal compute command encoder");
        encoder.setLabel(Some(&NSString::from_str(label)));
        // SAFETY: MTL4ComputeCommandEncoder refines MTL4CommandEncoder.
        self.apply_pending_queue_barrier(unsafe { super::cast_protocol(&*encoder) });
        encoder
    }

    pub(crate) fn new(
        command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
        shared: Rc<MetalShared>,
        frame_table_slot: Rc<MetalFrameTableSlot>,
    ) -> crate::error::RhiResult<Self> {
        command_buffer.beginCommandBufferWithAllocator(&frame_table_slot.command_allocator);
        // The queue owns the residency set for every submitted command buffer; attaching it here
        // too would repeat a setup call per frame without changing residency.

        let argument_table =
            if let Some(argument_table) = frame_table_slot.argument_table.borrow_mut().take() {
                argument_table
            } else {
                Self::create_argument_table(shared.device.as_ref())?
            };

        let root_arena = RootArena::new(&frame_table_slot.root_table_buffer);

        unsafe {
            argument_table.setAddress_atIndex(shared.textures.heap().gpuAddress(), 1);
            argument_table.setAddress_atIndex(shared.samplers.heap().gpuAddress(), 2);
            // Recycled tables would otherwise carry the previous frame's root pointer.
            argument_table.setAddress_atIndex(0, 0);
        }

        Ok(Self {
            command_buffer,
            render_encoder: None,
            compute_encoder: None,
            drawable_slot: None,
            current_topology: MTLPrimitiveType::Triangle,
            shared,
            argument_table,
            frame_table_slot,
            root_arena,
            current_threads_per_threadgroup: [1, 1, 1],
            pending_queue_barrier: None,
            written_targets: SmallVec::new(),
            ended: false,
        })
    }

    pub fn begin_render_pass(&mut self, desc: &RenderPassDesc<'_>) {
        self.end_active_encoders();
        self.order_after_previous_writes(desc);
        let pass_desc = MTL4RenderPassDescriptor::new();

        // Metal constrains from the origin and has no pass-level offset, so the area's origin
        // rides on the scissor. Zero keeps Metal's "use the attachment's size" default.
        let [area_x, area_y, area_w, area_h] = desc.render_area;
        if area_w != 0 && area_h != 0 {
            pass_desc.setRenderTargetWidth((area_x + area_w) as usize);
            pass_desc.setRenderTargetHeight((area_y + area_h) as usize);
        }

        let color_attachments = pass_desc.colorAttachments();
        for (i, color_att) in desc.color_attachments.iter().enumerate() {
            let attachment = unsafe { color_attachments.objectAtIndexedSubscript(i) };

            match color_att.target.kind() {
                RenderTargetKind::SwapchainImage(_idx) => {
                    if let Some(tex) = self.drawable_slot.as_ref().and_then(|slot| slot.texture()) {
                        attachment.setTexture(Some(&tex));
                    }
                }
                RenderTargetKind::Texture(id) => {
                    let tex = self.resolve_texture(id);
                    attachment.setTexture(Some(&tex));
                }
            }

            attachment.setLoadAction(match color_att.load_op {
                LoadOp::Clear => MTLLoadAction::Clear,
                LoadOp::Load => MTLLoadAction::Load,
                LoadOp::DontCare => MTLLoadAction::DontCare,
            });

            attachment.setStoreAction(match color_att.store_op {
                StoreOp::Store => MTLStoreAction::Store,
                StoreOp::DontCare => MTLStoreAction::DontCare,
            });

            if color_att.load_op == LoadOp::Clear {
                let c = color_att.clear_color;
                attachment.setClearColor(objc2_metal::MTLClearColor {
                    red: c[0] as f64,
                    green: c[1] as f64,
                    blue: c[2] as f64,
                    alpha: c[3] as f64,
                });
            }
        }

        if let Some(depth_att) = &desc.depth_attachment {
            let depth = pass_desc.depthAttachment();
            let texture = match depth_att.target.kind() {
                RenderTargetKind::Texture(id) => self.resolve_texture(id),
                // The swapchain has colour images only; depth comes from a texture you own.
                RenderTargetKind::SwapchainImage(_) => {
                    panic!("a swapchain image cannot be a depth attachment")
                }
            };
            depth.setTexture(Some(&texture));

            depth.setLoadAction(match depth_att.load_op {
                LoadOp::Clear => MTLLoadAction::Clear,
                LoadOp::Load => MTLLoadAction::Load,
                LoadOp::DontCare => MTLLoadAction::DontCare,
            });

            depth.setStoreAction(match depth_att.store_op {
                StoreOp::Store => MTLStoreAction::Store,
                StoreOp::DontCare => MTLStoreAction::DontCare,
            });

            if depth_att.load_op == LoadOp::Clear {
                depth.setClearDepth(depth_att.clear_depth as f64);
            }
        }

        let encoder = self
            .command_buffer
            .renderCommandEncoderWithDescriptor(&pass_desc)
            .expect("Failed to create Metal render command encoder");
        encoder.setLabel(Some(&NSString::from_str(desc.label.unwrap_or("render"))));

        encoder.setArgumentTable_atStages(&self.argument_table, ALL_RENDER_STAGES);

        // SAFETY: MTL4RenderCommandEncoder refines MTL4CommandEncoder.
        self.apply_pending_queue_barrier(unsafe { super::cast_protocol(&*encoder) });

        self.render_encoder = Some(encoder);
    }

    pub fn end_render_pass(&mut self) {
        if let Some(encoder) = self.render_encoder.take() {
            encoder.endEncoding();
        }
    }

    /// Order a pass against earlier passes in this command buffer that wrote the same attachment.
    /// Metal 4 tracks no hazards, so two render encoders on one texture are free to overlap;
    /// Vulkan gets this ordering from the attachment's layout transition instead.
    ///
    /// Merged into `pending_queue_barrier` so it lands on the new encoder alongside whatever the
    /// caller already asked for.
    fn order_after_previous_writes(&mut self, desc: &RenderPassDesc) {
        let targets = desc
            .color_attachments
            .iter()
            .map(|attachment| attachment.target)
            .chain(desc.depth_attachment.as_ref().map(|depth| depth.target))
            .map(RenderTarget::kind);

        let mut hazard = false;
        for target in targets {
            if self.written_targets.contains(&target) {
                hazard = true;
            } else {
                self.written_targets.push(target);
            }
        }
        if !hazard {
            return;
        }

        // Attachment writes drain through the fragment stage, and `Device` visibility is what
        // makes the stored pixels readable by the next pass's load.
        let stages = to_mtl_stages(StageFlags::RASTER_COLOR_OUT | StageFlags::RASTER_DEPTH_OUT);
        self.enqueue_queue_barrier(stages, stages, MTL4VisibilityOptions::Device);
    }

    pub fn set_graphics_pipeline(&mut self, pso: &GraphicsPso) {
        let mtl_pso = &pso.inner;
        let (pipeline, depth_stencil, depth_bias, cull_mode, winding, topology) = (
            &mtl_pso.pipeline,
            &mtl_pso.depth_stencil,
            mtl_pso.depth_bias,
            mtl_pso.cull_mode,
            mtl_pso.winding,
            mtl_pso.topology,
        );
        self.current_topology = topology;
        let encoder = self
            .render_encoder
            .as_ref()
            .expect("No active render encoder");
        encoder.setRenderPipelineState(pipeline);
        encoder.setCullMode(cull_mode);
        encoder.setFrontFacingWinding(winding);
        encoder.setDepthStencilState(Some(depth_stencil));
        encoder.setDepthBias_slopeScale_clamp(depth_bias.0, depth_bias.1, depth_bias.2);
    }

    pub fn set_compute_pipeline(&mut self, pso: &ComputePso) {
        let mtl_pso = &pso.inner;

        // Pipeline binds do not delimit compute passes. Keep the encoder alive so
        // encoder-scoped barriers continue to order dispatches across PSO changes.
        let encoder = self.compute_encoder.take().unwrap_or_else(|| {
            self.end_active_encoders();
            self.begin_compute_encoder(mtl_pso.label.as_deref().unwrap_or("compute"))
        });
        encoder.setComputePipelineState(&mtl_pso.pipeline);

        self.current_threads_per_threadgroup = mtl_pso.threads_per_threadgroup;
        encoder.setArgumentTable(Some(&self.argument_table));

        self.compute_encoder = Some(encoder);
    }

    pub fn set_root_data(&mut self, root: GpuPtr<u8>) {
        let (slot_addr, slot_ptr) = self.root_arena.alloc(std::mem::size_of::<u64>());
        unsafe {
            std::ptr::write_unaligned(slot_ptr.cast::<u64>(), root.address);
            self.argument_table.setAddress_atIndex(slot_addr, 0);
        }
    }

    pub fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        let topology = self.current_topology;
        let encoder = self
            .render_encoder
            .as_ref()
            .expect("No active render encoder");
        unsafe {
            encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                topology,
                first_vertex as usize,
                vertex_count as usize,
                instance_count as usize,
                first_instance as usize,
            );
        }
    }

    pub fn draw_indexed(
        &mut self,
        indices: GpuPtr<u8>,
        index_count: u32,
        instance_count: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        // Indices are U32 and Metal 4 consumes the buffer as a raw GPU address.
        let index_addr_gpu: MTLGPUAddress = indices.address;
        let index_len = (index_count as u64) * 4;
        let topology = self.current_topology;
        let encoder = self
            .render_encoder
            .as_ref()
            .expect("No active render encoder");
        unsafe {
            encoder
                .drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferLength_instanceCount_baseVertex_baseInstance(
                    topology,
                    index_count as usize,
                    MTLIndexType::UInt32,
                    index_addr_gpu,
                    index_len as usize,
                    instance_count as usize,
                    vertex_offset as isize,
                    first_instance as usize,
                );
        }
    }

    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        let encoder = self
            .compute_encoder
            .as_ref()
            .expect("No active compute encoder");
        let threads_per_group = self.current_threads_per_threadgroup;
        let tg = MTLSize {
            width: threads_per_group[0] as usize,
            height: threads_per_group[1] as usize,
            depth: threads_per_group[2] as usize,
        };
        let groups = MTLSize {
            width: x as usize,
            height: y as usize,
            depth: z as usize,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, tg);
    }

    pub fn dispatch_indirect(&mut self, args: GpuPtr<u8>) {
        let encoder = self
            .compute_encoder
            .as_ref()
            .expect("No active compute encoder");
        let arg_addr_gpu: MTLGPUAddress = args.address;
        let threads_per_group = self.current_threads_per_threadgroup;
        let tg = MTLSize {
            width: threads_per_group[0] as usize,
            height: threads_per_group[1] as usize,
            depth: threads_per_group[2] as usize,
        };
        unsafe {
            encoder.dispatchThreadgroupsWithIndirectBuffer_threadsPerThreadgroup(arg_addr_gpu, tg);
        }
    }

    pub fn draw_indirect(&mut self, args: GpuPtr<u8>) {
        let arg_addr_gpu: MTLGPUAddress = args.address;
        let topology = self.current_topology;
        let encoder = self
            .render_encoder
            .as_ref()
            .expect("No active render encoder");
        encoder.drawPrimitives_indirectBuffer(topology, arg_addr_gpu);
    }

    pub fn draw_indexed_indirect(
        &mut self,
        indices: GpuPtr<u8>,
        max_index_count: u32,
        args: GpuPtr<u8>,
    ) {
        let index_addr_gpu: MTLGPUAddress = indices.address;
        let index_len = (max_index_count as u64) * 4;
        let arg_addr_gpu: MTLGPUAddress = args.address;
        let topology = self.current_topology;
        let encoder = self
            .render_encoder
            .as_ref()
            .expect("No active render encoder");
        unsafe {
            encoder.drawIndexedPrimitives_indexType_indexBuffer_indexBufferLength_indirectBuffer(
                topology,
                MTLIndexType::UInt32,
                index_addr_gpu,
                index_len as usize,
                arg_addr_gpu,
            );
        }
    }

    pub fn memcpy(&mut self, dst: GpuPtr<u8>, src: GpuPtr<u8>, size: u64) {
        if size == 0 {
            return;
        }
        let (src_buffer, src_offset) = self.resolve_buffer(src, size);
        let (dst_buffer, dst_offset) = self.resolve_buffer(dst, size);

        // Metal 4 routes buffer copies through the compute encoder.
        let encoder = self.begin_copy_encoder("memcpy");
        unsafe {
            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &src_buffer,
                src_offset as usize,
                &dst_buffer,
                dst_offset as usize,
                size as usize,
            );
        }
        self.compute_encoder = Some(encoder);
    }

    pub fn copy_buffer_to_texture(
        &mut self,
        src: GpuPtr<u8>,
        texture: &Texture,
        region: ResolvedRegion,
    ) {
        let copy = self.prepare_texture_copy(src, texture, region, "copy_buffer_to_texture");

        let encoder = self.begin_copy_encoder("copy to texture");
        unsafe {
            encoder.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                &copy.buffer,
                copy.offset as usize,
                copy.bytes_per_row,
                copy.bytes_per_image,
                copy.size,
                &copy.texture,
                region.layer as usize,
                region.mip as usize,
                copy.origin,
            );
        }
        self.compute_encoder = Some(encoder);
    }

    pub fn copy_texture_to_buffer(
        &mut self,
        dst: GpuPtr<u8>,
        texture: &Texture,
        region: ResolvedRegion,
    ) {
        let copy = self.prepare_texture_copy(dst, texture, region, "copy_texture_to_buffer");

        let encoder = self.begin_copy_encoder("copy from texture");
        unsafe {
            encoder.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                &copy.texture,
                region.layer as usize,
                region.mip as usize,
                copy.origin,
                copy.size,
                &copy.buffer,
                copy.offset as usize,
                copy.bytes_per_row,
                copy.bytes_per_image,
            );
        }
        self.compute_encoder = Some(encoder);
    }

    pub fn copy_texture_to_texture(
        &mut self,
        src: &Texture,
        src_region: ResolvedRegion,
        dst: &Texture,
        dst_region: ResolvedRegion,
    ) {
        let src_texture = self.resolve_texture(src.id());
        let dst_texture = self.resolve_texture(dst.id());
        let size = MTLSize {
            width: src_region.extent[0] as usize,
            height: src_region.extent[1] as usize,
            depth: src_region.extent[2] as usize,
        };

        let encoder = self.begin_copy_encoder("copy texture to texture");
        unsafe {
            encoder.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                &src_texture,
                src_region.layer as usize,
                src_region.mip as usize,
                to_mtl_origin(src_region.origin),
                size,
                &dst_texture,
                dst_region.layer as usize,
                dst_region.mip as usize,
                to_mtl_origin(dst_region.origin),
            );
        }
        self.compute_encoder = Some(encoder);
    }

    /// Resolve the texture and linear buffer, and compute the row / image strides for a
    /// buffer↔texture copy on the Metal 4 compute encoder.
    fn prepare_texture_copy(
        &self,
        buffer_gpu: GpuPtr<u8>,
        texture: &Texture,
        region: ResolvedRegion,
        op: &'static str,
    ) -> TextureCopy {
        let mtl_texture = self.resolve_texture(texture.id());
        let bpp = bytes_per_pixel(texture.desc().format)
            .unwrap_or_else(|| panic!("Unsupported texture format for {op}"));
        let (bytes_per_row, bytes_per_image) = region.linear_strides(bpp);
        let (buffer, offset) = self.resolve_buffer(
            buffer_gpu,
            (bytes_per_image * region.extent[2] as usize) as u64,
        );
        TextureCopy {
            texture: mtl_texture,
            buffer,
            offset,
            size: MTLSize {
                width: region.extent[0] as usize,
                height: region.extent[1] as usize,
                depth: region.extent[2] as usize,
            },
            origin: to_mtl_origin(region.origin),
            bytes_per_row,
            bytes_per_image,
        }
    }

    pub(crate) fn end_active_encoders(&mut self) {
        if let Some(encoder) = self.render_encoder.take() {
            encoder.endEncoding();
        }
        if let Some(encoder) = self.compute_encoder.take() {
            encoder.endEncoding();
        }
    }

    pub fn finish(&mut self) {
        if self.ended {
            return;
        }
        self.end_active_encoders();
        self.command_buffer.endCommandBuffer();
        self.ended = true;
    }

    /// Write a GPU timestamp into `heap` at `index`. This is a command-buffer-level command, so
    /// any open encoder must be closed first; the RHI contract calls this outside a render pass.
    pub fn write_timestamp(&mut self, pool: &super::query::MetalQueryPool, query: u32) {
        self.end_active_encoders();
        // SAFETY: `query` is within the pool's count; the heap is of type Timestamp.
        unsafe {
            self.command_buffer
                .writeTimestampIntoHeap_atIndex(&pool.heap, query as usize)
        };
    }

    pub fn barrier_with_hazard(&mut self, src: StageFlags, dst: StageFlags, hazard: HazardFlags) {
        self.encode_barrier(src, dst, hazard);
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
        let viewport = MTLViewport {
            originX: x as f64,
            originY: y as f64,
            width: width as f64,
            height: height as f64,
            znear: min_depth as f64,
            zfar: max_depth as f64,
        };
        let encoder = self
            .render_encoder
            .as_ref()
            .expect("No active render encoder");
        encoder.setViewport(viewport);
    }

    pub fn set_scissor(&mut self, x: i32, y: i32, width: u32, height: u32) {
        let scissor = MTLScissorRect {
            x: x.max(0) as usize,
            y: y.max(0) as usize,
            width: width as usize,
            height: height as usize,
        };
        let encoder = self
            .render_encoder
            .as_ref()
            .expect("No active render encoder");
        encoder.setScissorRect(scissor);
    }

    /// Only a barrier at an encoder's head makes that encoder wait, so `dst` splits: what this
    /// encoder runs, and what needs a publish here plus a wait on every encoder after.
    fn encode_barrier(&mut self, src: StageFlags, dst: StageFlags, hazard: HazardFlags) {
        let after = to_mtl_stages(src);
        let before = to_mtl_stages(dst);
        let visibility = visibility_from_hazard(hazard);

        let Some(encoder_stages) = self.open_encoder_stages() else {
            self.enqueue_queue_barrier(after, before, visibility);
            return;
        };
        let inside = before & encoder_stages;
        let outside = before & !encoder_stages;

        let Some(encoder) = self.open_encoder() else {
            return;
        };
        if !inside.is_empty() {
            encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                after, inside, visibility,
            );
        }
        if !outside.is_empty() {
            encoder
                .barrierAfterStages_beforeQueueStages_visibilityOptions(after, outside, visibility);
        }
    }

    /// The open encoder as the protocol both render and compute encoders conform to.
    ///
    /// `MTL4CommandEncoder` carries the barrier calls, so the two cases differ only in which
    /// field holds the encoder — not in what is done with it.
    fn open_encoder(&self) -> Option<&ProtocolObject<dyn MTL4CommandEncoder>> {
        if let Some(encoder) = self.render_encoder.as_ref() {
            // SAFETY: MTL4RenderCommandEncoder refines MTL4CommandEncoder.
            return Some(unsafe { super::cast_protocol(&**encoder) });
        }
        if let Some(encoder) = self.compute_encoder.as_ref() {
            // SAFETY: MTL4ComputeCommandEncoder refines MTL4CommandEncoder.
            return Some(unsafe { super::cast_protocol(&**encoder) });
        }
        None
    }

    /// Blit: Metal 4 routes copies through the compute encoder. Not AccelerationStructure:
    /// builds always take their own encoder.
    fn open_encoder_stages(&self) -> Option<MTLStages> {
        if self.render_encoder.is_some() {
            return Some(
                MTLStages::Vertex
                    | MTLStages::Fragment
                    | MTLStages::Tile
                    | MTLStages::Object
                    | MTLStages::Mesh,
            );
        }
        if self.compute_encoder.is_some() {
            return Some(MTLStages::Dispatch | MTLStages::Blit);
        }
        None
    }

    fn enqueue_queue_barrier(
        &mut self,
        after_queue_stages: MTLStages,
        before_stages: MTLStages,
        visibility: MTL4VisibilityOptions,
    ) {
        if let Some(pending) = self.pending_queue_barrier.as_mut() {
            pending.after_queue_stages |= after_queue_stages;
            pending.before_stages |= before_stages;
            pending.visibility |= visibility;
            return;
        }
        self.pending_queue_barrier = Some(PendingQueueBarrier {
            after_queue_stages,
            before_stages,
            visibility,
        });
    }

    /// Flush a barrier queued while no encoder was open onto the encoder just opened.
    fn apply_pending_queue_barrier(&mut self, encoder: &ProtocolObject<dyn MTL4CommandEncoder>) {
        if let Some(pending) = self.pending_queue_barrier.take() {
            encoder.barrierAfterQueueStages_beforeStages_visibilityOptions(
                pending.after_queue_stages,
                pending.before_stages,
                pending.visibility,
            );
        }
    }

    pub fn set_meshlet_pipeline(&mut self, pso: &MeshletPso) {
        use objc2_metal::MTL4RenderCommandEncoder as _;

        let mtl_pso = &pso.inner;
        let (pipeline, depth_stencil, depth_bias, cull_mode, winding) = (
            &mtl_pso.default_pipeline,
            &mtl_pso.depth_stencil,
            mtl_pso.depth_bias,
            mtl_pso.cull_mode,
            mtl_pso.winding,
        );
        let encoder = self
            .render_encoder
            .as_ref()
            .expect("set_meshlet_pipeline: no active render encoder");
        encoder.setRenderPipelineState(pipeline);
        encoder.setCullMode(cull_mode);
        encoder.setFrontFacingWinding(winding);
        encoder.setDepthStencilState(Some(depth_stencil));
        encoder.setDepthBias_slopeScale_clamp(depth_bias.0, depth_bias.1, depth_bias.2);
    }

    /// Draw using the bound mesh-shader pipeline, set via `CommandBuffer::set_pipeline`.
    pub fn draw_meshlets(&mut self, x: u32, y: u32, z: u32) {
        use objc2_metal::MTL4RenderCommandEncoder as _;
        let tg_obj = MESH_THREADS_PER_OBJECT_GROUP;
        let tg_mesh = MESH_THREADS_PER_MESH_GROUP;
        let groups = MTLSize {
            width: x as usize,
            height: y as usize,
            depth: z as usize,
        };
        let encoder = self
            .render_encoder
            .as_ref()
            .expect("No active render encoder");
        encoder.drawMeshThreadgroups_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
            groups, tg_obj, tg_mesh,
        );
    }

    /// Indirect mesh draw. Pipeline must be set via `CommandBuffer::set_pipeline`.
    /// `args` points to one indirect mesh dispatch command.
    pub fn draw_meshlets_indirect(&mut self, args: GpuPtr<u8>) {
        use objc2_metal::MTL4RenderCommandEncoder as _;
        let tg_obj = MESH_THREADS_PER_OBJECT_GROUP;
        let tg_mesh = MESH_THREADS_PER_MESH_GROUP;
        let encoder = self
            .render_encoder
            .as_ref()
            .expect("No active render encoder");
        encoder.drawMeshThreadgroupsWithIndirectBuffer_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
            args.address,
            tg_obj,
            tg_mesh,
        );
    }

    pub fn build_blas(&mut self, accel: &crate::accel::AccelerationStructure, desc: &BlasDesc<'_>) {
        use super::accel::make_blas_geometry_descriptors;
        use objc2_metal::{
            MTL4ComputeCommandEncoder as _, MTL4PrimitiveAccelerationStructureDescriptor,
            MTLBuffer as _,
        };

        let (mtl_as, scratch) = (
            &accel.inner.acceleration_structure,
            &accel.inner.scratch_buffer,
        );
        let geometries = make_blas_geometry_descriptors(desc);

        let primitive_desc = MTL4PrimitiveAccelerationStructureDescriptor::new();
        primitive_desc.setGeometryDescriptors(Some(&geometries.array));
        // SAFETY: MTL4PrimitiveAccelerationStructureDescriptor derives from
        // MTLAccelerationStructureDescriptor.
        let primitive_base: &objc2_metal::MTLAccelerationStructureDescriptor =
            unsafe { super::accel::downcast_base(&*primitive_desc) };
        super::accel::set_accel_usage(primitive_base, desc.flags);

        self.end_active_encoders();
        let encoder = self.begin_compute_encoder("build BLAS");

        let scratch_addr = scratch.gpuAddress();
        let scratch_range = objc2_metal::MTL4BufferRange {
            bufferAddress: scratch_addr,
            length: !0u64, // full remaining length
        };

        unsafe {
            encoder.buildAccelerationStructure_descriptor_scratchBuffer(
                mtl_as,
                // SAFETY: MTL4PrimitiveAccelerationStructureDescriptor derives from
                // MTL4AccelerationStructureDescriptor.
                downcast_base::<_, objc2_metal::MTL4AccelerationStructureDescriptor>(
                    &*primitive_desc,
                ),
                scratch_range,
            );
        }
        encoder.endEncoding();
    }

    pub fn build_tlas(&mut self, accel: &crate::accel::AccelerationStructure, desc: &TlasDesc) {
        use objc2_metal::{
            MTL4ComputeCommandEncoder as _, MTL4InstanceAccelerationStructureDescriptor,
            MTLBuffer as _,
        };

        let (mtl_as, scratch) = (
            &accel.inner.acceleration_structure,
            &accel.inner.scratch_buffer,
        );
        let instance_desc = MTL4InstanceAccelerationStructureDescriptor::new();
        unsafe {
            instance_desc.setInstanceDescriptorBuffer(objc2_metal::MTL4BufferRange {
                bufferAddress: desc.instance_buffer.address,
                // Metal uses its indirect instance layout, written by `Device::write_tlas_instance`.
                length: (desc.instance_count as u64)
                    * std::mem::size_of::<
                        objc2_metal::MTLIndirectAccelerationStructureInstanceDescriptor,
                    >() as u64,
            });
            instance_desc.setInstanceCount(desc.instance_count as usize);
        }
        // SAFETY: MTL4InstanceAccelerationStructureDescriptor derives from
        // MTLAccelerationStructureDescriptor.
        let instance_base: &objc2_metal::MTLAccelerationStructureDescriptor =
            unsafe { super::accel::downcast_base(&*instance_desc) };
        super::accel::set_accel_usage(instance_base, desc.flags);

        self.end_active_encoders();
        let encoder = self.begin_compute_encoder("build TLAS");

        let scratch_addr = scratch.gpuAddress();
        let scratch_range = objc2_metal::MTL4BufferRange {
            bufferAddress: scratch_addr,
            length: !0u64,
        };

        unsafe {
            encoder.buildAccelerationStructure_descriptor_scratchBuffer(
                mtl_as,
                // SAFETY: MTL4InstanceAccelerationStructureDescriptor derives from
                // MTL4AccelerationStructureDescriptor.
                downcast_base::<_, objc2_metal::MTL4AccelerationStructureDescriptor>(
                    &*instance_desc,
                ),
                scratch_range,
            );
        }
        encoder.endEncoding();
    }
}

impl Drop for MetalCommandBuffer {
    fn drop(&mut self) {
        let slot = self.frame_table_slot.clone();
        let previous = slot
            .argument_table
            .borrow_mut()
            .replace(self.argument_table.clone());
        debug_assert!(previous.is_none());
        slot.in_use.set(false);
        // The queue holds submitted command buffers until their fence signals, so reaching
        // this drop means the GPU is done with the slot.
        if let Some(pool) = slot.pool.clone() {
            pool.borrow_mut().push(slot);
        }
    }
}

impl MetalDevice {
    /// Build a fresh table slot. `pool` is the free list it returns to on drop; swapchain slots
    /// pass `None`.
    fn new_table_slot(
        &self,
        pool: Option<SharedTableSlotPool>,
        what: &str,
    ) -> RhiResult<Rc<MetalFrameTableSlot>> {
        let root_table_buffer = create_root_table_buffer(
            self.shared.device.as_ref(),
            self.shared.residency_set.as_ref(),
            &self.shared.residency_dirty,
        )?;
        let command_allocator = self.shared.device.newCommandAllocator().ok_or_else(|| {
            RhiError::CommandBuffer(format!("Failed to create {what} MTL4CommandAllocator").into())
        })?;
        let command_buffer = self.shared.device.newCommandBuffer().ok_or_else(|| {
            RhiError::CommandBuffer(format!("Failed to create {what} MTL4CommandBuffer").into())
        })?;
        Ok(Rc::new(MetalFrameTableSlot {
            root_table_buffer,
            command_allocator,
            command_buffer,
            argument_table: RefCell::new(None),
            in_use: Cell::new(false),
            pool,
        }))
    }

    fn acquire_frame_table_slot(&self, frame_index: usize) -> RhiResult<Rc<MetalFrameTableSlot>> {
        let mut slots = self.frame_table_slots.borrow_mut();
        let slot_entry = slots
            .get_mut(frame_index)
            .ok_or_else(|| RhiError::CommandBuffer("invalid Metal frame index".into()))?;
        let slot = if let Some(slot) = slot_entry {
            slot.clone()
        } else {
            let slot = self.new_table_slot(None, "frame")?;
            *slot_entry = Some(slot.clone());
            slot
        };
        if slot.in_use.replace(true) {
            return Err(RhiError::CommandBuffer(
                "Metal frame table slot is still in use".into(),
            ));
        }
        slot.command_allocator.reset();
        Ok(slot)
    }

    /// Take a generic table slot from the free list, or build one when empty. Without this,
    /// `create_command_buffer` rebuilt an allocator, command buffer, argument table and 1 MiB root
    /// ring on every call — Vulkan pools all of its command buffers, so this keeps parity.
    fn acquire_pooled_table_slot(&self) -> RhiResult<Rc<MetalFrameTableSlot>> {
        let pooled = self.table_slot_pool.borrow_mut().pop();
        let slot = match pooled {
            Some(slot) => slot,
            None => self.new_table_slot(Some(self.table_slot_pool.clone()), "pooled")?,
        };
        debug_assert!(!slot.in_use.get(), "pooled slot handed out while in use");
        slot.in_use.set(true);
        slot.command_allocator.reset();
        Ok(slot)
    }

    /// Wrap an acquired slot in a recording command buffer, releasing it again on failure so a
    /// failed create never strands it as in-use.
    fn create_command_buffer_in_slot(
        &self,
        slot: Rc<MetalFrameTableSlot>,
    ) -> RhiResult<CommandBuffer> {
        let mtl_cmd = MetalCommandBuffer::new(
            slot.command_buffer.clone(),
            self.shared.clone(),
            slot.clone(),
        );
        let mtl_cmd = match mtl_cmd {
            Ok(cmd) => cmd,
            Err(error) => {
                slot.in_use.set(false);
                if let Some(pool) = slot.pool.clone() {
                    pool.borrow_mut().push(slot);
                }
                return Err(error);
            }
        };

        Ok(CommandBuffer::new(Box::new(mtl_cmd)))
    }

    pub fn create_command_buffer(&self) -> RhiResult<CommandBuffer> {
        let slot = self.acquire_pooled_table_slot()?;
        self.create_command_buffer_in_slot(slot)
    }

    /// As [`create_command_buffer`](Self::create_command_buffer), but bound to `frame_index`'s
    /// table slot and to the swapchain's drawable, so a render pass can target the backbuffer.
    pub fn create_command_buffer_for_swapchain(
        &self,
        sc: &MetalSwapchain,
        frame_index: usize,
    ) -> RhiResult<CommandBuffer> {
        let frame_table_slot = self.acquire_frame_table_slot(frame_index)?;
        let mut cmd_buf = self.create_command_buffer_in_slot(frame_table_slot)?;

        cmd_buf.inner.drawable_slot = Some(sc.drawable.clone());

        Ok(cmd_buf)
    }
}
