use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer,
    MTL4CommandEncoder, MTL4ComputeCommandEncoder, MTL4CounterHeap, MTL4RenderCommandEncoder,
    MTL4RenderPassDescriptor, MTL4VisibilityOptions, MTLBuffer, MTLDevice, MTLGPUAddress,
    MTLIndexType, MTLLoadAction, MTLOrigin, MTLPrimitiveType, MTLRenderStages, MTLResidencySet,
    MTLResourceOptions, MTLScissorRect, MTLSize, MTLStages, MTLStoreAction, MTLTexture,
    MTLViewport,
};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::barrier::{HazardFlags, StageFlags};
use crate::command::{LoadOp, RenderPassDesc, RenderTarget, RenderTargetKind, StoreOp};
use crate::pipeline::{ComputePso, GraphicsPso, MeshletPso};
use crate::texture::{ResolvedRegion, Texture, bytes_per_pixel};
use crate::types::{BlasDesc, GpuPtr, TextureId, TlasDesc};

use super::as_allocation;
use super::device::MetalShared;
use super::swapchain::SharedDrawableSlot;

// Metal root-table entries are 16 bytes.
fn to_mtl_origin(origin: [u32; 3]) -> MTLOrigin {
    MTLOrigin {
        x: origin[0] as usize,
        y: origin[1] as usize,
        z: origin[2] as usize,
    }
}

const ALL_RENDER_STAGES: MTLRenderStages = MTLRenderStages(
    MTLRenderStages::Vertex.0
        | MTLRenderStages::Fragment.0
        | MTLRenderStages::Object.0
        | MTLRenderStages::Mesh.0,
);

const ROOT_TABLE_SLOT_BYTES: usize = 16;
const ROOT_TABLE_RING_ENTRIES: usize = 65_536;
const ROOT_TABLE_RING_BYTES: usize = ROOT_TABLE_SLOT_BYTES * ROOT_TABLE_RING_ENTRIES;

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
        .newBufferWithLength_options(ROOT_TABLE_RING_BYTES, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| {
            crate::error::RhiError::CommandBuffer(
                "Failed to allocate Metal root table ring buffer".into(),
            )
        })?;
    add_buffer_to_residency(root_table_buffer.as_ref(), residency_set, residency_dirty);
    Ok(root_table_buffer)
}

/// One command buffer's worth of reusable native state, recycled rather than rebuilt: the root
/// ring alone is a 1 MiB allocation that has to join the residency set. Only handed out again
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

pub(crate) type SharedFrameTableSlots = Rc<RefCell<Vec<Option<Rc<MetalFrameTableSlot>>>>>;
/// Free list of generic (non-swapchain) table slots.
pub(crate) type SharedTableSlotPool = Rc<RefCell<Vec<Rc<MetalFrameTableSlot>>>>;

pub struct MetalCommandBuffer {
    pub(crate) command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    /// Never read: held so the allocator outlives the command buffer that draws from it.
    #[allow(dead_code)]
    pub(crate) command_allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    pub(crate) render_encoder: Option<Retained<ProtocolObject<dyn MTL4RenderCommandEncoder>>>,
    pub(crate) compute_encoder: Option<Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>>,
    /// Set when this command buffer renders to the swapchain; the drawable is taken on first use.
    pub(crate) drawable_slot: Option<SharedDrawableSlot>,
    pub(crate) depth_texture: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
    current_topology: MTLPrimitiveType,
    pub(crate) shared: Rc<MetalShared>,
    argument_table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    frame_table_slot: Rc<MetalFrameTableSlot>,
    root_table_ptr: *mut u8,
    root_table_gpu_base: MTLGPUAddress,
    root_table_cursor: usize,
    root_table_capacity: usize,
    current_threads_per_threadgroup: [u32; 3],
    current_mesh_tpg_object: MTLSize,
    current_mesh_tpg_mesh: MTLSize,
    pending_queue_barrier: Option<PendingQueueBarrier>,
    pending_split_barrier: Option<(StageFlags, HazardFlags)>,
    /// Render targets already written by an earlier pass in this command buffer. Metal 4 tracks no
    /// hazards of its own, so a second pass on the same attachment needs an explicit dependency.
    written_targets: Vec<RenderTargetKind>,
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
                crate::error::RhiError::CommandBuffer(format!(
                    "Failed to create Metal 4 argument table: {e}"
                ))
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
        let textures = self.shared.textures.borrow();
        textures
            .get(id.0 as usize)
            .and_then(|t| t.as_ref())
            .expect("Invalid texture ID")
            .clone()
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
        self.apply_pending_queue_barrier_compute(&encoder);
        encoder
    }

    fn alloc_root_bytes(&mut self, size: usize) -> (MTLGPUAddress, *mut u8) {
        let size = (size + ROOT_TABLE_SLOT_BYTES - 1) & !(ROOT_TABLE_SLOT_BYTES - 1);
        let end = self.root_table_cursor + size;
        assert!(
            end <= self.root_table_capacity,
            "Metal root table ring overflow ({} bytes). Increase ROOT_TABLE_RING_ENTRIES.",
            self.root_table_capacity
        );
        let offset = self.root_table_cursor;
        self.root_table_cursor = end;
        let ptr = unsafe { self.root_table_ptr.add(offset) };
        let addr = self.root_table_gpu_base + offset as u64;
        (addr, ptr)
    }

    pub(crate) fn new(
        command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
        command_allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
        shared: Rc<MetalShared>,
        frame_table_slot: Rc<MetalFrameTableSlot>,
    ) -> crate::error::RhiResult<Self> {
        command_buffer.beginCommandBufferWithAllocator(&command_allocator);
        // The queue owns the residency set for every submitted command buffer; attaching it here
        // too would repeat a setup call per frame without changing residency.

        let argument_table =
            if let Some(argument_table) = frame_table_slot.argument_table.borrow_mut().take() {
                argument_table
            } else {
                Self::create_argument_table(shared.device.as_ref())?
            };

        let root_table_ptr = frame_table_slot.root_table_buffer.contents().as_ptr() as *mut u8;
        let root_table_gpu_base = frame_table_slot.root_table_buffer.gpuAddress();

        unsafe {
            argument_table.setAddress_atIndex(shared.texture_heap.gpuAddress(), 1);
            argument_table.setAddress_atIndex(shared.sampler_heap.gpuAddress(), 2);
            // Recycled tables would otherwise carry the previous frame's root pointer.
            argument_table.setAddress_atIndex(0, 0);
        }

        Ok(Self {
            command_buffer,
            command_allocator,
            render_encoder: None,
            compute_encoder: None,
            drawable_slot: None,
            depth_texture: None,
            current_topology: MTLPrimitiveType::Triangle,
            shared,
            argument_table,
            frame_table_slot,
            root_table_ptr,
            root_table_gpu_base,
            root_table_cursor: 0,
            root_table_capacity: ROOT_TABLE_RING_BYTES,
            current_threads_per_threadgroup: [1, 1, 1],
            current_mesh_tpg_object: MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            current_mesh_tpg_mesh: MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
            pending_queue_barrier: None,
            pending_split_barrier: None,
            written_targets: Vec::new(),
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
                RenderTargetKind::SwapchainImage(_) => self
                    .depth_texture
                    .as_ref()
                    .expect(
                        "Metal swapchain has no depth texture; use an explicit depth attachment",
                    )
                    .clone(),
                RenderTargetKind::Texture(id) => self.resolve_texture(id),
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

        self.apply_pending_queue_barrier_render(&encoder);

        self.render_encoder = Some(encoder);
    }

    pub fn end_render_pass(&mut self) {
        if let Some(encoder) = self.render_encoder.take() {
            encoder.endEncoding();
        }
    }

    /// Order a pass against the earlier passes in this command buffer that wrote the same
    /// attachment. Metal 4 leaves every resource untracked, so two render encoders on one texture
    /// are free to overlap: the second pass's `Load` can sample the pre-`Store` contents, or its
    /// own `Store` can be overtaken by the first pass's. The Vulkan backend gets this ordering for
    /// free from the attachment's layout transition; here it has to be stated.
    ///
    /// The barrier is merged into `pending_queue_barrier` rather than encoded directly, so it
    /// lands on the new encoder alongside whatever the caller already asked for.
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
        let (pipeline, depth_stencil, depth_bias, cull_mode, winding, topology) = match &pso.inner {
            crate::pipeline::GraphicsPsoInner::Metal(mtl_pso) => (
                &mtl_pso.pipeline,
                &mtl_pso.depth_stencil,
                mtl_pso.depth_bias,
                mtl_pso.cull_mode,
                mtl_pso.winding,
                mtl_pso.topology,
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!("wrong backend"),
        };
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
        let mtl_pso = backend_expect!(&pso.inner, crate::pipeline::ComputePsoInner::Metal);

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
        let (slot_addr, slot_ptr) = self.alloc_root_bytes(std::mem::size_of::<u64>());
        unsafe {
            std::ptr::write_unaligned(slot_ptr as *mut u64, root.address);
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

    pub fn draw_indexed(&mut self, indices: GpuPtr<u8>, index_count: u32, instance_count: u32) {
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
                    0,  // base vertex
                    0,  // first instance
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
        let (mtl_texture, buffer, offset, size, origin, bytes_per_row, bytes_per_image) =
            self.prepare_texture_copy(src, texture, region, "copy_buffer_to_texture");

        let encoder = self.begin_copy_encoder("copy to texture");
        unsafe {
            encoder.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                &buffer,
                offset as usize,
                bytes_per_row,
                bytes_per_image,
                size,
                &mtl_texture,
                region.layer as usize,
                region.mip as usize,
                origin,
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
        let (mtl_texture, buffer, offset, size, origin, bytes_per_row, bytes_per_image) =
            self.prepare_texture_copy(dst, texture, region, "copy_texture_to_buffer");

        let encoder = self.begin_copy_encoder("copy from texture");
        unsafe {
            encoder.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                &mtl_texture,
                region.layer as usize,
                region.mip as usize,
                origin,
                size,
                &buffer,
                offset as usize,
                bytes_per_row,
                bytes_per_image,
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

    /// Resolve the texture and linear buffer, and compute the row /
    /// image strides for a buffer↔texture copy on the Metal 4 compute encoder.
    #[allow(clippy::type_complexity)]
    fn prepare_texture_copy(
        &self,
        buffer_gpu: GpuPtr<u8>,
        texture: &Texture,
        region: ResolvedRegion,
        op: &'static str,
    ) -> (
        Retained<ProtocolObject<dyn MTLTexture>>,
        Retained<ProtocolObject<dyn MTLBuffer>>,
        u64,
        MTLSize,
        MTLOrigin,
        usize,
        usize,
    ) {
        let mtl_texture = self.resolve_texture(texture.id());
        let bpp = bytes_per_pixel(texture.desc().format)
            .unwrap_or_else(|| panic!("Unsupported texture format for {op}"));
        let (bytes_per_row, bytes_per_image) = region.linear_strides(bpp);
        let (buffer, offset) = self.resolve_buffer(
            buffer_gpu,
            (bytes_per_image * region.extent[2] as usize) as u64,
        );
        let size = MTLSize {
            width: region.extent[0] as usize,
            height: region.extent[1] as usize,
            depth: region.extent[2] as usize,
        };
        let origin = to_mtl_origin(region.origin);
        (
            mtl_texture,
            buffer,
            offset,
            size,
            origin,
            bytes_per_row,
            bytes_per_image,
        )
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
    pub fn write_timestamp(&mut self, heap: &ProtocolObject<dyn MTL4CounterHeap>, index: usize) {
        self.end_active_encoders();
        // SAFETY: `index` is within the heap's count (caller's QueryPool contract); the heap is of
        // type Timestamp.
        unsafe {
            self.command_buffer
                .writeTimestampIntoHeap_atIndex(heap, index)
        };
    }

    pub fn barrier(&mut self, src: StageFlags, dst: StageFlags) {
        self.encode_barrier(src, dst, None);
    }

    pub fn barrier_with_hazard(&mut self, src: StageFlags, dst: StageFlags, hazard: HazardFlags) {
        self.encode_barrier(src, dst, Some(hazard));
    }

    pub fn signal_after(&mut self, src: StageFlags, hazard: HazardFlags) {
        if let Some((pending_src, pending_hazard)) = self.pending_split_barrier.as_mut() {
            *pending_src |= src;
            *pending_hazard |= hazard;
            return;
        }
        self.pending_split_barrier = Some((src, hazard));
    }

    pub fn wait_before(&mut self, dst: StageFlags, hazard: HazardFlags) {
        let (src, pending_hazard) = self
            .pending_split_barrier
            .take()
            .expect("wait_before called without a matching signal_after");
        self.barrier_with_hazard(src, dst, pending_hazard | hazard);
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

    fn encode_barrier(&mut self, src: StageFlags, dst: StageFlags, hazard: Option<HazardFlags>) {
        let after = to_mtl_stages(src);
        let before = to_mtl_stages(dst);
        let visibility = visibility_from_hazard(hazard.unwrap_or_else(HazardFlags::empty));

        if let Some(encoder) = self.render_encoder.as_ref() {
            let needs_queue_barrier = dst.contains(StageFlags::COMPUTE)
                || dst.contains(StageFlags::TRANSFER)
                || dst.contains(StageFlags::ALL_COMMANDS);
            if needs_queue_barrier {
                encoder.barrierAfterStages_beforeQueueStages_visibilityOptions(
                    after, before, visibility,
                );
            } else {
                encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                    after, before, visibility,
                );
            }
            return;
        }

        if let Some(encoder) = self.compute_encoder.as_ref() {
            let needs_queue_barrier = dst.contains(StageFlags::VERTEX_SHADER)
                || dst.contains(StageFlags::PIXEL_SHADER)
                || dst.contains(StageFlags::TRANSFER)
                || dst.contains(StageFlags::RASTER_COLOR_OUT)
                || dst.contains(StageFlags::RASTER_DEPTH_OUT)
                || dst.contains(StageFlags::ALL_GRAPHICS)
                || dst.contains(StageFlags::ALL_COMMANDS);
            if needs_queue_barrier {
                encoder.barrierAfterStages_beforeQueueStages_visibilityOptions(
                    after, before, visibility,
                );
            } else {
                encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                    after, before, visibility,
                );
            }
            return;
        }

        self.enqueue_queue_barrier(after, before, visibility);
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

    fn apply_pending_queue_barrier_render(
        &mut self,
        encoder: &ProtocolObject<dyn MTL4RenderCommandEncoder>,
    ) {
        if let Some(pending) = self.pending_queue_barrier.take() {
            encoder.barrierAfterQueueStages_beforeStages_visibilityOptions(
                pending.after_queue_stages,
                pending.before_stages,
                pending.visibility,
            );
        }
    }

    fn apply_pending_queue_barrier_compute(
        &mut self,
        encoder: &ProtocolObject<dyn MTL4ComputeCommandEncoder>,
    ) {
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

        let (pipeline, depth_stencil, depth_bias, cull_mode, winding) = match &pso.inner {
            crate::pipeline::MeshletPsoInner::Metal(mtl_pso) => (
                &mtl_pso.default_pipeline,
                &mtl_pso.depth_stencil,
                mtl_pso.depth_bias,
                mtl_pso.cull_mode,
                mtl_pso.winding,
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!("wrong backend"),
        };
        self.current_mesh_tpg_object = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        self.current_mesh_tpg_mesh = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };

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
        let tg_obj = self.current_mesh_tpg_object;
        let tg_mesh = self.current_mesh_tpg_mesh;
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
        let tg_obj = self.current_mesh_tpg_object;
        let tg_mesh = self.current_mesh_tpg_mesh;
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

    pub fn build_blas(&mut self, accel: &crate::accel::AccelerationStructure, desc: &BlasDesc) {
        use super::accel::make_blas_geometry_descriptors;
        use objc2_metal::{
            MTL4ComputeCommandEncoder as _, MTL4PrimitiveAccelerationStructureDescriptor,
            MTLBuffer as _,
        };

        let (mtl_as, scratch) = match &accel.inner {
            #[cfg(feature = "metal")]
            crate::accel::AccelInner::Metal(a) => (&a.acceleration_structure, &a.scratch_buffer),
            #[allow(unreachable_patterns)]
            _ => unreachable!("acceleration structure belongs to another backend"),
        };
        let geometries = make_blas_geometry_descriptors(desc);

        let primitive_desc = MTL4PrimitiveAccelerationStructureDescriptor::new();
        primitive_desc.setGeometryDescriptors(Some(&geometries.array));
        let primitive_base = unsafe {
            &*(primitive_desc.as_ref() as *const MTL4PrimitiveAccelerationStructureDescriptor
                as *const objc2_metal::MTLAccelerationStructureDescriptor)
        };
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
                &*(primitive_desc.as_ref() as *const MTL4PrimitiveAccelerationStructureDescriptor
                    as *const objc2_metal::MTL4AccelerationStructureDescriptor),
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

        let (mtl_as, scratch) = match &accel.inner {
            #[cfg(feature = "metal")]
            crate::accel::AccelInner::Metal(a) => (&a.acceleration_structure, &a.scratch_buffer),
            #[allow(unreachable_patterns)]
            _ => unreachable!("acceleration structure belongs to another backend"),
        };
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
        let instance_base = unsafe {
            &*(instance_desc.as_ref() as *const MTL4InstanceAccelerationStructureDescriptor
                as *const objc2_metal::MTLAccelerationStructureDescriptor)
        };
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
                &*(instance_desc.as_ref() as *const MTL4InstanceAccelerationStructureDescriptor
                    as *const objc2_metal::MTL4AccelerationStructureDescriptor),
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

fn to_mtl_stages(flags: StageFlags) -> MTLStages {
    let mut stages = MTLStages::empty();

    if flags.contains(StageFlags::ALL_COMMANDS) {
        return MTLStages::All;
    }

    if flags.contains(StageFlags::VERTEX_SHADER) {
        stages |= MTLStages::Vertex;
    }
    if flags.contains(StageFlags::PIXEL_SHADER) || flags.contains(StageFlags::RASTER_COLOR_OUT) {
        stages |= MTLStages::Fragment;
    }
    if flags.contains(StageFlags::COMPUTE) {
        stages |= MTLStages::Dispatch;
    }
    if flags.contains(StageFlags::TRANSFER) {
        stages |= MTLStages::Blit;
    }
    if flags.contains(StageFlags::RASTER_DEPTH_OUT) {
        stages |= MTLStages::Fragment;
    }

    if flags.contains(StageFlags::ALL_GRAPHICS) {
        stages |= MTLStages::Vertex;
        stages |= MTLStages::Fragment;
        stages |= MTLStages::Tile;
        stages |= MTLStages::Mesh;
        stages |= MTLStages::Object;
    }

    if stages.is_empty() {
        MTLStages::All
    } else {
        stages
    }
}

fn visibility_from_hazard(hazard: HazardFlags) -> MTL4VisibilityOptions {
    if hazard.is_empty() {
        return MTL4VisibilityOptions::None;
    }

    // Device visibility is required for GPU-written arguments, depth, and descriptor aliases.
    let needs_device = hazard.intersects(
        HazardFlags::DRAW_ARGUMENTS | HazardFlags::DEPTH_STENCIL | HazardFlags::DESCRIPTORS,
    );
    let needs_alias = hazard.contains(HazardFlags::DESCRIPTORS);

    let mut visibility = if needs_device {
        MTL4VisibilityOptions::Device
    } else {
        MTL4VisibilityOptions::None
    };
    if needs_alias {
        visibility |= MTL4VisibilityOptions::ResourceAlias;
    }
    visibility
}
