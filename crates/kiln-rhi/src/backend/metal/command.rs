use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer,
    MTL4CommandEncoder, MTL4ComputeCommandEncoder, MTL4CounterHeap, MTL4RenderCommandEncoder,
    MTL4RenderPassDescriptor, MTL4VisibilityOptions, MTLBuffer, MTLDepthStencilState, MTLDevice,
    MTLGPUAddress, MTLIndexType, MTLLoadAction, MTLOrigin, MTLPrimitiveType, MTLRenderStages,
    MTLResidencySet, MTLResourceOptions, MTLScissorRect, MTLSize, MTLStages, MTLStencilOperation,
    MTLStoreAction, MTLTexture, MTLViewport,
};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use crate::barrier::{HazardFlags, StageFlags};
use crate::command::{LoadOp, RenderPassDesc, RenderTargetKind, StoreOp};
use crate::pipeline::{ComputePso, DepthStencilState, GraphicsPso, MeshletPso};
use crate::texture::{Texture, bytes_per_pixel};
use crate::types::*;

use super::as_allocation;
use super::device::MetalShared;
use super::swapchain::SharedDrawableSlot;

// Metal root-table entries are 16 bytes.
const ROOT_TABLE_SLOT_BYTES: usize = 16;
const ROOT_TABLE_RING_ENTRIES: usize = 65_536;
const ROOT_TABLE_RING_BYTES: usize = ROOT_TABLE_SLOT_BYTES * ROOT_TABLE_RING_ENTRIES;
/// "Contents unknown" — 0 is a legitimate slot value, so it cannot mean invalid.
const INVALID_TABLE_ADDRESS: MTLGPUAddress = u64::MAX;

/// Keyed by [`DepthStencilKey`], which already hashes the depth-bias floats — so the bias is not
/// stored alongside it.
type CachedDepthStencil = Retained<ProtocolObject<dyn MTLDepthStencilState>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct DepthStencilKey {
    depth_mode: DepthFlags,
    depth_test: CompareOp,
    depth_bias: u32,
    depth_bias_slope_factor: u32,
    depth_bias_clamp: u32,
    stencil_read_mask: u8,
    stencil_write_mask: u8,
    stencil_front: StencilKey,
    stencil_back: StencilKey,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct StencilKey {
    test: CompareOp,
    fail_op: StencilOp,
    pass_op: StencilOp,
    depth_fail_op: StencilOp,
    reference: u8,
}

#[derive(Clone, Copy, PartialEq)]
struct CurrentDepthStencil {
    key: DepthStencilKey,
    bias: Option<(f32, f32, f32)>,
}

impl From<&DepthStencilState> for DepthStencilKey {
    fn from(state: &DepthStencilState) -> Self {
        let stencil_key = |stencil: &crate::pipeline::StencilDesc| StencilKey {
            test: stencil.test,
            fail_op: stencil.fail_op,
            pass_op: stencil.pass_op,
            depth_fail_op: stencil.depth_fail_op,
            reference: stencil.reference,
        };
        Self {
            depth_mode: state.depth_mode,
            depth_test: state.depth_test,
            depth_bias: state.depth_bias.to_bits(),
            depth_bias_slope_factor: state.depth_bias_slope_factor.to_bits(),
            depth_bias_clamp: state.depth_bias_clamp.to_bits(),
            stencil_read_mask: state.stencil_read_mask,
            stencil_write_mask: state.stencil_write_mask,
            stencil_front: stencil_key(&state.stencil_front),
            stencil_back: stencil_key(&state.stencil_back),
        }
    }
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
    /// Bindless heap addresses, bound at argument-table slots 1 and 2.
    texture_heap_addr: MTLGPUAddress,
    sampler_heap_addr: MTLGPUAddress,
    depth_stencil_states: HashMap<DepthStencilKey, CachedDepthStencil>,
    current_threads_per_threadgroup: [u32; 3],
    current_mesh_tpg_object: MTLSize,
    current_mesh_tpg_mesh: MTLSize,
    pending_queue_barrier: Option<PendingQueueBarrier>,
    pending_split_barrier: Option<(StageFlags, HazardFlags)>,
    bound_root_table: MTLGPUAddress,
    bound_texture_heap: MTLGPUAddress,
    bound_sampler_heap: MTLGPUAddress,
    current_root_table: MTLGPUAddress,
    current_depth_stencil: Option<CurrentDepthStencil>,
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

    /// Bytes available from `addr` to the end of its allocation. Used only where the
    /// element count is GPU-driven (indirect indexed draws) and the CPU
    /// cannot compute an exact length. Metal 4 consumes the GPU address directly, so
    /// non-indirect draws pass their addresses through without any lookup.
    fn allocation_remaining(&self, addr: GpuPtr<u8>) -> u64 {
        let addr_u64 = addr.address;
        let allocations = self.shared.allocations.borrow();
        if let Some((&base, alloc)) = allocations.range(..=addr_u64).next_back() {
            let offset = addr_u64 - base;
            if offset < alloc.size {
                return alloc.size - offset;
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

    /// Bind the root-data pointer at slot 0, skipping the message send when it is already there.
    /// Every draw and dispatch funnels through here, so slot 0 has exactly one writer.
    fn bind_root_table(&mut self, root_table: MTLGPUAddress) {
        if self.bound_root_table == root_table {
            return;
        }
        unsafe { self.argument_table.setAddress_atIndex(root_table, 0) };
        self.bound_root_table = root_table;
    }

    /// Re-bind the bindless heap slots (1 and 2) if they aren't already. Slot 0 is left alone:
    /// every draw and dispatch binds it itself.
    fn refresh_argument_table(&mut self) {
        unsafe {
            if self.bound_texture_heap != self.texture_heap_addr {
                self.argument_table
                    .setAddress_atIndex(self.texture_heap_addr, 1);
                self.bound_texture_heap = self.texture_heap_addr;
            }
            if self.bound_sampler_heap != self.sampler_heap_addr {
                self.argument_table
                    .setAddress_atIndex(self.sampler_heap_addr, 2);
                self.bound_sampler_heap = self.sampler_heap_addr;
            }
        }
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

    fn add_accel_to_residency(
        &self,
        accel: &ProtocolObject<dyn objc2_metal::MTLAccelerationStructure>,
    ) {
        self.shared
            .residency_set
            .addAllocation(as_allocation(accel));
        self.shared.residency_dirty.set(true);
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

        let mut cmd = Self {
            command_buffer,
            command_allocator,
            render_encoder: None,
            compute_encoder: None,
            drawable_slot: None,
            depth_texture: None,
            current_topology: MTLPrimitiveType::Triangle,
            texture_heap_addr: shared.texture_heap.gpuAddress(),
            sampler_heap_addr: shared.sampler_heap.gpuAddress(),
            shared,
            argument_table,
            frame_table_slot,
            root_table_ptr,
            root_table_gpu_base,
            root_table_cursor: 0,
            root_table_capacity: ROOT_TABLE_RING_BYTES,
            depth_stencil_states: HashMap::new(),
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
            bound_root_table: INVALID_TABLE_ADDRESS,
            bound_texture_heap: INVALID_TABLE_ADDRESS,
            bound_sampler_heap: INVALID_TABLE_ADDRESS,
            current_root_table: 0,
            current_depth_stencil: None,
            ended: false,
        };

        cmd.refresh_argument_table();
        Ok(cmd)
    }

    pub fn begin_render_pass(&mut self, desc: &RenderPassDesc) {
        self.end_active_encoders();
        self.current_depth_stencil = None;
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

            match depth_att.target.kind() {
                RenderTargetKind::SwapchainImage(_) => {
                    if let Some(depth_tex) = &self.depth_texture {
                        depth.setTexture(Some(depth_tex));
                    }
                }
                RenderTargetKind::Texture(id) => {
                    let tex = self.resolve_texture(id);
                    depth.setTexture(Some(&tex));
                }
            }

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

        self.apply_pending_queue_barrier_render(&encoder);

        self.render_encoder = Some(encoder);
    }

    pub fn end_render_pass(&mut self) {
        if let Some(encoder) = self.render_encoder.take() {
            encoder.endEncoding();
        }
    }

    pub fn set_graphics_pipeline(&mut self, pso: &GraphicsPso) {
        let (pipeline, cull_mode, winding, topology) = match &pso.inner {
            crate::pipeline::GraphicsPsoInner::Metal(mtl_pso) => (
                mtl_pso.pipeline.clone(),
                mtl_pso.cull_mode,
                mtl_pso.winding,
                mtl_pso.topology,
            ),
            #[allow(unreachable_patterns)]
            _ => unreachable!("wrong backend"),
        };
        self.current_topology = topology;
        self.refresh_argument_table();
        let encoder = self
            .render_encoder
            .as_ref()
            .expect("No active render encoder");
        encoder.setRenderPipelineState(&pipeline);
        encoder.setCullMode(cull_mode);
        encoder.setFrontFacingWinding(winding);
        encoder.setArgumentTable_atStages(
            &self.argument_table,
            MTLRenderStages::Vertex | MTLRenderStages::Fragment,
        );
    }

    pub fn set_compute_pipeline(&mut self, pso: &ComputePso) {
        let mtl_pso = backend_expect!(&pso.inner, crate::pipeline::ComputePsoInner::Metal);

        self.end_active_encoders();
        let encoder = self.begin_compute_encoder(mtl_pso.label.as_deref().unwrap_or("compute"));
        encoder.setComputePipelineState(&mtl_pso.pipeline);

        self.current_threads_per_threadgroup = mtl_pso.threads_per_threadgroup;
        self.refresh_argument_table();
        encoder.setArgumentTable(Some(&self.argument_table));

        self.compute_encoder = Some(encoder);
    }

    pub fn set_depth_stencil_state(&mut self, state: &DepthStencilState) {
        let depth_bias = if state.depth_bias != 0.0 || state.depth_bias_slope_factor != 0.0 {
            Some((
                state.depth_bias,
                state.depth_bias_slope_factor,
                state.depth_bias_clamp,
            ))
        } else {
            None
        };

        let key = DepthStencilKey::from(state);
        let current = CurrentDepthStencil {
            key,
            bias: depth_bias,
        };
        if self.current_depth_stencil == Some(current) {
            return;
        }
        let cached = if let Some(cached) = self.depth_stencil_states.get(&key) {
            cached
        } else {
            let ds_desc = objc2_metal::MTLDepthStencilDescriptor::new();
            let depth_test = state.depth_mode.contains(DepthFlags::READ);
            let depth_write = state.depth_mode.contains(DepthFlags::WRITE);

            if depth_test {
                ds_desc.setDepthCompareFunction(compare_op_to_mtl(state.depth_test));
            } else {
                ds_desc.setDepthCompareFunction(objc2_metal::MTLCompareFunction::Always);
            }
            ds_desc.setDepthWriteEnabled(depth_write);

            if state.stencil_enabled() {
                let front = make_stencil_descriptor(
                    &state.stencil_front,
                    state.stencil_read_mask,
                    state.stencil_write_mask,
                );
                let back = make_stencil_descriptor(
                    &state.stencil_back,
                    state.stencil_read_mask,
                    state.stencil_write_mask,
                );
                ds_desc.setFrontFaceStencil(Some(&front));
                ds_desc.setBackFaceStencil(Some(&back));
            } else {
                ds_desc.setFrontFaceStencil(None);
                ds_desc.setBackFaceStencil(None);
            }

            let Some(ds_state) = self
                .shared
                .device
                .newDepthStencilStateWithDescriptor(&ds_desc)
            else {
                return;
            };
            self.depth_stencil_states.entry(key).or_insert(ds_state)
        };
        if let Some(encoder) = self.render_encoder.as_ref() {
            encoder.setDepthStencilState(Some(cached));
            let (bias, slope, clamp) = depth_bias.unwrap_or((0.0, 0.0, 0.0));
            encoder.setDepthBias_slopeScale_clamp(bias, slope, clamp);
        }
        self.current_depth_stencil = Some(current);
    }

    pub fn set_root_data(&mut self, root: GpuPtr<u8>) {
        let (slot_addr, slot_ptr) = self.alloc_root_bytes(std::mem::size_of::<u64>());
        unsafe {
            std::ptr::write_unaligned(slot_ptr as *mut u64, root.address);
        }
        self.current_root_table = slot_addr;
    }

    pub fn set_compute_root(&mut self, root: GpuPtr<u8>) {
        let (slot_addr, slot_ptr) = self.alloc_root_bytes(std::mem::size_of::<u64>());
        unsafe {
            std::ptr::write_unaligned(slot_ptr as *mut u64, root.address);
        }
        self.bind_root_table(slot_addr);
    }

    pub fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        let root_table = self.current_root_table;
        let topology = self.current_topology;
        self.bind_root_table(root_table);
        let encoder = self
            .render_encoder
            .as_ref()
            .expect("No active render encoder");
        unsafe {
            encoder.setArgumentTable_atStages(
                &self.argument_table,
                MTLRenderStages::Vertex | MTLRenderStages::Fragment,
            );
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
        let root_table = self.current_root_table;
        let topology = self.current_topology;
        self.bind_root_table(root_table);
        let encoder = self
            .render_encoder
            .as_ref()
            .expect("No active render encoder");
        unsafe {
            encoder.setArgumentTable_atStages(
                &self.argument_table,
                MTLRenderStages::Vertex | MTLRenderStages::Fragment,
            );
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
        let root_table = self.current_root_table;
        let topology = self.current_topology;
        self.bind_root_table(root_table);
        let encoder = self
            .render_encoder
            .as_ref()
            .expect("No active render encoder");
        unsafe {
            encoder.setArgumentTable_atStages(
                &self.argument_table,
                MTLRenderStages::Vertex | MTLRenderStages::Fragment,
            );
            encoder.drawPrimitives_indirectBuffer(topology, arg_addr_gpu);
        }
    }

    pub fn draw_indexed_indirect(&mut self, indices: GpuPtr<u8>, args: GpuPtr<u8>) {
        let index_addr_gpu: MTLGPUAddress = indices.address;
        // The GPU supplies the count, so bound the address range by the allocation remainder.
        let index_len = self.allocation_remaining(indices);
        let arg_addr_gpu: MTLGPUAddress = args.address;
        let root_table = self.current_root_table;
        let topology = self.current_topology;
        self.bind_root_table(root_table);
        let encoder = self
            .render_encoder
            .as_ref()
            .expect("No active render encoder");
        unsafe {
            encoder.setArgumentTable_atStages(
                &self.argument_table,
                MTLRenderStages::Vertex | MTLRenderStages::Fragment,
            );
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

        self.end_active_encoders();
        // Metal 4 routes buffer copies through the compute encoder.
        let encoder = self.begin_compute_encoder("memcpy");
        unsafe {
            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &src_buffer,
                src_offset as usize,
                &dst_buffer,
                dst_offset as usize,
                size as usize,
            );
        }
        encoder.endEncoding();
    }

    pub fn copy_buffer_to_texture(
        &mut self,
        texture_gpu: GpuPtr<u8>,
        src: GpuPtr<u8>,
        texture: &Texture,
    ) {
        let (mtl_texture, buffer, offset, size, origin, bytes_per_row, bytes_per_image) =
            self.prepare_texture_copy(texture_gpu, src, texture, "copy_buffer_to_texture");

        self.end_active_encoders();
        let encoder = self.begin_compute_encoder("copy to texture");
        unsafe {
            encoder.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                &buffer,
                offset as usize,
                bytes_per_row,
                bytes_per_image,
                size,
                &mtl_texture,
                0,
                0,
                origin,
            );
        }
        encoder.endEncoding();
    }

    pub fn copy_texture_to_buffer(
        &mut self,
        dst: GpuPtr<u8>,
        texture_gpu: GpuPtr<u8>,
        texture: &Texture,
    ) {
        let (mtl_texture, buffer, offset, size, origin, bytes_per_row, bytes_per_image) =
            self.prepare_texture_copy(texture_gpu, dst, texture, "copy_texture_to_buffer");

        self.end_active_encoders();
        let encoder = self.begin_compute_encoder("copy from texture");
        unsafe {
            encoder.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                &mtl_texture,
                0,
                0,
                origin,
                size,
                &buffer,
                offset as usize,
                bytes_per_row,
                bytes_per_image,
            );
        }
        encoder.endEncoding();
    }

    /// Validate the texture address, resolve the linear buffer, and compute the row /
    /// image strides for a buffer↔texture copy on the Metal 4 compute encoder.
    #[allow(clippy::type_complexity)]
    fn prepare_texture_copy(
        &self,
        texture_gpu: GpuPtr<u8>,
        buffer_gpu: GpuPtr<u8>,
        texture: &Texture,
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
        assert_eq!(
            texture_gpu,
            texture.gpu(),
            "{op} texture_gpu must match the address used to create the texture"
        );
        let mtl_texture = self.resolve_texture(texture.id());
        let width = texture.desc().width;
        let height = texture.desc().height;
        let bpp = bytes_per_pixel(texture.desc().format)
            .unwrap_or_else(|| panic!("Unsupported texture format for {op}"));
        let bytes_per_row = width as usize * bpp;
        let bytes_per_image = bytes_per_row * height as usize;
        let (buffer, offset) = self.resolve_buffer(buffer_gpu, bytes_per_image as u64);
        let size = MTLSize {
            width: width as usize,
            height: height as usize,
            depth: 1,
        };
        let origin = MTLOrigin { x: 0, y: 0, z: 0 };
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

    /// `gpuSetPipeline` for mesh pipelines — binds PSO, refreshes bindless heaps, binds argument table.
    pub fn set_meshlet_pipeline(&mut self, pso: &MeshletPso) {
        use objc2_metal::MTL4RenderCommandEncoder as _;

        let (pipeline, cull_mode, winding) = match &pso.inner {
            crate::pipeline::MeshletPsoInner::Metal(mtl_pso) => (
                mtl_pso.default_pipeline.clone(),
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

        self.refresh_argument_table();

        let encoder = self
            .render_encoder
            .as_ref()
            .expect("set_meshlet_pipeline: no active render encoder");
        encoder.setRenderPipelineState(&pipeline);
        encoder.setCullMode(cull_mode);
        encoder.setFrontFacingWinding(winding);
        encoder.setArgumentTable_atStages(
            &self.argument_table,
            MTLRenderStages::Mesh | MTLRenderStages::Fragment,
        );
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
        let root_table = self.current_root_table;
        self.bind_root_table(root_table);
        let Some(encoder) = self.render_encoder.as_ref() else {
            return;
        };
        encoder.setArgumentTable_atStages(
            &self.argument_table,
            MTLRenderStages::Mesh | MTLRenderStages::Fragment,
        );
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
        let root_table = self.current_root_table;
        self.bind_root_table(root_table);
        let Some(encoder) = self.render_encoder.as_ref() else {
            return;
        };
        encoder.setArgumentTable_atStages(
            &self.argument_table,
            MTLRenderStages::Mesh | MTLRenderStages::Fragment,
        );
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
            crate::accel::AccelInner::Metal(a) => {
                (a.acceleration_structure.clone(), a.scratch_buffer.clone())
            }
            #[allow(unreachable_patterns)]
            _ => unreachable!("acceleration structure belongs to another backend"),
        };
        let geometries = make_blas_geometry_descriptors(desc);

        self.add_accel_to_residency(&mtl_as);

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
                &mtl_as,
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
            crate::accel::AccelInner::Metal(a) => {
                (a.acceleration_structure.clone(), a.scratch_buffer.clone())
            }
            #[allow(unreachable_patterns)]
            _ => unreachable!("acceleration structure belongs to another backend"),
        };
        self.add_accel_to_residency(&mtl_as);

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
                &mtl_as,
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

fn make_stencil_descriptor(
    desc: &crate::pipeline::StencilDesc,
    read_mask: u8,
    write_mask: u8,
) -> objc2::rc::Retained<objc2_metal::MTLStencilDescriptor> {
    let s = objc2_metal::MTLStencilDescriptor::new();
    s.setStencilCompareFunction(compare_op_to_mtl(desc.test));
    s.setStencilFailureOperation(stencil_op_to_mtl(desc.fail_op));
    s.setDepthFailureOperation(stencil_op_to_mtl(desc.depth_fail_op));
    s.setDepthStencilPassOperation(stencil_op_to_mtl(desc.pass_op));
    s.setReadMask(read_mask as u32);
    s.setWriteMask(write_mask as u32);
    s
}

fn stencil_op_to_mtl(op: StencilOp) -> MTLStencilOperation {
    match op {
        StencilOp::Keep => MTLStencilOperation::Keep,
        StencilOp::Zero => MTLStencilOperation::Zero,
        StencilOp::Replace => MTLStencilOperation::Replace,
        StencilOp::IncrementClamp => MTLStencilOperation::IncrementClamp,
        StencilOp::DecrementClamp => MTLStencilOperation::DecrementClamp,
        StencilOp::Invert => MTLStencilOperation::Invert,
        StencilOp::IncrementWrap => MTLStencilOperation::IncrementWrap,
        StencilOp::DecrementWrap => MTLStencilOperation::DecrementWrap,
    }
}

fn compare_op_to_mtl(op: CompareOp) -> objc2_metal::MTLCompareFunction {
    match op {
        CompareOp::Never => objc2_metal::MTLCompareFunction::Never,
        CompareOp::Less => objc2_metal::MTLCompareFunction::Less,
        CompareOp::Equal => objc2_metal::MTLCompareFunction::Equal,
        CompareOp::LessOrEqual => objc2_metal::MTLCompareFunction::LessEqual,
        CompareOp::Greater => objc2_metal::MTLCompareFunction::Greater,
        CompareOp::NotEqual => objc2_metal::MTLCompareFunction::NotEqual,
        CompareOp::GreaterOrEqual => objc2_metal::MTLCompareFunction::GreaterEqual,
        CompareOp::Always => objc2_metal::MTLCompareFunction::Always,
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
