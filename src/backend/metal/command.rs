use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer,
    MTL4CommandEncoder, MTL4ComputeCommandEncoder, MTL4RenderCommandEncoder,
    MTL4RenderPassDescriptor, MTL4VisibilityOptions, MTLAllocation, MTLBuffer,
    MTLComputePipelineState, MTLDepthStencilState, MTLDevice, MTLGPUAddress,
    MTLIndexType, MTLIndirectCommandBuffer, MTLIndirectCommandBufferDescriptor,
    MTLIndirectCommandType, MTLLoadAction, MTLOrigin, MTLPrimitiveType, MTLRenderPipelineState,
    MTLRenderStages, MTLResidencySet, MTLResourceOptions, MTLSamplerState, MTLScissorRect, MTLSize,
    MTLStages, MTLStencilOperation, MTLStoreAction, MTLTexture, MTLViewport,
};

use crate::barrier::{HazardFlags, StageFlags};
use crate::command::{
    DrawIndirectMultiArgs, LoadOp, RenderPassDesc, RenderTarget,
    SignalValueDesc, StoreOp, WaitValueDesc,
};
use crate::pipeline::{
    BlendState, ComputePso, DepthStencilState, GraphicsPso, MeshletPso,
};
use crate::texture::{Texture, bytes_per_pixel};
use crate::types::*;

use super::device::{SharedAllocations, SharedSamplers, SharedTextures};

const ROOT_TABLE_BYTES: usize = 32;
const ROOT_TABLE_RING_ENTRIES: usize = 65_536;
const ROOT_TABLE_RING_BYTES: usize = ROOT_TABLE_BYTES * ROOT_TABLE_RING_ENTRIES;
const METAL_BINDLESS_TEXTURE_CAPACITY: usize = 65_536;
const METAL_BINDLESS_SAMPLER_CAPACITY: usize = 256;
const MDI_ICB_THREADGROUP_SIZE: usize = 64;

#[derive(Clone)]
struct MetalPipelineBinding {
    pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    cull_mode: objc2_metal::MTLCullMode,
    winding: objc2_metal::MTLWinding,
    texture_heap_slot: bool,
    sampler_heap_slot: bool,
    stages: MTLRenderStages,
}

#[allow(dead_code)]
struct GeneratedMdiIcb {
    icb: Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>,
    range_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    max_draw_count_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    primitive_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// 8-byte argument buffer holding the ICB's gpuResourceID, bound at buffer slot 3 so
    /// the encode kernel can reach the ICB through its `RhiIcbContainer` argument struct.
    icb_arg_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
}

#[derive(Clone, Copy)]
struct PendingQueueBarrier {
    after_queue_stages: MTLStages,
    before_stages: MTLStages,
    visibility: MTL4VisibilityOptions,
}

pub struct MetalCommandBuffer {
    pub(crate) command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    #[allow(dead_code)]
    pub(crate) command_allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    /// Active render encoder (created by begin_render_pass, consumed by end_render_pass).
    pub(crate) render_encoder: Option<Retained<ProtocolObject<dyn MTL4RenderCommandEncoder>>>,
    /// Active compute encoder (created by set_compute_pipeline, consumed by end or encoder switch).
    pub(crate) compute_encoder: Option<Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>>,
    /// Drawable texture for rendering to swapchain.
    pub(crate) drawable_texture: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
    /// Depth texture from swapchain.
    pub(crate) depth_texture: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
    /// Current primitive topology (set by set_graphics_pipeline).
    current_topology: MTLPrimitiveType,
    /// Device reference for depth stencil state creation.
    pub(crate) device: Retained<ProtocolObject<dyn MTLDevice>>,
    /// Argument table used for buffer bindings (root + argument buffers).
    argument_table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    /// CPU-updated root table ring buffer.
    #[allow(dead_code)]
    root_table_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    root_table_ptr: *mut u8,
    root_table_gpu_base: MTLGPUAddress,
    root_table_cursor: usize,
    root_table_capacity: usize,
    /// Bindless texture heap: flat array of MTLTexture gpuResourceIDs (u64 each), indexed by `TextureId`.
    /// Per `NoGraphicsApi.md` line 266, Metal 4 textures expose a 64-bit `gpuResourceID`
    /// that goes straight into shader-visible memory — no argument-buffer indirection.
    texture_heap_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Bindless sampler heap: flat array of MTLSamplerState gpuResourceIDs (u64 each), indexed by `SamplerId`.
    sampler_heap_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Shared texture list for heap population.
    pub(crate) textures: SharedTextures,
    /// Shared sampler list for heap population.
    pub(crate) samplers: SharedSamplers,
    /// Shared buffer allocation registry for GPU pointer resolution.
    pub(crate) allocations: SharedAllocations,
    /// Residency set for command-local allocations.
    residency_set: Retained<ProtocolObject<dyn MTLResidencySet>>,
    /// Dirty bit for queue residency commits.
    residency_dirty: std::rc::Rc<std::cell::Cell<bool>>,
    /// Keep depth-stencil state objects alive for Metal 4 command lifetime.
    depth_stencil_states: Vec<Retained<ProtocolObject<dyn MTLDepthStencilState>>>,
    /// Active blend state used for pipeline selection.
    pub(crate) current_blend_state: BlendState,
    /// Cached threads-per-threadgroup from the bound compute pipeline.
    current_threads_per_threadgroup: [u32; 3],
    /// Threads per object threadgroup for the bound meshlet pipeline.
    current_mesh_tpg_object: MTLSize,
    /// Threads per mesh threadgroup for the bound meshlet pipeline.
    current_mesh_tpg_mesh: MTLSize,
    /// Root-constant payload size for the currently bound pipeline.
    root_constant_size: u32,
    /// Queue barrier to apply on the next encoder begin.
    pending_queue_barrier: Option<PendingQueueBarrier>,
    /// Pending split barrier producer state.
    pending_split_barrier: Option<(StageFlags, HazardFlags)>,
    /// Pending value waits consumed at queue submit.
    pub(crate) pending_value_waits: Vec<WaitValueDesc>,
    /// Pending value signals consumed at queue submit.
    pub(crate) pending_value_signals: Vec<SignalValueDesc>,
    active_texture_heap_slot_enabled: bool,
    active_sampler_heap_slot_enabled: bool,
    active_texture_heap_ptr_override: Option<MTLGPUAddress>,
    /// Render pass description, kept while a pass is open so the encoder can be reopened
    /// (with Load actions) when an MDI draw splits the pass for GPU-side ICB generation.
    render_pass_desc: Option<RenderPassDesc>,
    current_root_table: MTLGPUAddress,
    /// Render state tracked so it can be re-applied to a fresh encoder after an MDI split.
    current_pipeline: Option<MetalPipelineBinding>,
    current_depth_stencil: Option<(
        Retained<ProtocolObject<dyn MTLDepthStencilState>>,
        Option<(f32, f32, f32)>,
    )>,
    current_viewport: Option<MTLViewport>,
    current_scissor: Option<MTLScissorRect>,
    mdi_icb_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    mdi_icb_resources: Vec<GeneratedMdiIcb>,
}

impl MetalCommandBuffer {
    fn resolve_buffer(
        &self,
        addr: GpuAddress,
        size: u64,
    ) -> (Retained<ProtocolObject<dyn MTLBuffer>>, u64) {
        let addr_u64 = addr.0;
        let allocations = self.allocations.borrow();
        if let Some((&base, alloc)) = allocations.range(..=addr_u64).next_back()
            && addr_u64 + size <= base + alloc.size
        {
            return (alloc.buffer.clone(), addr_u64 - base);
        }
        panic!("GPU address {addr_u64:#x} not found in allocation registry");
    }

    /// Bytes available from `addr` to the end of its allocation. Used only where the
    /// element count is GPU-driven (indirect indexed draws, MDI capacity) and the CPU
    /// cannot compute an exact length. Metal 4 consumes the GPU address directly, so
    /// non-indirect draws pass their addresses through without any lookup.
    fn allocation_remaining(&self, addr: GpuAddress) -> u64 {
        let addr_u64 = addr.0;
        let allocations = self.allocations.borrow();
        if let Some((&base, alloc)) = allocations.range(..=addr_u64).next_back()
            && addr_u64 < base + alloc.size
        {
            return base + alloc.size - addr_u64;
        }
        panic!("GPU address {addr_u64:#x} not found in allocation registry");
    }

    fn resolve_texture(&self, id: TextureId) -> Retained<ProtocolObject<dyn MTLTexture>> {
        let textures = self.textures.borrow();
        textures
            .get(id.0 as usize)
            .and_then(|t| t.as_ref())
            .expect("Invalid texture ID")
            .clone()
    }

    fn refresh_argument_table(&mut self) {
        unsafe {
            self.argument_table.setAddress_atIndex(0, 0);
            let tex_addr = if self.active_texture_heap_slot_enabled {
                self.active_texture_heap_ptr_override
                    .or_else(|| self.texture_heap_buffer.as_ref().map(|b| b.gpuAddress()))
                    .unwrap_or(0)
            } else {
                0
            };
            self.argument_table.setAddress_atIndex(tex_addr, 1);
            let sampler_addr = if self.active_sampler_heap_slot_enabled {
                self.sampler_heap_buffer
                    .as_ref()
                    .map(|b| b.gpuAddress())
                    .unwrap_or(0)
            } else {
                0
            };
            self.argument_table.setAddress_atIndex(sampler_addr, 2);
        }
    }

    fn add_allocation_to_residency(&self, buffer: &ProtocolObject<dyn MTLBuffer>) {
        let allocation = unsafe {
            &*(buffer as *const ProtocolObject<dyn MTLBuffer>
                as *const ProtocolObject<dyn MTLAllocation>)
        };
        self.residency_set.addAllocation(allocation);
        self.residency_dirty.set(true);
    }

    fn remove_allocation_from_residency(&self, buffer: &ProtocolObject<dyn MTLBuffer>) {
        let allocation = unsafe {
            &*(buffer as *const ProtocolObject<dyn MTLBuffer>
                as *const ProtocolObject<dyn MTLAllocation>)
        };
        self.residency_set.removeAllocation(allocation);
        self.residency_dirty.set(true);
    }

    fn make_command_buffer_resource(
        &self,
        byte_len: usize,
        options: MTLResourceOptions,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let buffer = self
            .device
            .newBufferWithLength_options(byte_len, options)
            .expect("Failed to allocate Metal command-local buffer");
        self.add_allocation_to_residency(buffer.as_ref());
        buffer
    }

    fn primitive_id(topology: MTLPrimitiveType) -> u32 {
        if topology == MTLPrimitiveType::Point {
            0
        } else if topology == MTLPrimitiveType::Line {
            1
        } else if topology == MTLPrimitiveType::LineStrip {
            2
        } else if topology == MTLPrimitiveType::TriangleStrip {
            4
        } else {
            3
        }
    }

    fn generate_one_mdi_icb(
        &mut self,
        encoder: &ProtocolObject<dyn MTL4ComputeCommandEncoder>,
        topology: MTLPrimitiveType,
        args: GpuAddress,
        draw_count: GpuAddress,
        max_draw_count: u32,
    ) -> GeneratedMdiIcb {
        let args_addr: MTLGPUAddress = args.0;
        let draw_count_addr: MTLGPUAddress = draw_count.0;

        let icb_desc = MTLIndirectCommandBufferDescriptor::new();
        icb_desc.setCommandTypes(MTLIndirectCommandType::Draw);
        icb_desc.setInheritPipelineState(true);
        icb_desc.setInheritBuffers(true);
        icb_desc.setInheritDepthStencilState(true);
        icb_desc.setInheritDepthBias(true);
        icb_desc.setInheritDepthClipMode(true);
        icb_desc.setInheritCullMode(true);
        icb_desc.setInheritFrontFacingWinding(true);
        icb_desc.setInheritTriangleFillMode(true);
        icb_desc.setMaxVertexBufferBindCount(0);
        icb_desc.setMaxFragmentBufferBindCount(0);

        let icb = unsafe {
            self.device
                .newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                    &icb_desc,
                    max_draw_count as usize,
                    MTLResourceOptions::StorageModePrivate,
                )
        }
        .expect("Failed to create Metal ICB for MDI");
        let icb_allocation = unsafe {
            &*(icb.as_ref() as *const ProtocolObject<dyn MTLIndirectCommandBuffer>
                as *const ProtocolObject<dyn MTLAllocation>)
        };
        self.residency_set.addAllocation(icb_allocation);
        self.residency_dirty.set(true);

        let range_buffer = self.make_command_buffer_resource(
            std::mem::size_of::<objc2_metal::MTLIndirectCommandBufferExecutionRange>(),
            MTLResourceOptions::StorageModeShared,
        );
        let max_draw_count_buffer = self.make_command_buffer_resource(
            std::mem::size_of::<u32>(),
            MTLResourceOptions::StorageModeShared,
        );
        let primitive_buffer = self.make_command_buffer_resource(
            std::mem::size_of::<u32>(),
            MTLResourceOptions::StorageModeShared,
        );
        // Argument buffer for the ICB: a single `command_buffer` handle at offset 0.
        let icb_arg_buffer = self.make_command_buffer_resource(
            std::mem::size_of::<u64>(),
            MTLResourceOptions::StorageModeShared,
        );

        // MTLResourceID is a transparent wrapper over a u64 GPU handle.
        let icb_resource_id: u64 = unsafe { std::mem::transmute(icb.gpuResourceID()) };

        unsafe {
            std::ptr::write_unaligned(
                max_draw_count_buffer.contents().as_ptr() as *mut u32,
                max_draw_count,
            );
            std::ptr::write_unaligned(
                primitive_buffer.contents().as_ptr() as *mut u32,
                Self::primitive_id(topology),
            );
            std::ptr::write_unaligned(
                icb_arg_buffer.contents().as_ptr() as *mut u64,
                icb_resource_id,
            );
        }

        unsafe {
            encoder.resetCommandsInBuffer_withRange(
                &icb,
                NSRange::new(0, max_draw_count as usize),
            );
            self.argument_table.setAddress_atIndex(args_addr, 0);
            self.argument_table.setAddress_atIndex(draw_count_addr, 1);
            self.argument_table
                .setAddress_atIndex(range_buffer.gpuAddress(), 2);
            // Bind the ICB through its argument buffer (a `command_buffer` handle at id 0).
            // MSL does not allow `command_buffer` as a direct buffer-slot parameter.
            self.argument_table
                .setAddress_atIndex(icb_arg_buffer.gpuAddress(), 3);
            self.argument_table
                .setAddress_atIndex(max_draw_count_buffer.gpuAddress(), 4);
            self.argument_table
                .setAddress_atIndex(primitive_buffer.gpuAddress(), 5);
            encoder.setArgumentTable(Some(&self.argument_table));
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: max_draw_count as usize,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: MDI_ICB_THREADGROUP_SIZE,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.optimizeIndirectCommandBuffer_withRange(
                &icb,
                NSRange::new(0, max_draw_count as usize),
            );
        }

        GeneratedMdiIcb {
            icb,
            range_buffer,
            max_draw_count_buffer,
            primitive_buffer,
            icb_arg_buffer,
        }
    }

    /// Ensure `slot` holds a shared-storage MTLBuffer of at least `required_len` bytes,
    /// reallocating (and re-registering with the residency set) if needed.
    fn ensure_heap_buffer(
        slot: &mut Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
        device: &ProtocolObject<dyn MTLDevice>,
        residency_set: &ProtocolObject<dyn MTLResidencySet>,
        residency_dirty: &std::rc::Rc<std::cell::Cell<bool>>,
        required_len: usize,
        label: &'static str,
    ) {
        if slot.as_ref().is_some_and(|b| b.length() >= required_len) {
            return;
        }
        if let Some(old) = slot.take() {
            let old_alloc = unsafe {
                &*(old.as_ref() as *const ProtocolObject<dyn MTLBuffer>
                    as *const ProtocolObject<dyn MTLAllocation>)
            };
            residency_set.removeAllocation(old_alloc);
            residency_dirty.set(true);
        }
        let new_buf = device
            .newBufferWithLength_options(required_len, MTLResourceOptions::StorageModeShared)
            .unwrap_or_else(|| panic!("Failed to allocate Metal {label} argument buffer"));
        let new_alloc = unsafe {
            &*(new_buf.as_ref() as *const ProtocolObject<dyn MTLBuffer>
                as *const ProtocolObject<dyn MTLAllocation>)
        };
        residency_set.addAllocation(new_alloc);
        residency_dirty.set(true);
        *slot = Some(new_buf);
    }

    /// Refresh the bindless texture and/or sampler heaps. The heaps are flat `[u64]` arrays
    /// of `gpuResourceID`s, indexed by `TextureId` / `SamplerId`. The shader binds them as
    /// `device const ulong* heap [[buffer(N)]]` and casts the loaded `ulong` into the
    /// appropriate `texture<...>` / `sampler` handle using Metal 4 syntax.
    fn refresh_bindless_heaps(&mut self, refresh_textures: bool, refresh_samplers: bool) {
        if refresh_textures {
            let textures = self.textures.borrow();
            assert!(
                textures.len() <= METAL_BINDLESS_TEXTURE_CAPACITY,
                "Metal texture heap overflow: {} textures exceed capacity {}",
                textures.len(),
                METAL_BINDLESS_TEXTURE_CAPACITY,
            );
            Self::ensure_heap_buffer(
                &mut self.texture_heap_buffer,
                &self.device,
                &self.residency_set,
                &self.residency_dirty,
                METAL_BINDLESS_TEXTURE_CAPACITY * std::mem::size_of::<u64>(),
                "texture heap",
            );
            let dst = self
                .texture_heap_buffer
                .as_ref()
                .expect("texture heap buffer must exist after allocation")
                .contents()
                .as_ptr() as *mut u64;
            for (i, tex_opt) in textures.iter().enumerate() {
                let id = tex_opt
                    .as_deref()
                    .map(|t| t.gpuResourceID().to_raw())
                    .unwrap_or(0);
                unsafe { std::ptr::write_unaligned(dst.add(i), id) };
            }
        }

        if refresh_samplers {
            let samplers = self.samplers.borrow();
            assert!(
                samplers.len() <= METAL_BINDLESS_SAMPLER_CAPACITY,
                "Metal sampler heap overflow: {} samplers exceed capacity {}",
                samplers.len(),
                METAL_BINDLESS_SAMPLER_CAPACITY,
            );
            Self::ensure_heap_buffer(
                &mut self.sampler_heap_buffer,
                &self.device,
                &self.residency_set,
                &self.residency_dirty,
                METAL_BINDLESS_SAMPLER_CAPACITY * std::mem::size_of::<u64>(),
                "sampler heap",
            );
            let dst = self
                .sampler_heap_buffer
                .as_ref()
                .expect("sampler heap buffer must exist after allocation")
                .contents()
                .as_ptr() as *mut u64;
            for (i, sampler) in samplers.iter().enumerate() {
                let id = sampler.gpuResourceID().to_raw();
                unsafe { std::ptr::write_unaligned(dst.add(i), id) };
            }
        }
    }

    fn alloc_root_bytes(&mut self, size: usize) -> (MTLGPUAddress, *mut u8) {
        let size = (size + 15) & !15;
        if self.root_table_cursor + size > self.root_table_capacity {
            panic!(
                "Metal root table ring overflow ({} bytes). Increase ROOT_TABLE_RING_ENTRIES.",
                self.root_table_capacity
            );
        }
        let offset = self.root_table_cursor;
        self.root_table_cursor += size;
        let ptr = unsafe { self.root_table_ptr.add(offset) };
        let addr = self.root_table_gpu_base + offset as u64;
        (addr, ptr)
    }

    fn alloc_root_table_slot(&mut self) -> (MTLGPUAddress, *mut u8) {
        self.alloc_root_bytes(ROOT_TABLE_BYTES)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
        command_allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        residency_set: Retained<ProtocolObject<dyn objc2_metal::MTLResidencySet>>,
        residency_dirty: std::rc::Rc<std::cell::Cell<bool>>,
        textures: SharedTextures,
        samplers: SharedSamplers,
        allocations: SharedAllocations,
        mdi_icb_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    ) -> crate::error::RhiResult<Self> {
        command_buffer.beginCommandBufferWithAllocator(&command_allocator);
        command_buffer.useResidencySet(&residency_set);

        let root_table_buffer = device
            .newBufferWithLength_options(
                ROOT_TABLE_RING_BYTES,
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or_else(|| {
                crate::error::RhiError::CommandBuffer(
                    "Failed to allocate Metal root table ring buffer".into(),
                )
            })?;

        // Register for residency.
        let allocation = unsafe {
            &*(root_table_buffer.as_ref() as *const ProtocolObject<dyn MTLBuffer>
                as *const ProtocolObject<dyn MTLAllocation>)
        };
        residency_set.addAllocation(allocation);
        residency_dirty.set(true);

        let desc = MTL4ArgumentTableDescriptor::new();
        desc.setMaxBufferBindCount(6);
        desc.setMaxTextureBindCount(0);
        desc.setMaxSamplerStateBindCount(0);
        desc.setInitializeBindings(true);
        desc.setSupportAttributeStrides(false);

        let argument_table = device
            .newArgumentTableWithDescriptor_error(&desc)
            .map_err(|e| {
                crate::error::RhiError::CommandBuffer(format!(
                    "Failed to create Metal 4 argument table: {e}"
                ))
            })?;

        let mut cmd = Self {
            command_buffer,
            command_allocator,
            render_encoder: None,
            compute_encoder: None,
            drawable_texture: None,
            depth_texture: None,
            current_topology: MTLPrimitiveType::Triangle,
            device,
            argument_table,
            root_table_ptr: root_table_buffer.contents().as_ptr() as *mut u8,
            root_table_gpu_base: root_table_buffer.gpuAddress(),
            root_table_buffer,
            root_table_cursor: 0,
            root_table_capacity: ROOT_TABLE_RING_BYTES,
            texture_heap_buffer: None,
            sampler_heap_buffer: None,
            textures,
            samplers,
            allocations,
            residency_set,
            residency_dirty,
            depth_stencil_states: Vec::new(),
            current_blend_state: BlendState::default(),
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
            root_constant_size: 0,
            pending_queue_barrier: None,
            pending_split_barrier: None,
            pending_value_waits: Vec::new(),
            pending_value_signals: Vec::new(),
            active_texture_heap_slot_enabled: false,
            active_sampler_heap_slot_enabled: false,
            active_texture_heap_ptr_override: None,
            render_pass_desc: None,
            current_root_table: 0,
            current_pipeline: None,
            current_depth_stencil: None,
            current_viewport: None,
            current_scissor: None,
            mdi_icb_pipeline,
            mdi_icb_resources: Vec::new(),
        };

        cmd.refresh_argument_table();
        Ok(cmd)
    }

    pub fn begin_render_pass(&mut self, desc: &RenderPassDesc) {
        self.end_active_encoders();
        // Open the render encoder eagerly and encode draws directly into it. The only
        // case that needs a buffered/replayed command list is MDI (see draw_indirect_multi),
        // which splits the pass on demand because its ICB must be generated by a compute
        // pass that precedes the render encoder.
        let encoder = self.begin_metal_render_encoder(desc, false);
        self.render_encoder = Some(encoder);
        self.render_pass_desc = Some(desc.clone());
        self.current_pipeline = None;
        self.current_depth_stencil = None;
        self.current_viewport = None;
        self.current_scissor = None;
    }

    /// Build a Metal render command encoder for `desc`. When `force_load` is set, every
    /// attachment's load action is forced to Load (preserving prior contents) — used when
    /// reopening the encoder after an MDI split so the already-rendered pixels survive.
    fn begin_metal_render_encoder(
        &mut self,
        desc: &RenderPassDesc,
        force_load: bool,
    ) -> Retained<ProtocolObject<dyn MTL4RenderCommandEncoder>> {
        let pass_desc = MTL4RenderPassDescriptor::new();

        // Configure color attachments
        let color_attachments = pass_desc.colorAttachments();
        for (i, color_att) in desc.color_attachments.iter().enumerate() {
            let attachment = unsafe { color_attachments.objectAtIndexedSubscript(i) };

            // Set texture based on render target
            match &color_att.target {
                RenderTarget::SwapchainImage(_idx) => {
                    if let Some(tex) = &self.drawable_texture {
                        attachment.setTexture(Some(tex));
                    }
                }
                RenderTarget::Texture(id) => {
                    let tex = self.resolve_texture(*id);
                    attachment.setTexture(Some(&tex));
                }
            }

            attachment.setLoadAction(if force_load {
                MTLLoadAction::Load
            } else {
                match color_att.load_op {
                    LoadOp::Clear => MTLLoadAction::Clear,
                    LoadOp::Load => MTLLoadAction::Load,
                    LoadOp::DontCare => MTLLoadAction::DontCare,
                }
            });

            attachment.setStoreAction(match color_att.store_op {
                StoreOp::Store => MTLStoreAction::Store,
                StoreOp::DontCare => MTLStoreAction::DontCare,
            });

            if !force_load && color_att.load_op == LoadOp::Clear {
                let c = color_att.clear_color;
                attachment.setClearColor(objc2_metal::MTLClearColor {
                    red: c[0] as f64,
                    green: c[1] as f64,
                    blue: c[2] as f64,
                    alpha: c[3] as f64,
                });
            }
        }

        // Configure depth attachment
        if let Some(depth_att) = &desc.depth_attachment {
            let depth = pass_desc.depthAttachment();

            match &depth_att.target {
                RenderTarget::SwapchainImage(_) => {
                    if let Some(depth_tex) = &self.depth_texture {
                        depth.setTexture(Some(depth_tex));
                    }
                }
                RenderTarget::Texture(id) => {
                    let tex = self.resolve_texture(*id);
                    depth.setTexture(Some(&tex));
                }
            }

            depth.setLoadAction(if force_load {
                MTLLoadAction::Load
            } else {
                match depth_att.load_op {
                    LoadOp::Clear => MTLLoadAction::Clear,
                    LoadOp::Load => MTLLoadAction::Load,
                    LoadOp::DontCare => MTLLoadAction::DontCare,
                }
            });

            depth.setStoreAction(match depth_att.store_op {
                StoreOp::Store => MTLStoreAction::Store,
                StoreOp::DontCare => MTLStoreAction::DontCare,
            });

            if !force_load && depth_att.load_op == LoadOp::Clear {
                depth.setClearDepth(depth_att.clear_depth as f64);
            }
        }

        // Create the render command encoder
        let encoder = self
            .command_buffer
            .renderCommandEncoderWithDescriptor(&pass_desc)
            .expect("Failed to create Metal render command encoder");

        self.apply_pending_queue_barrier_render(&encoder);

        encoder
    }

    pub fn end_render_pass(&mut self) {
        if let Some(encoder) = self.render_encoder.take() {
            encoder.endEncoding();
        }
        self.render_pass_desc = None;
    }

    /// Re-apply the tracked render state to a freshly-opened encoder. Used after an MDI
    /// split, where the new encoder starts with no pipeline/depth/viewport/scissor bound.
    fn reapply_render_state(&mut self, encoder: &ProtocolObject<dyn MTL4RenderCommandEncoder>) {
        if let Some(binding) = self.current_pipeline.clone() {
            encoder.setRenderPipelineState(&binding.pipeline);
            encoder.setCullMode(binding.cull_mode);
            encoder.setFrontFacingWinding(binding.winding);
            self.active_texture_heap_slot_enabled = binding.texture_heap_slot;
            self.active_sampler_heap_slot_enabled = binding.sampler_heap_slot;
            self.refresh_argument_table();
            encoder.setArgumentTable_atStages(&self.argument_table, binding.stages);
        }
        if let Some((ds_state, depth_bias)) = self.current_depth_stencil.clone() {
            encoder.setDepthStencilState(Some(&ds_state));
            if let Some((bias, slope, clamp)) = depth_bias {
                encoder.setDepthBias_slopeScale_clamp(bias, slope, clamp);
            }
        }
        if let Some(viewport) = self.current_viewport {
            encoder.setViewport(viewport);
        }
        if let Some(scissor) = self.current_scissor {
            encoder.setScissorRect(scissor);
        }
    }

    pub fn set_graphics_pipeline(&mut self, pso: &GraphicsPso) {
        let (
            pipeline,
            cull_mode,
            winding,
            topology,
            root_constant_size,
            has_texture_heap_slot,
            has_sampler_heap_slot,
        ) = match &pso.inner {
            crate::pipeline::GraphicsPsoInner::Metal(mtl_pso) => {
                let pipeline = mtl_pso.pipeline_for_blend(&self.current_blend_state);
                let has_slot = |slot: usize| mtl_pso.graphics_argument_buffer_slots.contains(&slot);
                (
                    pipeline,
                    mtl_pso.cull_mode,
                    mtl_pso.winding,
                    mtl_pso.topology,
                    mtl_pso.root_constant_size,
                    has_slot(1),
                    has_slot(2),
                )
            }
            #[allow(unreachable_patterns)]
            _ => unreachable!("wrong backend"),
        };
        self.current_topology = topology;
        self.root_constant_size = root_constant_size;
        self.active_texture_heap_slot_enabled = has_texture_heap_slot;
        self.active_sampler_heap_slot_enabled = has_sampler_heap_slot;
        self.refresh_bindless_heaps(has_texture_heap_slot, has_sampler_heap_slot);
        self.refresh_argument_table();
        let binding = MetalPipelineBinding {
            pipeline: pipeline.clone(),
            cull_mode,
            winding,
            texture_heap_slot: has_texture_heap_slot,
            sampler_heap_slot: has_sampler_heap_slot,
            stages: MTLRenderStages::Vertex | MTLRenderStages::Fragment,
        };
        self.current_pipeline = Some(binding.clone());
        let encoder = self.render_encoder.as_ref().expect("No active render encoder");
        encoder.setRenderPipelineState(&pipeline);
        encoder.setCullMode(cull_mode);
        encoder.setFrontFacingWinding(winding);
        encoder.setArgumentTable_atStages(&self.argument_table, binding.stages);
    }

    pub fn set_compute_pipeline(&mut self, pso: &ComputePso) {
        let mtl_pso = match &pso.inner {
            crate::pipeline::ComputePsoInner::Metal(p) => p,
            #[allow(unreachable_patterns)]
            _ => unreachable!("wrong backend"),
        };

        self.end_active_encoders();
        let encoder = self
            .command_buffer
            .computeCommandEncoder()
            .expect("Failed to create Metal compute command encoder");
        self.apply_pending_queue_barrier_compute(&encoder);
        encoder.setComputePipelineState(&mtl_pso.pipeline);

        self.current_threads_per_threadgroup = mtl_pso.threads_per_threadgroup;
        self.root_constant_size = mtl_pso.root_constant_size;
        let has_slot = |slot: usize| mtl_pso.compute_argument_buffer_slots.contains(&slot);
        self.active_texture_heap_slot_enabled = has_slot(1);
        self.active_sampler_heap_slot_enabled = has_slot(2);
        self.refresh_bindless_heaps(has_slot(1), has_slot(2));
        self.refresh_argument_table();
        encoder.setArgumentTable(Some(&self.argument_table));

        self.compute_encoder = Some(encoder);
    }

    pub fn set_depth_stencil_state(&mut self, state: &DepthStencilState) {
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

        if let Some(ds_state) = self.device.newDepthStencilStateWithDescriptor(&ds_desc) {
            let depth_bias = if state.depth_bias != 0.0 || state.depth_bias_slope_factor != 0.0 {
                Some((
                    state.depth_bias,
                    state.depth_bias_slope_factor,
                    state.depth_bias_clamp,
                ))
            } else {
                None
            };
            let encoder = self
                .render_encoder
                .as_ref()
                .expect("No active render encoder");
            encoder.setDepthStencilState(Some(&ds_state));
            if let Some((bias, slope, clamp)) = depth_bias {
                encoder.setDepthBias_slopeScale_clamp(bias, slope, clamp);
            }
            self.current_depth_stencil = Some((ds_state.clone(), depth_bias));
            self.depth_stencil_states.push(ds_state);
        }
    }

    pub fn set_blend_state(&mut self, state: &BlendState) {
        // Blend state is baked into the pipeline in Metal.
        // Dynamic blend is limited to blend constants.
        self.current_blend_state = state.clone();
    }

    pub fn set_root_data(&mut self, vertex_root: GpuAddress, pixel_root: GpuAddress) {
        // Regular draws: Slang lowers each stage's `uniform T*` root to buffer(0) -> { T* }.
        // Stash the root pointer in a holder slot and bind the slot at buffer(0) for both
        // stages (same one-level indirection as set_compute_root). Vertex and fragment
        // share a single argument table, so this serves a *shared* root — pass the same
        // pointer for both stages, which the two-pointer model explicitly allows. Wholly
        // independent vertex/pixel roots would need a per-stage argument table (future).
        //
        // MDI draws bypass this and call set_root_table directly, keeping the strided
        // ROOT_TABLE layout their GPU-side draw generation depends on.
        debug_assert!(
            self.render_encoder.is_some(),
            "set_root_data called outside a render pass"
        );
        let root = if vertex_root.0 != 0 { vertex_root } else { pixel_root };
        let (slot_addr, slot_ptr) = self.alloc_root_bytes(std::mem::size_of::<u64>());
        unsafe {
            std::ptr::write_unaligned(slot_ptr as *mut u64, root.0);
        }
        self.current_root_table = slot_addr;
    }

    fn set_root_table(
        &mut self,
        vertex_root_base: GpuAddress,
        vertex_stride: u32,
        pixel_root_base: GpuAddress,
        pixel_stride: u32,
    ) {
        debug_assert!(
            self.render_encoder.is_some(),
            "set_root_table called outside a render pass"
        );
        if self.root_constant_size < ROOT_TABLE_BYTES as u32 {
            panic!(
                "Root table ({} bytes) exceeds pipeline limit ({} bytes)",
                ROOT_TABLE_BYTES, self.root_constant_size
            );
        }

        let mut bytes = [0u8; ROOT_TABLE_BYTES];
        bytes[0..8].copy_from_slice(&vertex_root_base.0.to_ne_bytes());
        bytes[8..12].copy_from_slice(&vertex_stride.to_ne_bytes());
        bytes[16..24].copy_from_slice(&pixel_root_base.0.to_ne_bytes());
        bytes[24..28].copy_from_slice(&pixel_stride.to_ne_bytes());

        let (addr, ptr) = self.alloc_root_table_slot();
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
            self.argument_table.setAddress_atIndex(addr, 0);
        }
        self.current_root_table = addr;
    }

    pub fn set_compute_root(&mut self, root: GpuAddress) {
        debug_assert!(
            self.compute_encoder.is_some(),
            "set_compute_root called outside a compute pass"
        );
        let root_bytes = std::mem::size_of::<GpuAddress>() as u32;
        if self.root_constant_size < root_bytes {
            panic!(
                "Compute root pointer ({} bytes) exceeds pipeline limit ({} bytes)",
                root_bytes, self.root_constant_size
            );
        }
        // Slang lowers a shader root pointer to a Metal argument-buffer slot that *contains*
        // the pointer: buffer(0) -> { T* inner }. Vulkan instead pushes the pointer inline,
        // so it derefs once to reach the data. To make one Slang source work on both, we
        // add the matching indirection on Metal: stash the root pointer in a ring slot and
        // bind the slot's address at buffer(0), so the shader's `inner` resolves to `root`.
        let (slot_addr, slot_ptr) = self.alloc_root_bytes(std::mem::size_of::<u64>());
        unsafe {
            std::ptr::write_unaligned(slot_ptr as *mut u64, root.0);
            self.argument_table.setAddress_atIndex(slot_addr, 0);
        }
    }

    /// Bind a (TLAS) acceleration structure as a shader resource at argument-table slot
    /// `slot`. Slang lowers a `RaytracingAccelerationStructure` (trailing entry param) to a
    /// resource at the next buffer slot after the root, so ray-query compute kernels use
    /// slot 1. Call after `set_compute_pipeline` and before `dispatch`. The structure is
    /// already resident (registered at create time).
    pub fn bind_acceleration_structure(
        &mut self,
        slot: u32,
        accel: &crate::accel::AccelerationStructure,
    ) {
        let rid = match &accel.inner {
            crate::accel::AccelInner::Metal(a) => a.gpu_resource_id,
            #[allow(unreachable_patterns)]
            _ => return,
        };
        let resource_id: objc2_metal::MTLResourceID = unsafe { std::mem::transmute(rid) };
        unsafe {
            self.argument_table
                .setResource_atBufferIndex(resource_id, slot as usize);
        }
    }

    pub fn set_active_texture_heap_ptr(&mut self, heap_ptr: GpuAddress) {
        self.active_texture_heap_ptr_override = if heap_ptr.0 == 0 {
            None
        } else {
            Some(heap_ptr.0)
        };
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
        let encoder = self
            .render_encoder
            .as_ref()
            .expect("No active render encoder");
        unsafe {
            self.argument_table.setAddress_atIndex(root_table, 0);
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

    pub fn draw_indexed(&mut self, indices: GpuAddress, index_count: u32, instance_count: u32) {
        // Index format is always U32 — the spec has no IndexFormat concept.
        // Metal 4 takes the index buffer as a raw GPU address; the exact byte span it
        // reads is index_count * 4, so no allocation lookup is needed.
        let index_addr_gpu: MTLGPUAddress = indices.0;
        let index_len = (index_count as u64) * 4;
        let root_table = self.current_root_table;
        let topology = self.current_topology;
        let encoder = self
            .render_encoder
            .as_ref()
            .expect("No active render encoder");
        unsafe {
            self.argument_table.setAddress_atIndex(root_table, 0);
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

    pub fn dispatch_indirect(&mut self, args: GpuAddress) {
        let encoder = self
            .compute_encoder
            .as_ref()
            .expect("No active compute encoder");
        let arg_addr_gpu: MTLGPUAddress = args.0;
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

    pub fn draw_indexed_indirect(&mut self, indices: GpuAddress, args: GpuAddress) {
        // Index format is always U32.
        let index_addr_gpu: MTLGPUAddress = indices.0;
        // The index count is GPU-driven, so bound the index buffer length by the bytes
        // remaining in the allocation.
        let index_len = self.allocation_remaining(indices);
        let arg_addr_gpu: MTLGPUAddress = args.0;
        let root_table = self.current_root_table;
        let topology = self.current_topology;
        let encoder = self
            .render_encoder
            .as_ref()
            .expect("No active render encoder");
        unsafe {
            self.argument_table.setAddress_atIndex(root_table, 0);
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

    pub fn draw_indirect_multi(
        &mut self,
        vertex_root: GpuAddress,
        vertex_stride: u32,
        pixel_root: GpuAddress,
        pixel_stride: u32,
        args: GpuAddress,
        draw_count: GpuAddress,
    ) {
        let desc = self
            .render_pass_desc
            .clone()
            .expect("draw_indirect_multi must be recorded inside a render pass");
        self.set_root_table(vertex_root, vertex_stride, pixel_root, pixel_stride);

        let stride = std::mem::size_of::<DrawIndirectMultiArgs>() as u64;
        let arg_remaining = self.allocation_remaining(args);
        let max_draw_count = u32::try_from(arg_remaining / stride).unwrap_or(u32::MAX);
        assert!(
            max_draw_count > 0,
            "draw_indirect_multi args allocation has no complete draw records"
        );

        let topology = self.current_topology;
        let root_table = self.current_root_table;

        // MDI needs its ICB generated by a compute pass that runs *before* the render
        // encoder executes it. Split the pass: end the current render encoder (storing
        // what was drawn so far), run the ICB-generation compute dispatch, then reopen the
        // render encoder with Load actions (preserving prior contents) and execute the
        // generated commands. Subsequent draws continue on the reopened encoder.
        if let Some(encoder) = self.render_encoder.take() {
            encoder.endEncoding();
        }

        let compute = self
            .command_buffer
            .computeCommandEncoder()
            .expect("Failed to create Metal compute encoder for ICB generation");
        self.apply_pending_queue_barrier_compute(&compute);
        compute.setComputePipelineState(&self.mdi_icb_pipeline);
        let generated =
            self.generate_one_mdi_icb(&compute, topology, args, draw_count, max_draw_count);
        compute.endEncoding();
        self.enqueue_queue_barrier(
            MTLStages::Dispatch,
            MTLStages::Vertex | MTLStages::Fragment,
            MTL4VisibilityOptions::Device,
        );

        let encoder = self.begin_metal_render_encoder(&desc, true);
        self.reapply_render_state(&encoder);
        unsafe {
            self.argument_table.setAddress_atIndex(root_table, 0);
            encoder.setArgumentTable_atStages(
                &self.argument_table,
                MTLRenderStages::Vertex | MTLRenderStages::Fragment,
            );
            encoder.executeCommandsInBuffer_indirectBuffer(
                &generated.icb,
                generated.range_buffer.gpuAddress(),
            );
        }
        self.mdi_icb_resources.push(generated);
        self.render_encoder = Some(encoder);
    }

    pub fn memcpy(&mut self, dst: GpuAddress, src: GpuAddress, size: u64) {
        if size == 0 {
            return;
        }
        let (src_buffer, src_offset) = self.resolve_buffer(src, size);
        let (dst_buffer, dst_offset) = self.resolve_buffer(dst, size);

        self.end_active_encoders();
        // Metal 4 routes buffer copies through the compute encoder.
        let encoder = self
            .command_buffer
            .computeCommandEncoder()
            .expect("Failed to create Metal 4 copy encoder");
        // Honor a barrier enqueued before this copy (e.g. between two transfer copies).
        // Copy encoders are discrete and short-lived, so the dependency arrives as a
        // pending queue barrier that must be applied as this encoder begins.
        self.apply_pending_queue_barrier_compute(&encoder);
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

    pub fn copy_to_texture(&mut self, texture_gpu: GpuAddress, src: GpuAddress, texture: &Texture) {
        let (mtl_texture, buffer, offset, size, origin, bytes_per_row, bytes_per_image) =
            self.prepare_texture_copy(texture_gpu, src, texture, "copy_to_texture");

        self.end_active_encoders();
        let encoder = self
            .command_buffer
            .computeCommandEncoder()
            .expect("Failed to create Metal 4 copy encoder");
        // Honor a barrier enqueued before this copy (e.g. between two transfer copies).
        // Copy encoders are discrete and short-lived, so the dependency arrives as a
        // pending queue barrier that must be applied as this encoder begins.
        self.apply_pending_queue_barrier_compute(&encoder);
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

    pub fn copy_from_texture(
        &mut self,
        dst: GpuAddress,
        texture_gpu: GpuAddress,
        texture: &Texture,
    ) {
        let (mtl_texture, buffer, offset, size, origin, bytes_per_row, bytes_per_image) =
            self.prepare_texture_copy(texture_gpu, dst, texture, "copy_from_texture");

        self.end_active_encoders();
        let encoder = self
            .command_buffer
            .computeCommandEncoder()
            .expect("Failed to create Metal 4 copy encoder");
        // Honor a barrier enqueued before this copy (e.g. between two transfer copies).
        // Copy encoders are discrete and short-lived, so the dependency arrives as a
        // pending queue barrier that must be applied as this encoder begins.
        self.apply_pending_queue_barrier_compute(&encoder);
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
        texture_gpu: GpuAddress,
        buffer_gpu: GpuAddress,
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
            texture.gpu_address(),
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
        self.end_active_encoders();
        self.command_buffer.endCommandBuffer();
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

    pub fn signal_after_value(&mut self, desc: &SignalValueDesc) {
        self.signal_after(desc.src_stage, HazardFlags::empty());
        self.pending_value_signals.push(*desc);
    }

    pub fn wait_before_value(&mut self, desc: &WaitValueDesc) {
        self.wait_before(desc.dst_stage, desc.hazard);
        self.pending_value_waits.push(*desc);
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
        self.current_viewport = Some(viewport);
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
        self.current_scissor = Some(scissor);
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

    // -- Mesh shader (meshlet) pipeline + draws --

    /// `gpuSetPipeline` for mesh pipelines — binds PSO, refreshes bindless heaps, binds argument table.
    pub fn set_meshlet_pipeline(&mut self, pso: &MeshletPso) {
        use objc2_metal::MTL4RenderCommandEncoder as _;

        let (
            pipeline,
            cull_mode,
            winding,
            root_constant_size,
            has_texture_heap_slot,
            has_sampler_heap_slot,
        ) = match &pso.inner {
            crate::pipeline::MeshletPsoInner::Metal(mtl_pso) => {
                let has_slot = |slot: usize| mtl_pso.argument_buffer_slots.contains(&slot);
                (
                    mtl_pso.default_pipeline.clone(),
                    mtl_pso.cull_mode,
                    mtl_pso.winding,
                    mtl_pso.root_constant_size,
                    has_slot(1),
                    has_slot(2),
                )
            }
            #[allow(unreachable_patterns)]
            _ => unreachable!("wrong backend"),
        };
        self.root_constant_size = root_constant_size;
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

        self.active_texture_heap_slot_enabled = has_texture_heap_slot;
        self.active_sampler_heap_slot_enabled = has_sampler_heap_slot;
        self.refresh_bindless_heaps(has_texture_heap_slot, has_sampler_heap_slot);
        self.refresh_argument_table();

        let binding = MetalPipelineBinding {
            pipeline: pipeline.clone(),
            cull_mode,
            winding,
            texture_heap_slot: has_texture_heap_slot,
            sampler_heap_slot: has_sampler_heap_slot,
            stages: MTLRenderStages::Mesh | MTLRenderStages::Fragment,
        };
        self.current_pipeline = Some(binding.clone());
        let encoder = self
            .render_encoder
            .as_ref()
            .expect("set_meshlet_pipeline: no active render encoder");
        encoder.setRenderPipelineState(&pipeline);
        encoder.setCullMode(cull_mode);
        encoder.setFrontFacingWinding(winding);
        // Bind the argument table to mesh + fragment stages (object stage handled implicitly).
        encoder.setArgumentTable_atStages(
            &self.argument_table,
            MTLRenderStages::Mesh | MTLRenderStages::Fragment,
        );
    }

    /// Draw using the bound mesh-shader pipeline. Pipeline must be set via `set_meshlet_pipeline`.
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
        let encoder = match self.render_encoder.as_ref() {
            Some(e) => e.clone(),
            None => {
                log::warn!("draw_meshlets: no active render encoder");
                return;
            }
        };
        unsafe {
            self.argument_table.setAddress_atIndex(root_table, 0);
        }
        encoder.setArgumentTable_atStages(
            &self.argument_table,
            MTLRenderStages::Mesh | MTLRenderStages::Fragment,
        );
        encoder.drawMeshThreadgroups_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
            groups, tg_obj, tg_mesh,
        );
    }

    /// Indirect mesh draw. Pipeline must be set via `set_meshlet_pipeline`.
    /// `args` points to one indirect mesh dispatch command.
    pub fn draw_meshlets_indirect(&mut self, args: GpuAddress) {
        use objc2_metal::MTL4RenderCommandEncoder as _;
        let tg_obj = self.current_mesh_tpg_object;
        let tg_mesh = self.current_mesh_tpg_mesh;
        let root_table = self.current_root_table;
        let encoder = match self.render_encoder.as_ref() {
            Some(e) => e.clone(),
            None => {
                log::warn!("draw_meshlets_indirect: no active render encoder");
                return;
            }
        };
        unsafe {
            self.argument_table.setAddress_atIndex(root_table, 0);
        }
        encoder.setArgumentTable_atStages(
            &self.argument_table,
            MTLRenderStages::Mesh | MTLRenderStages::Fragment,
        );
        // MTL4 indirect draw takes MTLGPUAddress directly.
        encoder.drawMeshThreadgroupsWithIndirectBuffer_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
            args.0,
            tg_obj,
            tg_mesh,
        );
    }

    // -- Acceleration structure builds (MTL4 compute encoder) --

    pub fn build_blas(&mut self, accel: &crate::accel::AccelerationStructure, desc: &BlasDesc) {
        use super::accel::make_blas_geometry_descriptors;
        use objc2_metal::{
            MTL4ComputeCommandEncoder as _, MTL4PrimitiveAccelerationStructureDescriptor,
            MTLBuffer as _,
        };

        let (vk_as, scratch) = match &accel.inner {
            #[cfg(feature = "metal")]
            crate::accel::AccelInner::Metal(a) => {
                (a.acceleration_structure.clone(), a.scratch_buffer.clone())
            }
            #[allow(unreachable_patterns)]
            _ => return,
        };
        let scratch = match scratch {
            Some(s) => s,
            None => {
                log::warn!(
                    "build_blas: no scratch buffer — was the BLAS created by device.create_blas?"
                );
                return;
            }
        };

        let geometries = match make_blas_geometry_descriptors(desc) {
            Ok(geometries) => geometries,
            Err(err) => {
                log::warn!("build_blas: invalid BLAS descriptor: {err}");
                return;
            }
        };

        let primitive_desc = MTL4PrimitiveAccelerationStructureDescriptor::new();
        primitive_desc.setGeometryDescriptors(Some(&geometries.array));

        // Get or create compute encoder for the build command.
        self.end_active_encoders();
        let encoder = self
            .command_buffer
            .computeCommandEncoder()
            .expect("Failed to create compute encoder for BLAS build");

        let scratch_addr = scratch.gpuAddress();
        let scratch_range = objc2_metal::MTL4BufferRange {
            bufferAddress: scratch_addr,
            length: !0u64, // full remaining length
        };

        unsafe {
            encoder.buildAccelerationStructure_descriptor_scratchBuffer(
                &vk_as,
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

        let (vk_as, scratch) = match &accel.inner {
            #[cfg(feature = "metal")]
            crate::accel::AccelInner::Metal(a) => {
                (a.acceleration_structure.clone(), a.scratch_buffer.clone())
            }
            #[allow(unreachable_patterns)]
            _ => return,
        };
        let scratch = match scratch {
            Some(s) => s,
            None => {
                log::warn!("build_tlas: no scratch buffer");
                return;
            }
        };

        let instance_desc = MTL4InstanceAccelerationStructureDescriptor::new();
        unsafe {
            instance_desc.setInstanceDescriptorBuffer(objc2_metal::MTL4BufferRange {
                bufferAddress: desc.instance_buffer.0,
                // Indirect instance-descriptor layout (see device.write_tlas_instance), not
                // the Vulkan-shaped TlasInstance.
                length: (desc.instance_count as u64)
                    * std::mem::size_of::<
                        objc2_metal::MTLIndirectAccelerationStructureInstanceDescriptor,
                    >() as u64,
            });
            instance_desc.setInstanceCount(desc.instance_count as usize);
        }

        self.end_active_encoders();
        let encoder = self
            .command_buffer
            .computeCommandEncoder()
            .expect("Failed to create compute encoder for TLAS build");

        let scratch_addr = scratch.gpuAddress();
        let scratch_range = objc2_metal::MTL4BufferRange {
            bufferAddress: scratch_addr,
            length: !0u64,
        };

        unsafe {
            encoder.buildAccelerationStructure_descriptor_scratchBuffer(
                &vk_as,
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
        self.remove_allocation_from_residency(self.root_table_buffer.as_ref());
        if let Some(tex_heap) = self.texture_heap_buffer.as_ref() {
            self.remove_allocation_from_residency(tex_heap.as_ref());
        }
        if let Some(sampler_heap) = self.sampler_heap_buffer.as_ref() {
            self.remove_allocation_from_residency(sampler_heap.as_ref());
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

    // DRAW_ARGUMENTS: GPU-written indirect args need Device visibility so the command
    //   processor sees the updated draw/dispatch parameters.
    // DEPTH_STENCIL: compute-written depth needs Device visibility to invalidate HiZ.
    // DESCRIPTORS: descriptor heap write needs Device + ResourceAlias for alias ordering.
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
