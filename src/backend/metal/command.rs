use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer,
    MTL4CommandEncoder, MTL4ComputeCommandEncoder, MTL4RenderCommandEncoder,
    MTL4RenderPassDescriptor, MTL4VisibilityOptions, MTLAllocation, MTLArgumentEncoder, MTLBuffer,
    MTLComputePipelineState, MTLDepthStencilState, MTLDevice, MTLFunction, MTLGPUAddress,
    MTLIndexType, MTLIndirectCommandBuffer, MTLIndirectCommandBufferDescriptor,
    MTLIndirectCommandType, MTLLoadAction, MTLOrigin, MTLPrimitiveType, MTLRenderPipelineState,
    MTLRenderStages, MTLResidencySet, MTLResourceOptions, MTLScissorRect, MTLSize, MTLStages,
    MTLStencilOperation, MTLStoreAction, MTLTexture, MTLViewport,
};

use crate::barrier::{HazardFlags, StageFlags};
use crate::command::{
    DrawIndexedIndirectArgs, DrawIndexedIndirectMultiArgs, LoadOp, RenderPassDesc, RenderTarget,
    SignalValueDesc, StoreOp, WaitValueDesc,
};
use crate::error::{RhiError, RhiResult};
use crate::pipeline::{
    BlendState, ComputePso, DepthStencilState, GraphicsPso, MeshletPso, RayTracingPso,
};
use crate::texture::{Texture, bytes_per_pixel};
use crate::types::*;

use super::device::{SharedAllocations, SharedSamplers, SharedTextures};

const ROOT_TABLE_BYTES: usize = 32;
const RT_TRACE_TABLE_BYTES: usize = 88;
const ROOT_TABLE_RING_ENTRIES: usize = 65_536;
const ROOT_TABLE_RING_BYTES: usize = ROOT_TABLE_BYTES * ROOT_TABLE_RING_ENTRIES;
const METAL_BINDLESS_TEXTURE_CAPACITY: usize = 65_536;
const METAL_BINDLESS_SAMPLER_CAPACITY: usize = 256;
const MDI_ICB_THREADGROUP_SIZE: usize = 64;

struct DeferredRenderPass {
    desc: RenderPassDesc,
    commands: Vec<MetalRenderCommand>,
}

#[derive(Clone)]
struct MetalPipelineBinding {
    pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    cull_mode: objc2_metal::MTLCullMode,
    winding: objc2_metal::MTLWinding,
    topology: MTLPrimitiveType,
    root_constant_size: u32,
    texture_heap_slot: bool,
    sampler_heap_slot: bool,
    storage_heap_slot: bool,
    stages: MTLRenderStages,
}

enum MetalRenderCommand {
    SetPipeline(MetalPipelineBinding),
    SetDepthStencil {
        state: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
        depth_bias: Option<(f32, f32, f32)>,
    },
    SetViewport(MTLViewport),
    SetScissor(MTLScissorRect),
    Draw {
        root_table: MTLGPUAddress,
        topology: MTLPrimitiveType,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    },
    DrawIndexed {
        root_table: MTLGPUAddress,
        topology: MTLPrimitiveType,
        index_buffer: MTLGPUAddress,
        index_buffer_len: u64,
        index_count: u32,
        instance_count: u32,
    },
    DrawIndexedIndirect {
        root_table: MTLGPUAddress,
        topology: MTLPrimitiveType,
        index_buffer: MTLGPUAddress,
        index_buffer_len: u64,
        args: MTLGPUAddress,
    },
    DrawIndexedIndirectMulti {
        root_table: MTLGPUAddress,
        topology: MTLPrimitiveType,
        args: GpuAddress,
        draw_count: GpuAddress,
        max_draw_count: u32,
        generated: Option<GeneratedMdiIcb>,
    },
    DrawMeshlets {
        root_table: MTLGPUAddress,
        groups: MTLSize,
        object_tpg: MTLSize,
        mesh_tpg: MTLSize,
    },
    DrawMeshletsIndirect {
        root_table: MTLGPUAddress,
        args: MTLGPUAddress,
        object_tpg: MTLSize,
        mesh_tpg: MTLSize,
    },
}

#[allow(dead_code)]
struct GeneratedMdiIcb {
    icb: Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>,
    range_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    argument_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    max_draw_count_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    primitive_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
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
    /// Encoded bindless texture heap argument buffer.
    texture_heap_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Encoded bindless sampler heap argument buffer.
    sampler_heap_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Encoded bindless storage texture heap argument buffer.
    storage_heap_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Shared texture list for argument buffer population.
    pub(crate) textures: SharedTextures,
    /// Shared sampler list for argument buffer population.
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
    active_storage_heap_slot_enabled: bool,
    active_texture_heap_ptr_override: Option<MTLGPUAddress>,
    active_render_pass: Option<DeferredRenderPass>,
    current_root_table: MTLGPUAddress,
    mdi_icb_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    mdi_icb_function: Retained<ProtocolObject<dyn MTLFunction>>,
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
        for alloc in allocations.iter() {
            let base = alloc.base.0;
            let end = base + alloc.size;
            if addr_u64 >= base && addr_u64 + size <= end {
                return (alloc.buffer.clone(), addr_u64 - base);
            }
        }
        panic!("GPU address {addr_u64:#x} not found in allocation registry");
    }

    fn resolve_gpu_address(&self, addr: GpuAddress, size: u64) -> (MTLGPUAddress, u64) {
        let addr_u64 = addr.0;
        let allocations = self.allocations.borrow();
        for alloc in allocations.iter() {
            let base = alloc.base.0;
            let end = base + alloc.size;
            if addr_u64 >= base && addr_u64 + size <= end {
                let offset = addr_u64 - base;
                let remaining = alloc.size - offset;
                return (addr_u64 as MTLGPUAddress, remaining);
            }
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
            if self.active_texture_heap_slot_enabled {
                if let Some(override_ptr) = self.active_texture_heap_ptr_override {
                    self.argument_table.setAddress_atIndex(override_ptr, 1);
                } else if let Some(tex_heap) = self.texture_heap_buffer.as_ref() {
                    self.argument_table
                        .setAddress_atIndex(tex_heap.gpuAddress(), 1);
                } else {
                    self.argument_table.setAddress_atIndex(0, 1);
                }
            } else {
                self.argument_table.setAddress_atIndex(0, 1);
            }
            if self.active_sampler_heap_slot_enabled {
                if let Some(sampler_heap) = self.sampler_heap_buffer.as_ref() {
                    self.argument_table
                        .setAddress_atIndex(sampler_heap.gpuAddress(), 2);
                } else {
                    self.argument_table.setAddress_atIndex(0, 2);
                }
            } else {
                self.argument_table.setAddress_atIndex(0, 2);
            }
            if self.active_storage_heap_slot_enabled {
                if let Some(storage_heap) = self.storage_heap_buffer.as_ref() {
                    self.argument_table
                        .setAddress_atIndex(storage_heap.gpuAddress(), 3);
                } else {
                    self.argument_table.setAddress_atIndex(0, 3);
                }
            } else {
                self.argument_table.setAddress_atIndex(0, 3);
            }
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

    fn deferred_commands_mut(&mut self) -> &mut Vec<MetalRenderCommand> {
        &mut self
            .active_render_pass
            .as_mut()
            .expect("No active deferred render pass")
            .commands
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

    fn generate_deferred_mdi_icbs(&mut self, pass: &mut DeferredRenderPass) {
        let mut indices = Vec::new();
        for (i, cmd) in pass.commands.iter().enumerate() {
            if matches!(cmd, MetalRenderCommand::DrawIndexedIndirectMulti { .. }) {
                indices.push(i);
            }
        }
        if indices.is_empty() {
            return;
        }

        let encoder = self
            .command_buffer
            .computeCommandEncoder()
            .expect("Failed to create Metal compute encoder for ICB generation");
        self.apply_pending_queue_barrier_compute(&encoder);
        encoder.setComputePipelineState(&self.mdi_icb_pipeline);

        for index in indices {
            let generated = match &pass.commands[index] {
                MetalRenderCommand::DrawIndexedIndirectMulti {
                    topology,
                    args,
                    draw_count,
                    max_draw_count,
                    ..
                } => self.generate_one_mdi_icb(&encoder, *topology, *args, *draw_count, *max_draw_count),
                _ => unreachable!(),
            };
            if let MetalRenderCommand::DrawIndexedIndirectMulti {
                generated: slot, ..
            } = &mut pass.commands[index]
            {
                *slot = Some(generated);
            }
        }

        encoder.endEncoding();
        self.compute_encoder = None;
        self.enqueue_queue_barrier(
            MTLStages::Dispatch,
            MTLStages::Vertex | MTLStages::Fragment,
            MTL4VisibilityOptions::Device,
        );
    }

    fn generate_one_mdi_icb(
        &mut self,
        encoder: &ProtocolObject<dyn MTL4ComputeCommandEncoder>,
        topology: MTLPrimitiveType,
        args: GpuAddress,
        draw_count: GpuAddress,
        max_draw_count: u32,
    ) -> GeneratedMdiIcb {
        let (args_addr, _) = self.resolve_gpu_address(
            args,
            std::mem::size_of::<DrawIndexedIndirectMultiArgs>() as u64,
        );
        let (draw_count_addr, _) =
            self.resolve_gpu_address(draw_count, std::mem::size_of::<u32>() as u64);

        let icb_desc = MTLIndirectCommandBufferDescriptor::new();
        icb_desc.setCommandTypes(MTLIndirectCommandType::DrawIndexed);
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
        .expect("Failed to create Metal ICB for indexed MDI");
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

        unsafe {
            std::ptr::write_unaligned(
                max_draw_count_buffer.contents().as_ptr() as *mut u32,
                max_draw_count,
            );
            std::ptr::write_unaligned(
                primitive_buffer.contents().as_ptr() as *mut u32,
                Self::primitive_id(topology),
            );
        }

        let icb_arg_encoder = unsafe { self.mdi_icb_function.newArgumentEncoderWithBufferIndex(3) };
        let argument_buffer = self.make_command_buffer_resource(
            icb_arg_encoder.encodedLength(),
            MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            icb_arg_encoder.setArgumentBuffer_offset(Some(&argument_buffer), 0);
            icb_arg_encoder.setIndirectCommandBuffer_atIndex(Some(&icb), 0);
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
            self.argument_table
                .setAddress_atIndex(argument_buffer.gpuAddress(), 3);
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
            argument_buffer,
            max_draw_count_buffer,
            primitive_buffer,
        }
    }

    fn ensure_texture_heap_buffer(&mut self, required_len: usize) {
        let needs_realloc = self
            .texture_heap_buffer
            .as_ref()
            .map(|b| b.length() < required_len)
            .unwrap_or(true);
        if !needs_realloc {
            return;
        }
        if let Some(old) = self.texture_heap_buffer.take() {
            self.remove_allocation_from_residency(old.as_ref());
        }
        let new_buf = self
            .device
            .newBufferWithLength_options(required_len, MTLResourceOptions::StorageModeShared)
            .expect("Failed to allocate Metal texture heap argument buffer");
        self.add_allocation_to_residency(new_buf.as_ref());
        self.texture_heap_buffer = Some(new_buf);
    }

    fn ensure_sampler_heap_buffer(&mut self, required_len: usize) {
        let needs_realloc = self
            .sampler_heap_buffer
            .as_ref()
            .map(|b| b.length() < required_len)
            .unwrap_or(true);
        if !needs_realloc {
            return;
        }
        if let Some(old) = self.sampler_heap_buffer.take() {
            self.remove_allocation_from_residency(old.as_ref());
        }
        let new_buf = self
            .device
            .newBufferWithLength_options(required_len, MTLResourceOptions::StorageModeShared)
            .expect("Failed to allocate Metal sampler heap argument buffer");
        self.add_allocation_to_residency(new_buf.as_ref());
        self.sampler_heap_buffer = Some(new_buf);
    }

    fn ensure_storage_heap_buffer(&mut self, required_len: usize) {
        let needs_realloc = self
            .storage_heap_buffer
            .as_ref()
            .map(|b| b.length() < required_len)
            .unwrap_or(true);
        if !needs_realloc {
            return;
        }
        if let Some(old) = self.storage_heap_buffer.take() {
            self.remove_allocation_from_residency(old.as_ref());
        }
        let new_buf = self
            .device
            .newBufferWithLength_options(required_len, MTLResourceOptions::StorageModeShared)
            .expect("Failed to allocate Metal storage heap argument buffer");
        self.add_allocation_to_residency(new_buf.as_ref());
        self.storage_heap_buffer = Some(new_buf);
    }

    fn refresh_bindless_heaps(
        &mut self,
        frag_fn: &ProtocolObject<dyn MTLFunction>,
        refresh_texture_heap: bool,
        refresh_sampler_heap: bool,
    ) {
        unsafe {
            if refresh_texture_heap {
                let texture_encoder = frag_fn.newArgumentEncoderWithBufferIndex(1);
                let texture_bytes = texture_encoder.encodedLength();
                self.ensure_texture_heap_buffer(texture_bytes);
                let texture_heap = self
                    .texture_heap_buffer
                    .as_ref()
                    .expect("texture heap buffer must exist after allocation");
                texture_encoder.setArgumentBuffer_offset(Some(texture_heap), 0);
                let textures = self.textures.borrow();
                if textures.len() > METAL_BINDLESS_TEXTURE_CAPACITY {
                    panic!(
                        "Metal texture heap overflow: {} textures exceed shader capacity {}",
                        textures.len(),
                        METAL_BINDLESS_TEXTURE_CAPACITY
                    );
                }
                for (i, tex_opt) in textures.iter().enumerate() {
                    texture_encoder.setTexture_atIndex(tex_opt.as_deref(), i);
                }
                drop(textures);
            }

            if refresh_sampler_heap {
                let sampler_encoder = frag_fn.newArgumentEncoderWithBufferIndex(2);
                let sampler_bytes = sampler_encoder.encodedLength();
                self.ensure_sampler_heap_buffer(sampler_bytes);
                let sampler_heap = self
                    .sampler_heap_buffer
                    .as_ref()
                    .expect("sampler heap buffer must exist after allocation");
                sampler_encoder.setArgumentBuffer_offset(Some(sampler_heap), 0);
                let samplers = self.samplers.borrow();
                if samplers.len() > METAL_BINDLESS_SAMPLER_CAPACITY {
                    panic!(
                        "Metal sampler heap overflow: {} samplers exceed shader capacity {}",
                        samplers.len(),
                        METAL_BINDLESS_SAMPLER_CAPACITY
                    );
                }
                for (i, sampler) in samplers.iter().enumerate() {
                    sampler_encoder.setSamplerState_atIndex(Some(sampler), i);
                }
                // Bound through argument table refresh.
            }
        }
    }

    fn refresh_storage_heap(&mut self, function: &ProtocolObject<dyn MTLFunction>) {
        unsafe {
            let storage_encoder = function.newArgumentEncoderWithBufferIndex(3);
            let storage_bytes = storage_encoder.encodedLength();
            self.ensure_storage_heap_buffer(storage_bytes);
            let storage_heap = self
                .storage_heap_buffer
                .as_ref()
                .expect("storage heap buffer must exist after allocation");
            storage_encoder.setArgumentBuffer_offset(Some(storage_heap), 0);
            let textures = self.textures.borrow();
            if textures.len() > METAL_BINDLESS_TEXTURE_CAPACITY {
                panic!(
                    "Metal storage heap overflow: {} textures exceed shader capacity {}",
                    textures.len(),
                    METAL_BINDLESS_TEXTURE_CAPACITY
                );
            }
            for (i, tex_opt) in textures.iter().enumerate() {
                let tex = match tex_opt.as_deref() {
                    Some(tex) => tex,
                    None => {
                        storage_encoder.setTexture_atIndex(None, i);
                        continue;
                    }
                };
                if tex
                    .usage()
                    .contains(objc2_metal::MTLTextureUsage::ShaderWrite)
                {
                    storage_encoder.setTexture_atIndex(Some(tex), i);
                } else {
                    storage_encoder.setTexture_atIndex(None, i);
                }
            }
            // Bound through argument table refresh.
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
        mdi_icb_function: Retained<ProtocolObject<dyn MTLFunction>>,
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
            storage_heap_buffer: None,
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
            active_storage_heap_slot_enabled: false,
            active_texture_heap_ptr_override: None,
            active_render_pass: None,
            current_root_table: 0,
            mdi_icb_pipeline,
            mdi_icb_function,
            mdi_icb_resources: Vec::new(),
        };

        cmd.refresh_argument_table();
        Ok(cmd)
    }

    pub fn begin_render_pass(&mut self, desc: &RenderPassDesc) {
        self.end_active_encoders();
        if self.active_render_pass.is_some() {
            panic!("begin_render_pass called while a deferred render pass is active");
        }
        self.active_render_pass = Some(DeferredRenderPass {
            desc: desc.clone(),
            commands: Vec::new(),
        });
    }

    fn begin_metal_render_encoder(
        &mut self,
        desc: &RenderPassDesc,
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

        // Create the render command encoder
        let encoder = self
            .command_buffer
            .renderCommandEncoderWithDescriptor(&pass_desc)
            .expect("Failed to create Metal render command encoder");

        self.apply_pending_queue_barrier_render(&encoder);

        encoder
    }

    pub fn end_render_pass(&mut self) {
        if let Some(mut pass) = self.active_render_pass.take() {
            self.generate_deferred_mdi_icbs(&mut pass);
            let encoder = self.begin_metal_render_encoder(&pass.desc);
            self.replay_deferred_render_pass(&encoder, pass);
            encoder.endEncoding();
            return;
        }
        if let Some(encoder) = self.render_encoder.take() {
            encoder.endEncoding();
        }
    }

    fn replay_deferred_render_pass(
        &mut self,
        encoder: &ProtocolObject<dyn MTL4RenderCommandEncoder>,
        pass: DeferredRenderPass,
    ) {
        for command in pass.commands {
            match command {
                MetalRenderCommand::SetPipeline(binding) => {
                    encoder.setRenderPipelineState(&binding.pipeline);
                    encoder.setCullMode(binding.cull_mode);
                    encoder.setFrontFacingWinding(binding.winding);
                    self.current_topology = binding.topology;
                    self.root_constant_size = binding.root_constant_size;
                    self.active_texture_heap_slot_enabled = binding.texture_heap_slot;
                    self.active_sampler_heap_slot_enabled = binding.sampler_heap_slot;
                    self.active_storage_heap_slot_enabled = binding.storage_heap_slot;
                    self.refresh_argument_table();
                    encoder.setArgumentTable_atStages(&self.argument_table, binding.stages);
                }
                MetalRenderCommand::SetDepthStencil { state, depth_bias } => {
                    encoder.setDepthStencilState(Some(&state));
                    if let Some((bias, slope, clamp)) = depth_bias {
                        encoder.setDepthBias_slopeScale_clamp(bias, slope, clamp);
                    }
                }
                MetalRenderCommand::SetViewport(viewport) => {
                    encoder.setViewport(viewport);
                }
                MetalRenderCommand::SetScissor(scissor) => {
                    encoder.setScissorRect(scissor);
                }
                MetalRenderCommand::Draw {
                    root_table,
                    topology,
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                } => unsafe {
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
                },
                MetalRenderCommand::DrawIndexed {
                    root_table,
                    topology,
                    index_buffer,
                    index_buffer_len,
                    index_count,
                    instance_count,
                } => unsafe {
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
                            index_buffer,
                            index_buffer_len as usize,
                            instance_count as usize,
                            0,
                            0,
                        );
                },
                MetalRenderCommand::DrawIndexedIndirect {
                    root_table,
                    topology,
                    index_buffer,
                    index_buffer_len,
                    args,
                } => unsafe {
                    self.argument_table.setAddress_atIndex(root_table, 0);
                    encoder.setArgumentTable_atStages(
                        &self.argument_table,
                        MTLRenderStages::Vertex | MTLRenderStages::Fragment,
                    );
                    encoder.drawIndexedPrimitives_indexType_indexBuffer_indexBufferLength_indirectBuffer(
                        topology,
                        MTLIndexType::UInt32,
                        index_buffer,
                        index_buffer_len as usize,
                        args,
                    );
                },
                MetalRenderCommand::DrawIndexedIndirectMulti {
                    root_table,
                    generated,
                    ..
                } => {
                    let generated = generated.expect("deferred MDI ICB was not generated");
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
                }
                MetalRenderCommand::DrawMeshlets {
                    root_table,
                    groups,
                    object_tpg,
                    mesh_tpg,
                } => unsafe {
                    self.argument_table.setAddress_atIndex(root_table, 0);
                    encoder.setArgumentTable_atStages(
                        &self.argument_table,
                        MTLRenderStages::Mesh | MTLRenderStages::Fragment,
                    );
                    encoder.drawMeshThreadgroups_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                        groups, object_tpg, mesh_tpg,
                    );
                }
                MetalRenderCommand::DrawMeshletsIndirect {
                    root_table,
                    args,
                    object_tpg,
                    mesh_tpg,
                } => unsafe {
                    self.argument_table.setAddress_atIndex(root_table, 0);
                    encoder.setArgumentTable_atStages(
                        &self.argument_table,
                        MTLRenderStages::Mesh | MTLRenderStages::Fragment,
                    );
                    encoder.drawMeshThreadgroupsWithIndirectBuffer_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                        args, object_tpg, mesh_tpg,
                    );
                }
            }
        }
    }

    pub fn set_graphics_pipeline(&mut self, pso: &GraphicsPso) {
        let (
            pipeline,
            cull_mode,
            winding,
            topology,
            root_constant_size,
            frag_fn,
            has_texture_heap_slot,
            has_sampler_heap_slot,
            has_storage_heap_slot,
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
                    mtl_pso.frag_fn.clone(),
                    has_slot(1),
                    has_slot(2),
                    has_slot(3),
                )
            }
            #[allow(unreachable_patterns)]
            _ => unreachable!("wrong backend"),
        };
        self.current_topology = topology;
        self.root_constant_size = root_constant_size;
        self.active_texture_heap_slot_enabled = has_texture_heap_slot;
        self.active_sampler_heap_slot_enabled = has_sampler_heap_slot;
        self.active_storage_heap_slot_enabled = has_storage_heap_slot;
        self.refresh_bindless_heaps(
            frag_fn.as_ref(),
            has_texture_heap_slot,
            has_sampler_heap_slot,
        );
        if has_storage_heap_slot {
            self.refresh_storage_heap(frag_fn.as_ref());
        }
        self.refresh_argument_table();
        let binding = MetalPipelineBinding {
            pipeline: pipeline.clone(),
            cull_mode,
            winding,
            topology,
            root_constant_size,
            texture_heap_slot: has_texture_heap_slot,
            sampler_heap_slot: has_sampler_heap_slot,
            storage_heap_slot: has_storage_heap_slot,
            stages: MTLRenderStages::Vertex | MTLRenderStages::Fragment,
        };
        if self.active_render_pass.is_some() {
            self.deferred_commands_mut()
                .push(MetalRenderCommand::SetPipeline(binding));
            return;
        }
        let encoder = self.render_encoder.as_ref().expect("No active render encoder");
        encoder.setRenderPipelineState(&pipeline);
        encoder.setCullMode(cull_mode);
        encoder.setFrontFacingWinding(winding);
        encoder.setArgumentTable_atStages(&self.argument_table, binding.stages);
    }

    pub fn set_compute_pipeline(&mut self, pso: &ComputePso) {
        self.end_active_encoders();

        let encoder = self
            .command_buffer
            .computeCommandEncoder()
            .expect("Failed to create Metal compute command encoder");

        self.apply_pending_queue_barrier_compute(&encoder);
        self.compute_encoder = Some(encoder);

        {
            let encoder = self.compute_encoder.as_ref().expect("No compute encoder");
            match &pso.inner {
                crate::pipeline::ComputePsoInner::Metal(mtl_pso) => {
                    encoder.setComputePipelineState(&mtl_pso.pipeline);
                    self.current_threads_per_threadgroup = mtl_pso.threads_per_threadgroup;
                    self.root_constant_size = mtl_pso.root_constant_size;
                    let has_slot =
                        |slot: usize| mtl_pso.compute_argument_buffer_slots.contains(&slot);
                    let has_texture_heap_slot = has_slot(1);
                    let has_sampler_heap_slot = has_slot(2);
                    let has_storage_heap_slot = has_slot(3);
                    self.active_texture_heap_slot_enabled = has_texture_heap_slot;
                    self.active_sampler_heap_slot_enabled = has_sampler_heap_slot;
                    self.active_storage_heap_slot_enabled = has_storage_heap_slot;
                    self.refresh_bindless_heaps(
                        mtl_pso.compute_fn.as_ref(),
                        has_texture_heap_slot,
                        has_sampler_heap_slot,
                    );
                    if has_storage_heap_slot {
                        self.refresh_storage_heap(mtl_pso.compute_fn.as_ref());
                    }
                }
                #[allow(unreachable_patterns)]
                _ => unreachable!("wrong backend"),
            }
        }
        self.refresh_argument_table();
        let encoder = self.compute_encoder.as_ref().expect("No compute encoder");
        encoder.setArgumentTable(Some(&self.argument_table));
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
            if self.active_render_pass.is_some() {
                self.deferred_commands_mut()
                    .push(MetalRenderCommand::SetDepthStencil {
                        state: ds_state.clone(),
                        depth_bias: if state.depth_bias != 0.0
                            || state.depth_bias_slope_factor != 0.0
                        {
                            Some((
                                state.depth_bias,
                                state.depth_bias_slope_factor,
                                state.depth_bias_clamp,
                            ))
                        } else {
                            None
                        },
                    });
                self.depth_stencil_states.push(ds_state);
                return;
            }
            let encoder = self
                .render_encoder
                .as_ref()
                .expect("No active render encoder");
            encoder.setDepthStencilState(Some(&ds_state));
            self.depth_stencil_states.push(ds_state);
        }

        // Depth bias (Metal: set directly on the encoder)
        if state.depth_bias != 0.0 || state.depth_bias_slope_factor != 0.0 {
            let encoder = self
                .render_encoder
                .as_ref()
                .expect("No active render encoder");
            encoder.setDepthBias_slopeScale_clamp(
                state.depth_bias,
                state.depth_bias_slope_factor,
                state.depth_bias_clamp,
            );
        }
    }

    pub fn set_blend_state(&mut self, state: &BlendState) {
        // Blend state is baked into the pipeline in Metal.
        // Dynamic blend is limited to blend constants.
        self.current_blend_state = state.clone();
    }

    pub fn set_root_data(&mut self, vertex_root: GpuAddress, pixel_root: GpuAddress) {
        self.set_root_table(vertex_root, 0, pixel_root, 0);
    }

    fn set_root_table(
        &mut self,
        vertex_root_base: GpuAddress,
        vertex_stride: u32,
        pixel_root_base: GpuAddress,
        pixel_stride: u32,
    ) {
        debug_assert!(
            self.render_encoder.is_some() || self.active_render_pass.is_some(),
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
        unsafe {
            self.argument_table.setAddress_atIndex(root.0, 0);
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
        if self.active_render_pass.is_some() {
            let root_table = self.current_root_table;
            let topology = self.current_topology;
            self.deferred_commands_mut().push(MetalRenderCommand::Draw {
                root_table,
                topology,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            });
            return;
        }
        let encoder = self
            .render_encoder
            .as_ref()
            .expect("No active render encoder");
        unsafe {
            encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                self.current_topology,
                first_vertex as usize,
                vertex_count as usize,
                instance_count as usize,
                first_instance as usize,
            );
        }
    }

    pub fn draw_indexed(&mut self, indices: GpuAddress, index_count: u32, instance_count: u32) {
        // Index format is always U32 — the spec has no IndexFormat concept.
        let bytes_needed = (index_count as u64) * 4;
        let (index_addr_gpu, index_len) = self.resolve_gpu_address(indices, bytes_needed);
        if self.active_render_pass.is_some() {
            let root_table = self.current_root_table;
            let topology = self.current_topology;
            self.deferred_commands_mut()
                .push(MetalRenderCommand::DrawIndexed {
                    root_table,
                    topology,
                    index_buffer: index_addr_gpu,
                    index_buffer_len: index_len,
                    index_count,
                    instance_count,
                });
            return;
        }

        let encoder = self
            .render_encoder
            .as_ref()
            .expect("No active render encoder");
        unsafe {
            encoder
                .drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferLength_instanceCount_baseVertex_baseInstance(
                    self.current_topology,
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
        let (arg_addr_gpu, _arg_len) = self.resolve_gpu_address(
            args,
            std::mem::size_of::<crate::command::DispatchIndirectArgs>() as u64,
        );
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
        let (index_addr_gpu, index_len) = self.resolve_gpu_address(indices, 4);
        let (arg_addr_gpu, _arg_len) =
            self.resolve_gpu_address(args, std::mem::size_of::<DrawIndexedIndirectArgs>() as u64);
        if self.active_render_pass.is_some() {
            let root_table = self.current_root_table;
            let topology = self.current_topology;
            self.deferred_commands_mut()
                .push(MetalRenderCommand::DrawIndexedIndirect {
                    root_table,
                    topology,
                    index_buffer: index_addr_gpu,
                    index_buffer_len: index_len,
                    args: arg_addr_gpu,
                });
            return;
        }

        let encoder = self
            .render_encoder
            .as_ref()
            .expect("No active render encoder");
        unsafe {
            encoder.drawIndexedPrimitives_indexType_indexBuffer_indexBufferLength_indirectBuffer(
                self.current_topology,
                MTLIndexType::UInt32,
                index_addr_gpu,
                index_len as usize,
                arg_addr_gpu,
            );
        }
    }

    pub fn draw_indexed_indirect_multi(
        &mut self,
        vertex_root: GpuAddress,
        vertex_stride: u32,
        pixel_root: GpuAddress,
        pixel_stride: u32,
        args: GpuAddress,
        draw_count: GpuAddress,
    ) -> RhiResult<()> {
        self.draw_indexed_indirect_multi_root_table(
            vertex_root,
            vertex_stride,
            pixel_root,
            pixel_stride,
            args,
            draw_count,
        )
    }

    pub fn draw_indexed_indirect_multi_root_table(
        &mut self,
        vertex_root_base: GpuAddress,
        vertex_stride: u32,
        pixel_root_base: GpuAddress,
        pixel_stride: u32,
        args: GpuAddress,
        draw_count: GpuAddress,
    ) -> RhiResult<()> {
        self.set_root_table(
            vertex_root_base,
            vertex_stride,
            pixel_root_base,
            pixel_stride,
        );
        let _ = self.resolve_gpu_address(
            args,
            std::mem::size_of::<DrawIndexedIndirectMultiArgs>() as u64,
        );
        let (_, arg_remaining) = self.resolve_gpu_address(
            args,
            std::mem::size_of::<DrawIndexedIndirectMultiArgs>() as u64,
        );
        let _ = self.resolve_gpu_address(draw_count, std::mem::size_of::<u32>() as u64);
        let max_draw_count = (arg_remaining
            / std::mem::size_of::<DrawIndexedIndirectMultiArgs>() as u64)
            .min(u32::MAX as u64) as u32;
        if max_draw_count == 0 {
            return Err(RhiError::CommandBuffer(
                "draw_indexed_indirect_multi args allocation has no complete draw records".into(),
            ));
        }
        if self.active_render_pass.is_none() {
            return Err(RhiError::CommandBuffer(
                "draw_indexed_indirect_multi must be recorded inside a render pass".into(),
            ));
        }
        let root_table = self.current_root_table;
        let topology = self.current_topology;
        self.deferred_commands_mut()
            .push(MetalRenderCommand::DrawIndexedIndirectMulti {
                root_table,
                topology,
                args,
                draw_count,
                max_draw_count,
                generated: None,
            });
        Ok(())
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
        assert_eq!(
            texture_gpu,
            texture.gpu_address(),
            "copy_to_texture texture_gpu must match the address used to create the texture"
        );
        let mtl_texture = self.resolve_texture(texture.id());
        let width = texture.desc().width;
        let height = texture.desc().height;
        let bpp = bytes_per_pixel(texture.desc().format)
            .expect("Unsupported texture format for copy_to_texture");
        let bytes_per_row = width as usize * bpp;
        let bytes_per_image = bytes_per_row * height as usize;
        let (src_buffer, src_offset) = self.resolve_buffer(src, bytes_per_image as u64);

        self.end_active_encoders();
        // Metal 4 routes texture copies through the compute encoder.
        let encoder = self
            .command_buffer
            .computeCommandEncoder()
            .expect("Failed to create Metal 4 copy encoder");
        let size = MTLSize {
            width: width as usize,
            height: height as usize,
            depth: 1,
        };
        let origin = MTLOrigin { x: 0, y: 0, z: 0 };
        unsafe {
            encoder.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                &src_buffer,
                src_offset as usize,
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
        assert_eq!(
            texture_gpu,
            texture.gpu_address(),
            "copy_from_texture texture_gpu must match the address used to create the texture"
        );
        let mtl_texture = self.resolve_texture(texture.id());
        let width = texture.desc().width;
        let height = texture.desc().height;
        let bpp = bytes_per_pixel(texture.desc().format)
            .expect("Unsupported texture format for copy_from_texture");
        let bytes_per_row = width as usize * bpp;
        let bytes_per_image = bytes_per_row * height as usize;
        let (dst_buffer, dst_offset) = self.resolve_buffer(dst, bytes_per_image as u64);

        self.end_active_encoders();
        // Metal 4 routes texture copies through the compute encoder.
        let encoder = self
            .command_buffer
            .computeCommandEncoder()
            .expect("Failed to create Metal 4 copy encoder");
        let size = MTLSize {
            width: width as usize,
            height: height as usize,
            depth: 1,
        };
        let origin = MTLOrigin { x: 0, y: 0, z: 0 };
        unsafe {
            encoder.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                &mtl_texture,
                0,
                0,
                origin,
                size,
                &dst_buffer,
                dst_offset as usize,
                bytes_per_row,
                bytes_per_image,
            );
        }
        encoder.endEncoding();
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
        if self.active_render_pass.is_some() {
            self.deferred_commands_mut()
                .push(MetalRenderCommand::SetViewport(viewport));
            return;
        }
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
        if self.active_render_pass.is_some() {
            self.deferred_commands_mut()
                .push(MetalRenderCommand::SetScissor(scissor));
            return;
        }
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
            frag_fn,
            has_texture_heap_slot,
            has_sampler_heap_slot,
            has_storage_heap_slot,
        ) = match &pso.inner {
            crate::pipeline::MeshletPsoInner::Metal(mtl_pso) => {
                let has_slot = |slot: usize| mtl_pso.argument_buffer_slots.contains(&slot);
                (
                    mtl_pso.default_pipeline.clone(),
                    mtl_pso.cull_mode,
                    mtl_pso.winding,
                    mtl_pso.root_constant_size,
                    mtl_pso.frag_fn.clone(),
                    has_slot(1),
                    has_slot(2),
                    has_slot(3),
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
        self.active_storage_heap_slot_enabled = has_storage_heap_slot;
        self.refresh_bindless_heaps(
            frag_fn.as_ref(),
            has_texture_heap_slot,
            has_sampler_heap_slot,
        );
        if has_storage_heap_slot {
            self.refresh_storage_heap(frag_fn.as_ref());
        }
        self.refresh_argument_table();

        let binding = MetalPipelineBinding {
            pipeline: pipeline.clone(),
            cull_mode,
            winding,
            topology: self.current_topology,
            root_constant_size,
            texture_heap_slot: has_texture_heap_slot,
            sampler_heap_slot: has_sampler_heap_slot,
            storage_heap_slot: has_storage_heap_slot,
            stages: MTLRenderStages::Mesh | MTLRenderStages::Fragment,
        };
        if self.active_render_pass.is_some() {
            self.deferred_commands_mut()
                .push(MetalRenderCommand::SetPipeline(binding));
            return;
        }
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
        if self.active_render_pass.is_some() {
            let root_table = self.current_root_table;
            self.deferred_commands_mut()
                .push(MetalRenderCommand::DrawMeshlets {
                    root_table,
                    groups,
                    object_tpg: tg_obj,
                    mesh_tpg: tg_mesh,
                });
            return;
        }
        let encoder = match self.render_encoder.as_ref() {
            Some(e) => e.clone(),
            None => {
                log::warn!("draw_meshlets: no active render encoder");
                return;
            }
        };
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
        if self.active_render_pass.is_some() {
            let root_table = self.current_root_table;
            self.deferred_commands_mut()
                .push(MetalRenderCommand::DrawMeshletsIndirect {
                    root_table,
                    args: args.0,
                    object_tpg: tg_obj,
                    mesh_tpg: tg_mesh,
                });
            return;
        }
        let encoder = match self.render_encoder.as_ref() {
            Some(e) => e.clone(),
            None => {
                log::warn!("draw_meshlets_indirect: no active render encoder");
                return;
            }
        };
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
                length: (desc.instance_count as u64)
                    * std::mem::size_of::<crate::types::TlasInstance>() as u64,
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

    // -- Ray tracing dispatch (Metal: compute kernel dispatch) --

    #[allow(clippy::too_many_arguments)]
    pub fn trace_rays(
        &mut self,
        pso: &RayTracingPso,
        raygen: &SbtRegion,
        miss: &SbtRegion,
        hit: &SbtRegion,
        width: u32,
        height: u32,
        depth: u32,
    ) -> RhiResult<()> {
        use objc2_metal::{
            MTL4ComputeCommandEncoder as _, MTLIntersectionFunctionTable as _,
            MTLVisibleFunctionTable as _,
        };

        self.end_active_encoders();
        let encoder = self
            .command_buffer
            .computeCommandEncoder()
            .ok_or_else(|| RhiError::CommandBuffer("Failed to create Metal RT encoder".into()))?;
        self.apply_pending_queue_barrier_compute(&encoder);

        let mtl_pso = match &pso.inner {
            crate::pipeline::RayTracingPsoInner::Metal(pso) => pso,
            #[allow(unreachable_patterns)]
            _ => {
                return Err(RhiError::Backend(
                    "trace_rays called with a non-Metal ray tracing pipeline".into(),
                ));
            }
        };

        encoder.setComputePipelineState(&mtl_pso.pipeline);
        self.current_threads_per_threadgroup = mtl_pso.threads_per_threadgroup;
        self.root_constant_size = mtl_pso.root_constant_size;

        let has_slot = |slot: usize| mtl_pso.compute_argument_buffer_slots.contains(&slot);
        let has_texture_heap_slot = has_slot(1);
        let has_sampler_heap_slot = has_slot(2);
        let has_storage_heap_slot = has_slot(3);
        self.active_texture_heap_slot_enabled = has_texture_heap_slot;
        self.active_sampler_heap_slot_enabled = has_sampler_heap_slot;
        self.active_storage_heap_slot_enabled = has_storage_heap_slot;
        self.refresh_bindless_heaps(
            mtl_pso.raygen_fn.as_ref(),
            has_texture_heap_slot,
            has_sampler_heap_slot,
        );
        if has_storage_heap_slot {
            self.refresh_storage_heap(mtl_pso.raygen_fn.as_ref());
        }

        let mut bytes = [0u8; RT_TRACE_TABLE_BYTES];
        write_sbt_region(&mut bytes[0..24], raygen);
        write_sbt_region(&mut bytes[24..48], miss);
        write_sbt_region(&mut bytes[48..72], hit);
        let intersection_table_id = mtl_pso
            .intersection_function_table
            .as_ref()
            .map(|table| table.gpuResourceID().to_raw())
            .unwrap_or(0);
        let visible_table_id = mtl_pso
            .visible_function_table
            .as_ref()
            .map(|table| table.gpuResourceID().to_raw())
            .unwrap_or(0);
        bytes[72..80].copy_from_slice(&intersection_table_id.to_ne_bytes());
        bytes[80..88].copy_from_slice(&visible_table_id.to_ne_bytes());
        let (rt_args_addr, rt_args_ptr) = self.alloc_root_bytes(RT_TRACE_TABLE_BYTES);
        self.refresh_argument_table();
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), rt_args_ptr, bytes.len());
            self.argument_table.setAddress_atIndex(rt_args_addr, 0);
        }
        encoder.setArgumentTable(Some(&self.argument_table));

        let tg = MTLSize {
            width: mtl_pso.threads_per_threadgroup[0] as usize,
            height: mtl_pso.threads_per_threadgroup[1] as usize,
            depth: mtl_pso.threads_per_threadgroup[2] as usize,
        };
        let groups = MTLSize {
            width: width.div_ceil(mtl_pso.threads_per_threadgroup[0]) as usize,
            height: height.div_ceil(mtl_pso.threads_per_threadgroup[1]) as usize,
            depth: depth.div_ceil(mtl_pso.threads_per_threadgroup[2]) as usize,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, tg);
        encoder.endEncoding();
        Ok(())
    }
}

fn write_sbt_region(dst: &mut [u8], region: &SbtRegion) {
    dst[0..8].copy_from_slice(&region.device_address.0.to_ne_bytes());
    dst[8..16].copy_from_slice(&region.stride.to_ne_bytes());
    dst[16..24].copy_from_slice(&region.size.to_ne_bytes());
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
        if let Some(storage_heap) = self.storage_heap_buffer.as_ref() {
            self.remove_allocation_from_residency(storage_heap.as_ref());
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
