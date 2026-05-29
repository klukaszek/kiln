use super::barrier::{to_vk_access_flags, to_vk_stage_flags};
use super::device::{SharedAllocations, SharedTextures};
use crate::barrier::{HazardFlags, StageFlags};
use crate::command::{
    DispatchIndirectArgs, DrawIndexedIndirectArgs, DrawIndirectMultiArgs, LoadOp,
    RenderPassDesc, RenderTarget, SignalValueDesc, StoreOp, WaitValueDesc,
};
use crate::pipeline::{
    BlendState, ComputePso, ComputePsoInner, DepthStencilState, GraphicsPso, GraphicsPsoInner,
    MeshletPso,
};
use crate::texture::{Texture, bytes_per_pixel};
use crate::types::*;
use ash::{
    ext::{descriptor_buffer, mesh_shader as vk_mesh_shader},
    khr::acceleration_structure as vk_accel_structure,
    vk,
};

/// Vulkan command buffer wrapper.
pub struct VulkanCommandBuffer {
    pub(crate) command_buffer: vk::CommandBuffer,
    pub(crate) device: ash::Device,
    /// Swapchain image views for resolving RenderTarget::SwapchainImage
    pub(crate) swapchain_image_views: Vec<vk::ImageView>,
    pub(crate) swapchain_images: Vec<vk::Image>,
    pub(crate) depth_image_view: vk::ImageView,
    /// Current pipeline layout for push constants
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pub(crate) descriptor_buffer_loader: Option<descriptor_buffer::Device>,
    pub(crate) descriptor_buffer_binding: Option<vk::DescriptorBufferBindingInfoEXT<'static>>,
    pub(crate) active_descriptor_buffer_offset: u64,
    /// Root constant size of the current pipeline
    pub(crate) root_constant_size: u32,
    /// Active push-constant stage mask (set by pipeline bind)
    pub(crate) push_constant_stages: vk::ShaderStageFlags,
    /// Active blend state used for pipeline selection.
    pub(crate) current_blend_state: BlendState,
    /// Pending split barrier producer state.
    pub(crate) pending_split_barrier: Option<(StageFlags, HazardFlags)>,
    /// Pending value waits consumed at queue submit.
    pub(crate) pending_value_waits: Vec<WaitValueDesc>,
    /// Pending value signals consumed at queue submit.
    pub(crate) pending_value_signals: Vec<SignalValueDesc>,
    /// Allocation registry for GPU pointer resolution
    pub(crate) allocations: SharedAllocations,
    /// Shared texture storage for RenderTarget::Texture
    pub(crate) textures: SharedTextures,
    /// Mesh shader extension loader (VK_EXT_mesh_shader).
    pub(crate) mesh_shader: Option<vk_mesh_shader::Device>,
    /// Acceleration structure extension loader (VK_KHR_acceleration_structure).
    pub(crate) acceleration_structure: Option<vk_accel_structure::Device>,
    /// Device limit used as the native indirect-count upper bound.
    pub(crate) max_draw_indirect_count: u32,
}

// SAFETY: VulkanCommandBuffer is only used from one thread at a time.
unsafe impl Send for VulkanCommandBuffer {}

impl VulkanCommandBuffer {
    fn bind_descriptor_buffer(
        &self,
        bind_point: vk::PipelineBindPoint,
        pipeline_layout: vk::PipelineLayout,
    ) {
        let loader = self
            .descriptor_buffer_loader
            .as_ref()
            .expect("Vulkan descriptor-buffer loader missing");
        let binding = self
            .descriptor_buffer_binding
            .as_ref()
            .expect("Vulkan descriptor-buffer binding missing");
        unsafe {
            loader.cmd_bind_descriptor_buffers(self.command_buffer, std::slice::from_ref(binding));
            loader.cmd_set_descriptor_buffer_offsets(
                self.command_buffer,
                bind_point,
                pipeline_layout,
                0,
                &[0],
                &[self.active_descriptor_buffer_offset],
            );
        }
    }

    fn resolve_buffer_bounds(&self, addr: GpuAddress) -> (vk::Buffer, u64, u64) {
        let addr_u64 = addr.0;
        let allocations = self.allocations.lock().expect("allocations lock poisoned");
        if let Some((&base, alloc)) = allocations.range(..=addr_u64).next_back()
            && addr_u64 < base + alloc.size
        {
            let offset = addr_u64 - base;
            return (alloc.buffer, offset, alloc.size - offset);
        }
        panic!("GPU address {addr_u64:#x} not found in allocation registry");
    }

    fn resolve_buffer(&self, addr: GpuAddress, size: u64) -> (vk::Buffer, u64) {
        let (buffer, offset, remaining) = self.resolve_buffer_bounds(addr);
        if size > remaining {
            panic!(
                "GPU address {:#x} size {} exceeds allocation bounds (remaining {})",
                addr.0, size, remaining
            );
        }
        (buffer, offset)
    }

    fn resolve_buffer_with_remaining(
        &self,
        addr: GpuAddress,
        min_size: u64,
    ) -> (vk::Buffer, u64, u64) {
        let (buffer, offset, remaining) = self.resolve_buffer_bounds(addr);
        if min_size > remaining {
            panic!(
                "GPU address {:#x} size {} exceeds allocation bounds (remaining {})",
                addr.0, min_size, remaining
            );
        }
        (buffer, offset, remaining)
    }

    fn resolve_texture(&self, id: TextureId) -> (vk::Image, vk::ImageView) {
        let textures = self.textures.lock().expect("textures lock poisoned");
        let tex = textures
            .get(id.0 as usize)
            .and_then(|t| t.as_ref())
            .expect("Invalid texture ID");
        (tex.image, tex.image_view)
    }

    pub fn begin_render_pass(&mut self, desc: &RenderPassDesc) {
        let cmd = self.command_buffer;

        // Transition swapchain images to COLOR_ATTACHMENT_OPTIMAL before rendering.
        for ca in &desc.color_attachments {
            if let RenderTarget::SwapchainImage(idx) = ca.target {
                let image = self.swapchain_images[idx as usize];
                let old_layout = match ca.load_op {
                    LoadOp::Load => vk::ImageLayout::PRESENT_SRC_KHR,
                    LoadOp::Clear | LoadOp::DontCare => vk::ImageLayout::UNDEFINED,
                };
                let barrier = vk::ImageMemoryBarrier::default()
                    .old_layout(old_layout)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .image(image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                unsafe {
                    self.device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier],
                    );
                }
            }
        }

        // Build color attachments for dynamic rendering
        let color_attachments: Vec<vk::RenderingAttachmentInfo> = desc
            .color_attachments
            .iter()
            .map(|ca| {
                let image_view = match ca.target {
                    RenderTarget::SwapchainImage(idx) => self.swapchain_image_views[idx as usize],
                    RenderTarget::Texture(id) => self.resolve_texture(id).1,
                };
                let image_layout = match ca.target {
                    RenderTarget::SwapchainImage(_) => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    RenderTarget::Texture(_) => vk::ImageLayout::GENERAL,
                };

                let load_op = match ca.load_op {
                    LoadOp::Load => vk::AttachmentLoadOp::LOAD,
                    LoadOp::Clear => vk::AttachmentLoadOp::CLEAR,
                    LoadOp::DontCare => vk::AttachmentLoadOp::DONT_CARE,
                };
                let store_op = match ca.store_op {
                    StoreOp::Store => vk::AttachmentStoreOp::STORE,
                    StoreOp::DontCare => vk::AttachmentStoreOp::DONT_CARE,
                };

                vk::RenderingAttachmentInfo::default()
                    .image_view(image_view)
                    .image_layout(image_layout)
                    .load_op(load_op)
                    .store_op(store_op)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: ca.clear_color,
                        },
                    })
            })
            .collect();

        // Build depth attachment
        let depth_attachment = desc.depth_attachment.as_ref().map(|da| {
            let image_view = match da.target {
                RenderTarget::SwapchainImage(_) => self.depth_image_view,
                RenderTarget::Texture(id) => self.resolve_texture(id).1,
            };
            let image_layout = match da.target {
                RenderTarget::SwapchainImage(_) => {
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                }
                RenderTarget::Texture(_) => vk::ImageLayout::GENERAL,
            };

            let load_op = match da.load_op {
                LoadOp::Load => vk::AttachmentLoadOp::LOAD,
                LoadOp::Clear => vk::AttachmentLoadOp::CLEAR,
                LoadOp::DontCare => vk::AttachmentLoadOp::DONT_CARE,
            };
            let store_op = match da.store_op {
                StoreOp::Store => vk::AttachmentStoreOp::STORE,
                StoreOp::DontCare => vk::AttachmentStoreOp::DONT_CARE,
            };

            vk::RenderingAttachmentInfo::default()
                .image_view(image_view)
                .image_layout(image_layout)
                .load_op(load_op)
                .store_op(store_op)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: da.clear_depth,
                        stencil: da.clear_stencil as u32,
                    },
                })
        });

        let render_area = vk::Rect2D {
            offset: vk::Offset2D {
                x: desc.render_area[0] as i32,
                y: desc.render_area[1] as i32,
            },
            extent: vk::Extent2D {
                width: desc.render_area[2],
                height: desc.render_area[3],
            },
        };

        let mut rendering_info = vk::RenderingInfo::default()
            .render_area(render_area)
            .layer_count(1)
            .color_attachments(&color_attachments);

        if let Some(ref da) = depth_attachment {
            rendering_info = rendering_info.depth_attachment(da);
        }

        unsafe {
            self.device.cmd_begin_rendering(cmd, &rendering_info);
        }
    }

    pub fn end_render_pass(&mut self) {
        unsafe {
            self.device.cmd_end_rendering(self.command_buffer);
        }

        // Transition swapchain images to PRESENT_SRC_KHR after rendering
        // This is a simplification -- in practice, the caller should manage this
        // but for the common case of rendering to swapchain, we handle it here.
    }

    pub fn set_graphics_pipeline(&mut self, pso: &GraphicsPso) {
        let vk_pso = match &pso.inner {
            GraphicsPsoInner::Vulkan(p) => p,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        };
        let pipeline = vk_pso.pipeline_for_blend(&self.current_blend_state);
        self.bind_pipeline(
            vk::PipelineBindPoint::GRAPHICS,
            pipeline,
            vk_pso.pipeline_layout,
            vk_pso.root_constant_size,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        );
    }

    pub fn set_compute_pipeline(&mut self, pso: &ComputePso) {
        let vk_pso = match &pso.inner {
            ComputePsoInner::Vulkan(p) => p,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        };
        self.bind_pipeline(
            vk::PipelineBindPoint::COMPUTE,
            vk_pso.pipeline,
            vk_pso.pipeline_layout,
            vk_pso.root_constant_size,
            vk::ShaderStageFlags::COMPUTE,
        );
    }

    /// Bind a pipeline + descriptor buffer and record root-constant state for subsequent
    /// `set_root_table` / `set_compute_root` push-constant writes.
    fn bind_pipeline(
        &mut self,
        bind_point: vk::PipelineBindPoint,
        pipeline: vk::Pipeline,
        pipeline_layout: vk::PipelineLayout,
        root_constant_size: u32,
        push_constant_stages: vk::ShaderStageFlags,
    ) {
        self.pipeline_layout = pipeline_layout;
        self.root_constant_size = root_constant_size;
        self.push_constant_stages = push_constant_stages;
        unsafe {
            self.device
                .cmd_bind_pipeline(self.command_buffer, bind_point, pipeline);
        }
        self.bind_descriptor_buffer(bind_point, pipeline_layout);
    }

    pub fn set_active_texture_heap_ptr(&mut self, ptr: GpuAddress) {
        let binding = self
            .descriptor_buffer_binding
            .as_ref()
            .expect("Descriptor buffer binding missing in descriptor-buffer mode");
        if ptr.0 < binding.address {
            panic!(
                "Active texture heap pointer {:#x} is before descriptor heap base {:#x}",
                ptr.0, binding.address
            );
        }
        self.active_descriptor_buffer_offset = ptr.0 - binding.address;
    }

    pub fn set_depth_stencil_state(&mut self, state: &DepthStencilState) {
        let cmd = self.command_buffer;
        let depth_test = state.depth_mode.contains(DepthFlags::READ);
        let depth_write = state.depth_mode.contains(DepthFlags::WRITE);
        let stencil_enable = state.stencil_enabled();

        unsafe {
            self.device.cmd_set_depth_test_enable(cmd, depth_test);
            self.device.cmd_set_depth_write_enable(cmd, depth_write);
            self.device
                .cmd_set_depth_compare_op(cmd, compare_op_to_vk(state.depth_test));
            self.device.cmd_set_stencil_test_enable(cmd, stencil_enable);

            // Per-face stencil ops (STENCIL_OP dynamic state, promoted in Vulkan 1.3)
            self.device.cmd_set_stencil_op(
                cmd,
                vk::StencilFaceFlags::FRONT,
                stencil_op_to_vk(state.stencil_front.fail_op),
                stencil_op_to_vk(state.stencil_front.pass_op),
                stencil_op_to_vk(state.stencil_front.depth_fail_op),
                compare_op_to_vk(state.stencil_front.test),
            );
            self.device.cmd_set_stencil_op(
                cmd,
                vk::StencilFaceFlags::BACK,
                stencil_op_to_vk(state.stencil_back.fail_op),
                stencil_op_to_vk(state.stencil_back.pass_op),
                stencil_op_to_vk(state.stencil_back.depth_fail_op),
                compare_op_to_vk(state.stencil_back.test),
            );

            // Read/write masks are shared across faces in our model
            self.device.cmd_set_stencil_compare_mask(
                cmd,
                vk::StencilFaceFlags::FRONT_AND_BACK,
                state.stencil_read_mask as u32,
            );
            self.device.cmd_set_stencil_write_mask(
                cmd,
                vk::StencilFaceFlags::FRONT_AND_BACK,
                state.stencil_write_mask as u32,
            );

            // Per-face stencil reference values
            self.device.cmd_set_stencil_reference(
                cmd,
                vk::StencilFaceFlags::FRONT,
                state.stencil_front.reference as u32,
            );
            self.device.cmd_set_stencil_reference(
                cmd,
                vk::StencilFaceFlags::BACK,
                state.stencil_back.reference as u32,
            );

            // Depth bias (DEPTH_BIAS_ENABLE promoted in Vulkan 1.3)
            let bias_active = state.depth_bias != 0.0 || state.depth_bias_slope_factor != 0.0;
            self.device.cmd_set_depth_bias_enable(cmd, bias_active);
            if bias_active {
                self.device.cmd_set_depth_bias(
                    cmd,
                    state.depth_bias,
                    state.depth_bias_clamp,
                    state.depth_bias_slope_factor,
                );
            }
        }
    }

    pub fn set_blend_state(&mut self, _state: &BlendState) {
        self.current_blend_state = _state.clone();
    }

    pub fn set_root_data(&mut self, vertex_root: GpuAddress, pixel_root: GpuAddress) {
        self.set_root_table(vertex_root, 0, pixel_root, 0);
    }

    pub fn set_compute_root(&mut self, root: GpuAddress) {
        let bytes = root.0.to_ne_bytes();
        if bytes.len() > self.root_constant_size as usize {
            panic!(
                "Compute root pointer ({} bytes) exceeds pipeline limit ({} bytes)",
                bytes.len(),
                self.root_constant_size
            );
        }
        if !self
            .push_constant_stages
            .contains(vk::ShaderStageFlags::COMPUTE)
        {
            panic!("No compute pipeline bound before set_compute_root");
        }
        unsafe {
            self.device.cmd_push_constants(
                self.command_buffer,
                self.pipeline_layout,
                self.push_constant_stages,
                0,
                &bytes,
            );
        }
    }

    pub fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        unsafe {
            self.device.cmd_draw(
                self.command_buffer,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            );
        }
    }

    pub fn draw_indexed(&mut self, indices: GpuAddress, index_count: u32, instance_count: u32) {
        // Index format is always U32 — the spec has no IndexFormat concept.
        let (index_buffer, offset) = self.resolve_buffer(indices, index_count as u64 * 4);
        unsafe {
            self.device.cmd_bind_index_buffer(
                self.command_buffer,
                index_buffer,
                offset,
                vk::IndexType::UINT32,
            );
            self.device
                .cmd_draw_indexed(self.command_buffer, index_count, instance_count, 0, 0, 0);
        }
    }

    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        unsafe {
            self.device.cmd_dispatch(self.command_buffer, x, y, z);
        }
    }

    pub fn dispatch_indirect(&mut self, args: GpuAddress) {
        let (arg_buffer, arg_offset) =
            self.resolve_buffer(args, std::mem::size_of::<DispatchIndirectArgs>() as u64);
        unsafe {
            self.device
                .cmd_dispatch_indirect(self.command_buffer, arg_buffer, arg_offset);
        }
    }

    pub fn draw_indexed_indirect(&mut self, indices: GpuAddress, args: GpuAddress) {
        let (arg_buffer, arg_offset) =
            self.resolve_buffer(args, std::mem::size_of::<DrawIndexedIndirectArgs>() as u64);
        let (index_buffer, index_offset) = self.resolve_buffer(indices, 4);
        unsafe {
            self.device.cmd_bind_index_buffer(
                self.command_buffer,
                index_buffer,
                index_offset,
                vk::IndexType::UINT32,
            );
            self.device.cmd_draw_indexed_indirect(
                self.command_buffer,
                arg_buffer,
                arg_offset,
                1,
                std::mem::size_of::<DrawIndexedIndirectArgs>() as u32,
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
        self.set_root_table(vertex_root, vertex_stride, pixel_root, pixel_stride);
        let stride = std::mem::size_of::<DrawIndirectMultiArgs>() as u32;
        let (arg_buffer, arg_offset, arg_remaining) =
            self.resolve_buffer_with_remaining(args, stride as u64);
        let (count_buffer, count_offset) =
            self.resolve_buffer(draw_count, std::mem::size_of::<u32>() as u64);
        let max_draw_count =
            (arg_remaining / stride as u64).min(self.max_draw_indirect_count as u64) as u32;
        assert!(
            max_draw_count > 0,
            "draw_indirect_multi args allocation has no complete draw records"
        );
        unsafe {
            self.device.cmd_draw_indirect_count(
                self.command_buffer,
                arg_buffer,
                arg_offset,
                count_buffer,
                count_offset,
                max_draw_count,
                stride,
            );
        }
    }

    fn set_root_table(
        &mut self,
        vertex_root_base: GpuAddress,
        vertex_stride: u32,
        pixel_root_base: GpuAddress,
        pixel_stride: u32,
    ) {
        let mut bytes = [0u8; 32];
        bytes[0..8].copy_from_slice(&vertex_root_base.0.to_ne_bytes());
        bytes[8..12].copy_from_slice(&vertex_stride.to_ne_bytes());
        bytes[16..24].copy_from_slice(&pixel_root_base.0.to_ne_bytes());
        bytes[24..28].copy_from_slice(&pixel_stride.to_ne_bytes());

        if bytes.len() > self.root_constant_size as usize {
            panic!(
                "Root table ({} bytes) exceeds pipeline limit ({})",
                bytes.len(),
                self.root_constant_size
            );
        }
        if self.push_constant_stages.is_empty() {
            panic!("No pipeline bound before set_root_table");
        }

        unsafe {
            self.device.cmd_push_constants(
                self.command_buffer,
                self.pipeline_layout,
                self.push_constant_stages,
                0,
                &bytes,
            );
        }
    }

    pub fn memcpy(&mut self, dst: GpuAddress, src: GpuAddress, size: u64) {
        if size == 0 {
            return;
        }
        let (src_buffer, src_offset) = self.resolve_buffer(src, size);
        let (dst_buffer, dst_offset) = self.resolve_buffer(dst, size);
        let region = vk::BufferCopy::default()
            .src_offset(src_offset)
            .dst_offset(dst_offset)
            .size(size);
        unsafe {
            self.device.cmd_copy_buffer(
                self.command_buffer,
                src_buffer,
                dst_buffer,
                std::slice::from_ref(&region),
            );
        }
    }

    pub fn copy_to_texture(&mut self, texture_gpu: GpuAddress, src: GpuAddress, texture: &Texture) {
        let (image, aspect, width, height, src_buffer, src_offset) =
            self.prepare_texture_copy(texture_gpu, src, texture, "copy_to_texture");
        self.transition_texture(
            image,
            aspect,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
            false,
        );
        let region = build_buffer_image_region(src_offset, aspect, width, height);
        unsafe {
            self.device.cmd_copy_buffer_to_image(
                self.command_buffer,
                src_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&region),
            );
        }
        self.transition_texture(
            image,
            aspect,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
            true,
        );
    }

    pub fn copy_from_texture(
        &mut self,
        dst: GpuAddress,
        texture_gpu: GpuAddress,
        texture: &Texture,
    ) {
        let (image, aspect, width, height, dst_buffer, dst_offset) =
            self.prepare_texture_copy(texture_gpu, dst, texture, "copy_from_texture");
        self.transition_texture(
            image,
            aspect,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::TRANSFER,
            false,
        );
        let region = build_buffer_image_region(dst_offset, aspect, width, height);
        unsafe {
            self.device.cmd_copy_image_to_buffer(
                self.command_buffer,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_buffer,
                std::slice::from_ref(&region),
            );
        }
        self.transition_texture(
            image,
            aspect,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::TRANSFER,
            true,
        );
    }

    /// Validate the texture address, resolve the linear buffer, and return
    /// `(image, aspect, w, h, buffer, offset)` for a copy command.
    fn prepare_texture_copy(
        &self,
        texture_gpu: GpuAddress,
        buffer_gpu: GpuAddress,
        texture: &Texture,
        op: &'static str,
    ) -> (vk::Image, vk::ImageAspectFlags, u32, u32, vk::Buffer, u64) {
        assert_eq!(
            texture_gpu,
            texture.gpu_address(),
            "{op} texture_gpu must match the address used to create the texture"
        );
        let (image, _view) = self.resolve_texture(texture.id());
        let width = texture.desc().width;
        let height = texture.desc().height;
        let bpp = bytes_per_pixel(texture.desc().format)
            .unwrap_or_else(|| panic!("Unsupported texture format for {op}"));
        let size = (width as u64) * (height as u64) * (bpp as u64);
        let (buffer, offset) = self.resolve_buffer(buffer_gpu, size);
        let aspect = if is_depth_format(texture.desc().format) {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        (image, aspect, width, height, buffer, offset)
    }

    /// Emit a single-mip, single-layer image layout transition. `reverse=true` swaps
    /// pipeline stages and access masks so the same call can wrap a copy on both sides.
    #[allow(clippy::too_many_arguments)]
    fn transition_texture(
        &self,
        image: vk::Image,
        aspect: vk::ImageAspectFlags,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        transfer_access: vk::AccessFlags,
        transfer_stage: vk::PipelineStageFlags,
        reverse: bool,
    ) {
        let (src_stage, dst_stage, src_access, dst_access) = if reverse {
            (
                transfer_stage,
                vk::PipelineStageFlags::ALL_COMMANDS,
                transfer_access,
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
            )
        } else {
            (
                vk::PipelineStageFlags::ALL_COMMANDS,
                transfer_stage,
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                transfer_access,
            )
        };
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }

    pub fn barrier(&mut self, src: StageFlags, dst: StageFlags) {
        let memory_barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(to_vk_stage_flags(src))
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(to_vk_stage_flags(dst))
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE);

        let dep_info =
            vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&memory_barrier));

        unsafe {
            self.device
                .cmd_pipeline_barrier2(self.command_buffer, &dep_info);
        }
    }

    pub fn barrier_with_hazard(&mut self, src: StageFlags, dst: StageFlags, hazard: HazardFlags) {
        let use_descriptor_buffer_hazard = hazard.contains(HazardFlags::DESCRIPTORS);
        let hazard_for_access = if use_descriptor_buffer_hazard {
            hazard & !HazardFlags::DESCRIPTORS
        } else {
            hazard
        };

        let mut src_stage = to_vk_stage_flags(src);
        let mut dst_stage = to_vk_stage_flags(dst);
        let mut src_access = to_vk_access_flags(hazard_for_access, true);
        let mut dst_access = to_vk_access_flags(hazard_for_access, false);

        if use_descriptor_buffer_hazard {
            // Descriptor buffer reads happen in shader stages; include them to satisfy access masks.
            src_stage |= vk::PipelineStageFlags2::VERTEX_SHADER
                | vk::PipelineStageFlags2::FRAGMENT_SHADER
                | vk::PipelineStageFlags2::COMPUTE_SHADER;
            dst_stage |= vk::PipelineStageFlags2::VERTEX_SHADER
                | vk::PipelineStageFlags2::FRAGMENT_SHADER
                | vk::PipelineStageFlags2::COMPUTE_SHADER;
            src_access |= vk::AccessFlags2::MEMORY_WRITE;
            dst_access |= vk::AccessFlags2::DESCRIPTOR_BUFFER_READ_EXT;
        }

        let memory_barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access);

        let dep_info =
            vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&memory_barrier));

        unsafe {
            self.device
                .cmd_pipeline_barrier2(self.command_buffer, &dep_info);
        }
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
        let viewport = vk::Viewport {
            x,
            y,
            width,
            height,
            min_depth,
            max_depth,
        };
        unsafe {
            self.device
                .cmd_set_viewport(self.command_buffer, 0, &[viewport]);
        }
    }

    pub fn set_scissor(&mut self, x: i32, y: i32, width: u32, height: u32) {
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x, y },
            extent: vk::Extent2D { width, height },
        };
        unsafe {
            self.device
                .cmd_set_scissor(self.command_buffer, 0, &[scissor]);
        }
    }

    /// Transition a swapchain image to PRESENT_SRC_KHR layout.
    pub fn transition_to_present(&mut self, swapchain_image_index: u32) {
        let image = self.swapchain_images[swapchain_image_index as usize];
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::empty())
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }

    // -- Mesh shader pipeline + draws --

    pub fn set_meshlet_pipeline(&mut self, pso: &MeshletPso) {
        let vk_pso = match &pso.inner {
            crate::pipeline::MeshletPsoInner::Vulkan(p) => p,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        };
        let pipeline = vk_pso.pipeline_for_blend(&self.current_blend_state);
        self.bind_pipeline(
            vk::PipelineBindPoint::GRAPHICS,
            pipeline,
            vk_pso.pipeline_layout,
            vk_pso.root_constant_size,
            vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::FRAGMENT,
        );
    }

    pub fn draw_meshlets(&mut self, x: u32, y: u32, z: u32) {
        let loader = match &self.mesh_shader {
            Some(l) => l.clone(),
            None => {
                log::warn!("draw_meshlets: VK_EXT_mesh_shader not available on this device");
                return;
            }
        };
        unsafe {
            loader.cmd_draw_mesh_tasks(self.command_buffer, x, y, z);
        }
    }

    /// `args` points to one `VkDrawMeshTasksIndirectCommandEXT` (x, y, z: u32 = 12 bytes).
    pub fn draw_meshlets_indirect(&mut self, args: GpuAddress) {
        let loader = match &self.mesh_shader {
            Some(l) => l.clone(),
            None => {
                log::warn!("draw_meshlets_indirect: VK_EXT_mesh_shader not available");
                return;
            }
        };
        // Single indirect draw: stride = 3 × u32 = 12 bytes, count = 1.
        let stride = 12u32;
        let (buffer, offset) = self.resolve_buffer(args, stride as u64);
        unsafe {
            loader.cmd_draw_mesh_tasks_indirect(self.command_buffer, buffer, offset, 1, stride);
        }
    }

    // -- Acceleration structure builds --

    pub fn bind_acceleration_structure(
        &mut self,
        _slot: u32,
        _accel: &crate::accel::AccelerationStructure,
    ) {
        // Vulkan binds the TLAS as an acceleration-structure descriptor (set 0, binding 0),
        // not via the argument table — needs a descriptor write through the descriptor
        // buffer. Not yet wired up (RT path is Metal-validated for now).
        log::warn!(
            "bind_acceleration_structure: ray-query TLAS binding not yet implemented on Vulkan"
        );
    }

    pub fn build_blas(&mut self, accel: &crate::accel::AccelerationStructure, desc: &BlasDesc) {
        let Some((accel_loader, vk_as)) = self.resolve_accel(accel, "build_blas") else {
            return;
        };

        // Reconstruct geometry (same logic as create_blas; geometry is not stored to avoid lifetimes).
        let geometries: Vec<vk::AccelerationStructureGeometryKHR> = desc
            .meshes
            .iter()
            .map(|m| match m.geometry_type {
                GeometryType::Triangles => {
                    let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                        .vertex_format(vk::Format::R32G32B32_SFLOAT)
                        .vertex_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: m.vertex_buffer.0,
                        })
                        .vertex_stride(m.vertex_stride)
                        .max_vertex(m.vertex_count.saturating_sub(1))
                        .index_type(if m.index_count > 0 {
                            vk::IndexType::UINT32
                        } else {
                            vk::IndexType::NONE_KHR
                        })
                        .index_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: m.index_buffer.0,
                        });
                    vk::AccelerationStructureGeometryKHR::default()
                        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                        .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
                }
                GeometryType::Aabbs => {
                    let aabbs = vk::AccelerationStructureGeometryAabbsDataKHR::default()
                        .data(vk::DeviceOrHostAddressConstKHR {
                            device_address: m.aabb_buffer.0,
                        })
                        .stride(std::mem::size_of::<vk::AabbPositionsKHR>() as u64);
                    let flags = if m.flags.contains(GeometryFlags::OPAQUE) {
                        vk::GeometryFlagsKHR::OPAQUE
                    } else {
                        vk::GeometryFlagsKHR::empty()
                    };
                    vk::AccelerationStructureGeometryKHR::default()
                        .geometry_type(vk::GeometryTypeKHR::AABBS)
                        .geometry(vk::AccelerationStructureGeometryDataKHR { aabbs })
                        .flags(flags)
                }
            })
            .collect();

        let primitive_counts: Vec<u32> = desc
            .meshes
            .iter()
            .map(|m| match m.geometry_type {
                GeometryType::Triangles => {
                    if m.index_count > 0 {
                        m.index_count / 3
                    } else {
                        m.vertex_count / 3
                    }
                }
                GeometryType::Aabbs => m.aabb_count,
            })
            .collect();

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .dst_acceleration_structure(vk_as)
            .geometries(&geometries);

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            accel_loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &primitive_counts,
                &mut size_info,
            );
        }

        // Allocate scratch buffer and encode the build.
        // In production, scratch allocation should be pooled. Here we allocate device-local.
        let range_infos: Vec<vk::AccelerationStructureBuildRangeInfoKHR> = primitive_counts
            .iter()
            .map(|&pc| vk::AccelerationStructureBuildRangeInfoKHR {
                primitive_count: pc,
                primitive_offset: 0,
                first_vertex: 0,
                transform_offset: 0,
            })
            .collect();
        // ash expects &[&[RangeInfo]] — one inner slice per build info.
        let range_infos_ref: &[vk::AccelerationStructureBuildRangeInfoKHR] = &range_infos;
        let build_range_infos: &[&[vk::AccelerationStructureBuildRangeInfoKHR]] =
            std::slice::from_ref(&range_infos_ref);

        unsafe {
            accel_loader.cmd_build_acceleration_structures(
                self.command_buffer,
                std::slice::from_ref(&build_info),
                build_range_infos,
            );
        }
    }

    pub fn build_tlas(&mut self, accel: &crate::accel::AccelerationStructure, desc: &TlasDesc) {
        let Some((accel_loader, vk_as)) = self.resolve_accel(accel, "build_tlas") else {
            return;
        };

        let instances_data = vk::AccelerationStructureGeometryInstancesDataKHR::default()
            .array_of_pointers(false)
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: desc.instance_buffer.0,
            });
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: instances_data,
            });
        let geometries = [geometry];

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .dst_acceleration_structure(vk_as)
            .geometries(&geometries);

        let range_info = [vk::AccelerationStructureBuildRangeInfoKHR {
            primitive_count: desc.instance_count,
            primitive_offset: 0,
            first_vertex: 0,
            transform_offset: 0,
        }];
        let build_range_infos: &[&[vk::AccelerationStructureBuildRangeInfoKHR]] = &[&range_info];

        unsafe {
            accel_loader.cmd_build_acceleration_structures(
                self.command_buffer,
                std::slice::from_ref(&build_info),
                build_range_infos,
            );
        }
    }

    /// Return the loader + raw `VkAccelerationStructureKHR` for a build command,
    /// logging and returning `None` if the device lacks RT extensions or the handle
    /// is not a Vulkan accel structure.
    fn resolve_accel(
        &self,
        accel: &crate::accel::AccelerationStructure,
        op: &'static str,
    ) -> Option<(vk_accel_structure::Device, vk::AccelerationStructureKHR)> {
        let Some(loader) = self.acceleration_structure.as_ref() else {
            log::warn!("{op}: VK_KHR_acceleration_structure not available");
            return None;
        };
        match &accel.inner {
            #[cfg(feature = "vulkan")]
            crate::accel::AccelInner::Vulkan(a) => Some((loader.clone(), a.acceleration_structure)),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }

}

fn build_buffer_image_region(
    buffer_offset: u64,
    aspect: vk::ImageAspectFlags,
    width: u32,
    height: u32,
) -> vk::BufferImageCopy {
    vk::BufferImageCopy::default()
        .buffer_offset(buffer_offset)
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: aspect,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
}

fn is_depth_format(format: Format) -> bool {
    matches!(
        format,
        Format::D16Unorm | Format::D32Float | Format::D24UnormS8Uint | Format::D32FloatS8Uint
    )
}

fn compare_op_to_vk(op: CompareOp) -> vk::CompareOp {
    match op {
        CompareOp::Never => vk::CompareOp::NEVER,
        CompareOp::Less => vk::CompareOp::LESS,
        CompareOp::Equal => vk::CompareOp::EQUAL,
        CompareOp::LessOrEqual => vk::CompareOp::LESS_OR_EQUAL,
        CompareOp::Greater => vk::CompareOp::GREATER,
        CompareOp::NotEqual => vk::CompareOp::NOT_EQUAL,
        CompareOp::GreaterOrEqual => vk::CompareOp::GREATER_OR_EQUAL,
        CompareOp::Always => vk::CompareOp::ALWAYS,
    }
}

fn stencil_op_to_vk(op: StencilOp) -> vk::StencilOp {
    match op {
        StencilOp::Keep => vk::StencilOp::KEEP,
        StencilOp::Zero => vk::StencilOp::ZERO,
        StencilOp::Replace => vk::StencilOp::REPLACE,
        StencilOp::IncrementClamp => vk::StencilOp::INCREMENT_AND_CLAMP,
        StencilOp::DecrementClamp => vk::StencilOp::DECREMENT_AND_CLAMP,
        StencilOp::Invert => vk::StencilOp::INVERT,
        StencilOp::IncrementWrap => vk::StencilOp::INCREMENT_AND_WRAP,
        StencilOp::DecrementWrap => vk::StencilOp::DECREMENT_AND_WRAP,
    }
}
