use super::barrier::{to_vk_access_flags, to_vk_stage_flags};
use super::device::{
    SharedAllocations, SharedTextures, build_accel_flags_to_vk, geometry_flags_to_vk,
};
use super::texture::texture_aspect;
use crate::barrier::{HazardFlags, StageFlags};
use crate::command::{
    DispatchIndirectArgs, DrawIndexedIndirectArgs, DrawIndirectArgs, LoadOp, RenderPassDesc,
    RenderTargetKind, StoreOp,
};
use crate::pipeline::{ComputePso, ComputePsoInner, GraphicsPso, GraphicsPsoInner, MeshletPso};
use crate::texture::{ResolvedRegion, Texture, bytes_per_pixel};
use crate::types::{BlasDesc, GeometryType, GpuPtr, TextureId, TlasDesc};
use ash::{
    ext::{debug_utils, descriptor_buffer, mesh_shader as vk_mesh_shader},
    khr::acceleration_structure as vk_accel_structure,
    vk,
};
use smallvec::SmallVec;
use std::rc::Rc;

/// Vulkan command buffer wrapper.
pub struct VulkanCommandBuffer {
    pub(crate) command_buffer: vk::CommandBuffer,
    pub(crate) device: ash::Device,
    pub(crate) swapchain_image_views: Rc<[vk::ImageView]>,
    pub(crate) swapchain_images: Rc<[vk::Image]>,
    pub(crate) depth_image_view: vk::ImageView,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pub(crate) descriptor_buffer_loader: descriptor_buffer::Device,
    pub(crate) descriptor_buffer_binding: vk::DescriptorBufferBindingInfoEXT<'static>,
    pub(crate) push_constant_stages: vk::ShaderStageFlags,
    pub(crate) debug_labels: Option<debug_utils::Device>,
    /// Set when the open pass pushed a label region for `end_render_pass` to pop.
    pub(crate) in_labelled_pass: bool,
    pub(crate) pending_split_barrier: Option<(StageFlags, HazardFlags)>,
    pub(crate) allocations: SharedAllocations,
    pub(crate) textures: SharedTextures,
    pub(crate) mesh_shader: Option<vk_mesh_shader::Device>,
    pub(crate) acceleration_structure: Option<vk_accel_structure::Device>,
    pub(crate) rendered_swapchain_images: SmallVec<[u32; 4]>,
    pub(crate) ended: bool,
}

impl VulkanCommandBuffer {
    pub(crate) fn finish(&mut self) -> crate::error::RhiResult<()> {
        if self.ended {
            return Ok(());
        }
        // Here rather than in `submit_frame`, so it holds whether or not the caller ends first.
        for index in std::mem::take(&mut self.rendered_swapchain_images) {
            self.transition_to_present(index);
        }
        unsafe {
            self.device
                .end_command_buffer(self.command_buffer)
                .map_err(|error| crate::error::RhiError::CommandBuffer(error.to_string()))?;
        }
        self.ended = true;
        Ok(())
    }

    fn bind_descriptor_buffer(
        &self,
        bind_point: vk::PipelineBindPoint,
        pipeline_layout: vk::PipelineLayout,
    ) {
        let loader = &self.descriptor_buffer_loader;
        let binding = &self.descriptor_buffer_binding;
        unsafe {
            loader.cmd_bind_descriptor_buffers(self.command_buffer, std::slice::from_ref(binding));
            loader.cmd_set_descriptor_buffer_offsets(
                self.command_buffer,
                bind_point,
                pipeline_layout,
                0,
                &[0],
                &[0],
            );
        }
    }

    fn resolve_buffer_bounds(&self, addr: GpuPtr<u8>) -> (vk::Buffer, u64, u64) {
        let addr_u64 = addr.address;
        let allocations = self.allocations.borrow();
        if let Some((&base, alloc)) = allocations.range(..=addr_u64).next_back() {
            let offset = addr_u64 - base;
            if offset < alloc.size {
                return (alloc.buffer, offset, alloc.size - offset);
            }
        }
        panic!("GPU address {addr_u64:#x} not found in allocation registry");
    }

    fn resolve_buffer(&self, addr: GpuPtr<u8>, size: u64) -> (vk::Buffer, u64) {
        let (buffer, offset, remaining) = self.resolve_buffer_bounds(addr);
        if size > remaining {
            panic!(
                "GPU address {:#x} size {} exceeds allocation bounds (remaining {})",
                addr.address, size, remaining
            );
        }
        (buffer, offset)
    }

    fn resolve_texture_info(&self, id: TextureId) -> (vk::Image, vk::ImageView, vk::ImageLayout) {
        let textures = self.textures.borrow();
        let tex = textures
            .get(id.0 as usize)
            .and_then(|t| t.as_ref())
            .expect("Invalid texture ID");
        (tex.image, tex.image_view, tex.layout)
    }

    pub fn begin_render_pass(&mut self, desc: &RenderPassDesc<'_>) {
        let cmd = self.command_buffer;

        // Opened before the attachment transitions so the capture attributes them to this pass.
        self.in_labelled_pass = false;
        if let (Some(loader), Some(label)) = (self.debug_labels.as_ref(), desc.label)
            && let Ok(label) = std::ffi::CString::new(label)
        {
            let info = vk::DebugUtilsLabelEXT::default().label_name(&label);
            unsafe { loader.cmd_begin_debug_utils_label(cmd, &info) };
            self.in_labelled_pass = true;
        }

        for ca in desc.color_attachments {
            if let RenderTargetKind::SwapchainImage(idx) = ca.target.kind() {
                let image = self.swapchain_images[idx as usize];
                let first_pass = !self.rendered_swapchain_images.contains(&idx);
                // The first pass starts from UNDEFINED or PRESENT; later passes resume from the
                // previous color-attachment write.
                let (old_layout, src_stage, src_access, extra_dst_access) = if first_pass {
                    let old = match ca.load_op {
                        LoadOp::Load => vk::ImageLayout::PRESENT_SRC_KHR,
                        LoadOp::Clear | LoadOp::DontCare => vk::ImageLayout::UNDEFINED,
                    };
                    (
                        old,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::empty(),
                    )
                } else {
                    (
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        vk::AccessFlags::COLOR_ATTACHMENT_READ,
                    )
                };
                let barrier = vk::ImageMemoryBarrier::default()
                    .old_layout(old_layout)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .src_access_mask(src_access)
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | extra_dst_access)
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
                        src_stage,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier],
                    );
                }
                if first_pass {
                    self.rendered_swapchain_images.push(idx);
                }
            }
        }

        let color_attachments: SmallVec<[vk::RenderingAttachmentInfo; 4]> = desc
            .color_attachments
            .iter()
            .map(|ca| {
                let (image_view, image_layout) = match ca.target.kind() {
                    RenderTargetKind::SwapchainImage(idx) => (
                        self.swapchain_image_views[idx as usize],
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    ),
                    RenderTargetKind::Texture(id) => {
                        let (_, image_view, image_layout) = self.resolve_texture_info(id);
                        (image_view, image_layout)
                    }
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

        let depth_attachment = desc.depth_attachment.as_ref().map(|da| {
            let (image_view, image_layout) = match da.target.kind() {
                RenderTargetKind::SwapchainImage(_) => (
                    self.depth_image_view,
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                ),
                RenderTargetKind::Texture(id) => {
                    let (_, image_view, image_layout) = self.resolve_texture_info(id);
                    (image_view, image_layout)
                }
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
                        stencil: 0,
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

            self.device.cmd_set_depth_bias_enable(cmd, false);
        }
    }

    pub fn end_render_pass(&mut self) {
        unsafe {
            self.device.cmd_end_rendering(self.command_buffer);
        }
        if self.in_labelled_pass
            && let Some(loader) = self.debug_labels.as_ref()
        {
            unsafe { loader.cmd_end_debug_utils_label(self.command_buffer) };
            self.in_labelled_pass = false;
        }
    }

    pub fn set_graphics_pipeline(&mut self, pso: &GraphicsPso) {
        let vk_pso = backend_expect!(&pso.inner, GraphicsPsoInner::Vulkan);
        self.bind_pipeline(
            vk::PipelineBindPoint::GRAPHICS,
            vk_pso.pipeline,
            vk_pso.pipeline_layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        );
    }

    pub fn set_compute_pipeline(&mut self, pso: &ComputePso) {
        let vk_pso = backend_expect!(&pso.inner, ComputePsoInner::Vulkan);
        self.bind_pipeline(
            vk::PipelineBindPoint::COMPUTE,
            vk_pso.pipeline,
            vk_pso.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
        );
    }

    fn bind_pipeline(
        &mut self,
        bind_point: vk::PipelineBindPoint,
        pipeline: vk::Pipeline,
        pipeline_layout: vk::PipelineLayout,
        push_constant_stages: vk::ShaderStageFlags,
    ) {
        self.pipeline_layout = pipeline_layout;
        self.push_constant_stages = push_constant_stages;
        unsafe {
            self.device
                .cmd_bind_pipeline(self.command_buffer, bind_point, pipeline);
        }
        self.bind_descriptor_buffer(bind_point, pipeline_layout);
    }

    pub fn set_root_data(&mut self, root: GpuPtr<u8>) {
        let bytes = root.address.to_ne_bytes();
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

    pub fn draw_indexed(&mut self, indices: GpuPtr<u8>, index_count: u32, instance_count: u32) {
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

    pub fn dispatch_indirect(&mut self, args: GpuPtr<u8>) {
        let (arg_buffer, arg_offset) =
            self.resolve_buffer(args, std::mem::size_of::<DispatchIndirectArgs>() as u64);
        unsafe {
            self.device
                .cmd_dispatch_indirect(self.command_buffer, arg_buffer, arg_offset);
        }
    }

    pub fn draw_indirect(&mut self, args: GpuPtr<u8>) {
        let (arg_buffer, arg_offset) =
            self.resolve_buffer(args, std::mem::size_of::<DrawIndirectArgs>() as u64);
        unsafe {
            self.device.cmd_draw_indirect(
                self.command_buffer,
                arg_buffer,
                arg_offset,
                1,
                std::mem::size_of::<DrawIndirectArgs>() as u32,
            );
        }
    }

    pub fn draw_indexed_indirect(
        &mut self,
        indices: GpuPtr<u8>,
        _max_index_count: u32,
        args: GpuPtr<u8>,
    ) {
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

    pub fn memcpy(&mut self, dst: GpuPtr<u8>, src: GpuPtr<u8>, size: u64) {
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

    pub fn copy_buffer_to_texture(
        &mut self,
        src: GpuPtr<u8>,
        texture: &Texture,
        region: ResolvedRegion,
    ) {
        let (image, aspect, layout, src_buffer, src_offset) =
            self.prepare_texture_copy(src, texture, region, "copy_buffer_to_texture");
        self.transition_texture(
            image,
            aspect,
            region,
            layout,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
            false,
        );
        let copy = build_buffer_image_region(src_offset, aspect, region);
        unsafe {
            self.device.cmd_copy_buffer_to_image(
                self.command_buffer,
                src_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&copy),
            );
        }
        self.transition_texture(
            image,
            aspect,
            region,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            layout,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
            true,
        );
    }

    pub fn copy_texture_to_buffer(
        &mut self,
        dst: GpuPtr<u8>,
        texture: &Texture,
        region: ResolvedRegion,
    ) {
        let (image, aspect, layout, dst_buffer, dst_offset) =
            self.prepare_texture_copy(dst, texture, region, "copy_texture_to_buffer");
        self.transition_texture(
            image,
            aspect,
            region,
            layout,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::TRANSFER,
            false,
        );
        let copy = build_buffer_image_region(dst_offset, aspect, region);
        unsafe {
            self.device.cmd_copy_image_to_buffer(
                self.command_buffer,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_buffer,
                std::slice::from_ref(&copy),
            );
        }
        self.transition_texture(
            image,
            aspect,
            region,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            layout,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::TRANSFER,
            true,
        );
    }

    pub fn copy_texture_to_texture(
        &mut self,
        src: &Texture,
        src_region: ResolvedRegion,
        dst: &Texture,
        dst_region: ResolvedRegion,
    ) {
        let (src_image, src_view_layout, src_aspect) = {
            let (image, _, layout) = self.resolve_texture_info(src.id());
            (image, layout, texture_aspect(src.desc().format))
        };
        let (dst_image, dst_view_layout, dst_aspect) = {
            let (image, _, layout) = self.resolve_texture_info(dst.id());
            (image, layout, texture_aspect(dst.desc().format))
        };
        self.transition_texture(
            src_image,
            src_aspect,
            src_region,
            src_view_layout,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::TRANSFER,
            false,
        );
        self.transition_texture(
            dst_image,
            dst_aspect,
            dst_region,
            dst_view_layout,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
            false,
        );
        let copy = vk::ImageCopy::default()
            .src_subresource(subresource_layers(src_aspect, src_region))
            .src_offset(to_vk_offset(src_region.origin))
            .dst_subresource(subresource_layers(dst_aspect, dst_region))
            .dst_offset(to_vk_offset(dst_region.origin))
            .extent(to_vk_extent(src_region.extent));
        unsafe {
            self.device.cmd_copy_image(
                self.command_buffer,
                src_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&copy),
            );
        }
        self.transition_texture(
            src_image,
            src_aspect,
            src_region,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            src_view_layout,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::TRANSFER,
            true,
        );
        self.transition_texture(
            dst_image,
            dst_aspect,
            dst_region,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            dst_view_layout,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
            true,
        );
    }

    /// Resolve the texture and linear buffer, and return
    /// `(image, aspect, w, h, current layout, buffer, offset)` for a copy command.
    fn prepare_texture_copy(
        &self,
        buffer_gpu: GpuPtr<u8>,
        texture: &Texture,
        region: ResolvedRegion,
        op: &'static str,
    ) -> (
        vk::Image,
        vk::ImageAspectFlags,
        vk::ImageLayout,
        vk::Buffer,
        u64,
    ) {
        let (image, _view, layout) = self.resolve_texture_info(texture.id());
        let bpp = bytes_per_pixel(texture.desc().format)
            .unwrap_or_else(|| panic!("Unsupported texture format for {op}"));
        let (_, bytes_per_image) = region.linear_strides(bpp);
        let size = (bytes_per_image as u64) * (region.extent[2] as u64);
        let (buffer, offset) = self.resolve_buffer(buffer_gpu, size);
        (
            image,
            texture_aspect(texture.desc().format),
            layout,
            buffer,
            offset,
        )
    }

    /// Transition just the mip and layer a copy touches. `reverse=true` swaps pipeline stages
    /// and access masks so the same call can wrap a copy on both sides.
    #[allow(clippy::too_many_arguments)]
    fn transition_texture(
        &self,
        image: vk::Image,
        aspect: vk::ImageAspectFlags,
        region: ResolvedRegion,
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
                base_mip_level: region.mip,
                level_count: 1,
                base_array_layer: region.layer,
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

    /// The no-hazard case: `to_vk_access_flags(empty)` is `NONE` and the descriptor-buffer path
    /// is skipped, leaving exactly the write-to-read dependency.
    pub fn barrier(&mut self, src: StageFlags, dst: StageFlags) {
        self.barrier_with_hazard(src, dst, HazardFlags::empty());
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
        // Hazard barriers retain the normal write-to-read dependency and add hazard-specific
        // access masks.
        let src_access =
            vk::AccessFlags2::MEMORY_WRITE | to_vk_access_flags(hazard_for_access, true);
        let mut dst_access = vk::AccessFlags2::MEMORY_READ
            | vk::AccessFlags2::MEMORY_WRITE
            | to_vk_access_flags(hazard_for_access, false);

        if use_descriptor_buffer_hazard {
            // Descriptor-buffer reads are not covered by MEMORY_READ.
            src_stage |= vk::PipelineStageFlags2::VERTEX_SHADER
                | vk::PipelineStageFlags2::FRAGMENT_SHADER
                | vk::PipelineStageFlags2::COMPUTE_SHADER;
            dst_stage |= vk::PipelineStageFlags2::VERTEX_SHADER
                | vk::PipelineStageFlags2::FRAGMENT_SHADER
                | vk::PipelineStageFlags2::COMPUTE_SHADER;
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

    pub fn set_viewport(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) {
        // Use a negative-height viewport so Vulkan matches the RHI's Y-up clip space.
        let viewport = vk::Viewport {
            x,
            y: y + height,
            width,
            height: -height,
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

    pub fn reset_queries(&mut self, pool: vk::QueryPool, first: u32, count: u32) {
        unsafe {
            self.device
                .cmd_reset_query_pool(self.command_buffer, pool, first, count);
        }
    }

    pub fn write_timestamp(&mut self, pool: vk::QueryPool, query: u32) {
        // Bracket GPU time with bottom-of-pipe timestamps.
        unsafe {
            self.device.cmd_write_timestamp2(
                self.command_buffer,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                pool,
                query,
            );
        }
    }

    /// Transition a swapchain image to PRESENT_SRC_KHR layout.
    /// Metal transitions on present, so this stays out of the public API.
    fn transition_to_present(&mut self, swapchain_image_index: u32) {
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

    pub fn set_meshlet_pipeline(&mut self, pso: &MeshletPso) {
        let vk_pso = backend_expect!(&pso.inner, crate::pipeline::MeshletPsoInner::Vulkan);
        self.bind_pipeline(
            vk::PipelineBindPoint::GRAPHICS,
            vk_pso.pipeline,
            vk_pso.pipeline_layout,
            vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::FRAGMENT,
        );
    }

    pub fn draw_meshlets(&mut self, x: u32, y: u32, z: u32) {
        let Some(loader) = self.mesh_shader.as_ref() else {
            return;
        };
        unsafe {
            loader.cmd_draw_mesh_tasks(self.command_buffer, x, y, z);
        }
    }

    /// `args` points to one `VkDrawMeshTasksIndirectCommandEXT` (x, y, z: u32 = 12 bytes).
    pub fn draw_meshlets_indirect(&mut self, args: GpuPtr<u8>) {
        let Some(loader) = self.mesh_shader.as_ref() else {
            return;
        };
        let stride = 12u32;
        let (buffer, offset) = self.resolve_buffer(args, stride as u64);
        unsafe {
            loader.cmd_draw_mesh_tasks_indirect(self.command_buffer, buffer, offset, 1, stride);
        }
    }

    pub fn build_blas(&mut self, accel: &crate::accel::AccelerationStructure, desc: &BlasDesc) {
        let Some((accel_loader, vk_as, scratch_address)) = self.resolve_accel(accel) else {
            return;
        };

        let geometries: Vec<vk::AccelerationStructureGeometryKHR> = desc
            .meshes
            .iter()
            .map(|m| match m.geometry_type {
                GeometryType::Triangles => {
                    let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                        .vertex_format(vk::Format::R32G32B32_SFLOAT)
                        .vertex_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: m.vertex_buffer.address,
                        })
                        .vertex_stride(m.vertex_stride)
                        .max_vertex(m.vertex_count.saturating_sub(1))
                        .index_type(if m.index_count > 0 {
                            vk::IndexType::UINT32
                        } else {
                            vk::IndexType::NONE_KHR
                        })
                        .index_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: m.index_buffer.address,
                        });
                    vk::AccelerationStructureGeometryKHR::default()
                        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                        .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
                        .flags(geometry_flags_to_vk(m.flags))
                }
                GeometryType::Aabbs => {
                    let aabbs = vk::AccelerationStructureGeometryAabbsDataKHR::default()
                        .data(vk::DeviceOrHostAddressConstKHR {
                            device_address: m.aabb_buffer.address,
                        })
                        .stride(std::mem::size_of::<vk::AabbPositionsKHR>() as u64);
                    vk::AccelerationStructureGeometryKHR::default()
                        .geometry_type(vk::GeometryTypeKHR::AABBS)
                        .geometry(vk::AccelerationStructureGeometryDataKHR { aabbs })
                        .flags(geometry_flags_to_vk(m.flags))
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
            .flags(build_accel_flags_to_vk(desc.flags))
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .dst_acceleration_structure(vk_as)
            .geometries(&geometries)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch_address,
            });

        let range_infos: Vec<vk::AccelerationStructureBuildRangeInfoKHR> = primitive_counts
            .iter()
            .map(|&pc| vk::AccelerationStructureBuildRangeInfoKHR {
                primitive_count: pc,
                primitive_offset: 0,
                first_vertex: 0,
                transform_offset: 0,
            })
            .collect();
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
        let Some((accel_loader, vk_as, scratch_address)) = self.resolve_accel(accel) else {
            return;
        };

        let instances_data = vk::AccelerationStructureGeometryInstancesDataKHR::default()
            .array_of_pointers(false)
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: desc.instance_buffer.address,
            });
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: instances_data,
            });
        let geometries = [geometry];

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(build_accel_flags_to_vk(desc.flags))
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .dst_acceleration_structure(vk_as)
            .geometries(&geometries)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch_address,
            });

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

    fn resolve_accel(
        &self,
        accel: &crate::accel::AccelerationStructure,
    ) -> Option<(
        vk_accel_structure::Device,
        vk::AccelerationStructureKHR,
        u64,
    )> {
        let loader = self.acceleration_structure.as_ref()?;
        match &accel.inner {
            #[cfg(feature = "vulkan")]
            crate::accel::AccelInner::Vulkan(a) => {
                Some((loader.clone(), a.acceleration_structure, a.scratch_address))
            }
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }
}

fn build_buffer_image_region(
    buffer_offset: u64,
    aspect: vk::ImageAspectFlags,
    region: ResolvedRegion,
) -> vk::BufferImageCopy {
    // Zero row/image length means "tightly packed to the copy extent", which is the layout
    // `ResolvedRegion::linear_strides` reports to the caller.
    vk::BufferImageCopy::default()
        .buffer_offset(buffer_offset)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(subresource_layers(aspect, region))
        .image_offset(to_vk_offset(region.origin))
        .image_extent(to_vk_extent(region.extent))
}

fn subresource_layers(
    aspect: vk::ImageAspectFlags,
    region: ResolvedRegion,
) -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers {
        aspect_mask: aspect,
        mip_level: region.mip,
        base_array_layer: region.layer,
        layer_count: 1,
    }
}

fn to_vk_offset(origin: [u32; 3]) -> vk::Offset3D {
    vk::Offset3D {
        x: origin[0] as i32,
        y: origin[1] as i32,
        z: origin[2] as i32,
    }
}

fn to_vk_extent(extent: [u32; 3]) -> vk::Extent3D {
    vk::Extent3D {
        width: extent[0],
        height: extent[1],
        depth: extent[2],
    }
}
