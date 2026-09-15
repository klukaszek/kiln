use super::barrier::{to_vk_access_flags, to_vk_stage_flags};
use super::device::{
    IMAGE_LAYOUT, SharedTextures, build_accel_flags_to_vk, geometry_flags_to_vk,
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
    ext::{debug_utils, descriptor_heap, mesh_shader as vk_mesh_shader},
    khr::{acceleration_structure as vk_accel_structure, device_address_commands},
    vk,
};
use smallvec::SmallVec;
use std::rc::Rc;

/// A pipeline kept alive for as long as a command buffer references it.
///
/// The handles are never read back; holding the `Rc` is the entire point, so that dropping the
/// application's `GraphicsPso`/`ComputePso`/`MeshletPso` mid-recording cannot destroy a
/// `VkPipeline` the command buffer still refers to.
#[derive(Clone)]
#[allow(dead_code)]
pub(crate) enum RetainedPipeline {
    Graphics(Rc<super::pipeline::VulkanGraphicsPso>),
    Compute(Rc<super::pipeline::VulkanComputePso>),
    Meshlet(Rc<super::pipeline::VulkanMeshletPso>),
}

/// Vulkan command buffer wrapper.
pub struct VulkanCommandBuffer {
    pub(crate) command_buffer: vk::CommandBuffer,
    pub(crate) device: ash::Device,
    pub(crate) swapchain_image_views: Rc<[vk::ImageView]>,
    pub(crate) swapchain_images: Rc<[vk::Image]>,
    pub(crate) depth_image_view: vk::ImageView,
    pub(crate) descriptor_heap_loader: descriptor_heap::Device,
    pub(crate) address_commands: device_address_commands::Device,
    pub(crate) debug_labels: Option<debug_utils::Device>,
    /// Set when the open pass pushed a label region for `end_render_pass` to pop.
    pub(crate) in_labelled_pass: bool,
    pub(crate) pending_split_barrier: Option<(StageFlags, HazardFlags)>,
    pub(crate) textures: SharedTextures,
    pub(crate) mesh_shader: vk_mesh_shader::Device,
    pub(crate) acceleration_structure: vk_accel_structure::Device,
    pub(crate) rendered_swapchain_images: SmallVec<[u32; 4]>,
    /// Pipelines bound into this buffer. A pipeline dropped by the application while still
    /// referenced here must not be destroyed until the submission retires, which matches Metal,
    /// where the command buffer holds its own reference.
    pub(crate) retained_pipelines: SmallVec<[RetainedPipeline; 4]>,
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

    /// An address range for an address-taking command. Nothing resolves back to a `VkBuffer`:
    /// the address is the resource.
    fn range(addr: GpuPtr<u8>, size: u64) -> vk::DeviceAddressRangeKHR {
        vk::DeviceAddressRangeKHR::default()
            .address(addr.address)
            .size(size)
    }

    /// Same, for the indirect-argument commands, which also carry a stride.
    fn strided_range(
        addr: GpuPtr<u8>,
        size: u64,
        stride: u64,
    ) -> vk::StridedDeviceAddressRangeKHR {
        vk::StridedDeviceAddressRangeKHR::default()
            .address(addr.address)
            .size(size)
            .stride(stride)
    }

    fn resolve_texture_info(&self, id: TextureId) -> (vk::Image, vk::ImageView) {
        let textures = self.textures.borrow();
        let tex = textures
            .get(id.0 as usize)
            .and_then(|t| t.as_ref())
            .expect("Invalid texture ID");
        (tex.image, tex.image_view)
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
                        vk::PipelineStageFlags2::NONE,
                        vk::AccessFlags2::NONE,
                        vk::AccessFlags2::NONE,
                    )
                } else {
                    (
                        IMAGE_LAYOUT,
                        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                        vk::AccessFlags2::COLOR_ATTACHMENT_READ,
                    )
                };
                let barrier = vk::ImageMemoryBarrier2::default()
                    .old_layout(old_layout)
                    .new_layout(IMAGE_LAYOUT)
                    .src_stage_mask(src_stage)
                    .src_access_mask(src_access)
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE | extra_dst_access)
                    .image(image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                unsafe {
                    self.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default()
                            .image_memory_barriers(std::slice::from_ref(&barrier)),
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
                let image_view = match ca.target.kind() {
                    RenderTargetKind::SwapchainImage(idx) => {
                        self.swapchain_image_views[idx as usize]
                    }
                    RenderTargetKind::Texture(id) => self.resolve_texture_info(id).1,
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
                    .image_layout(IMAGE_LAYOUT)
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
            let image_view = match da.target.kind() {
                RenderTargetKind::SwapchainImage(_) => self.depth_image_view,
                RenderTargetKind::Texture(id) => self.resolve_texture_info(id).1,
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
                .image_layout(IMAGE_LAYOUT)
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
        self.retained_pipelines
            .push(RetainedPipeline::Graphics(vk_pso.clone()));
        self.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, vk_pso.pipeline);
    }

    pub fn set_compute_pipeline(&mut self, pso: &ComputePso) {
        let vk_pso = backend_expect!(&pso.inner, ComputePsoInner::Vulkan);
        self.retained_pipelines
            .push(RetainedPipeline::Compute(vk_pso.clone()));
        self.bind_pipeline(vk::PipelineBindPoint::COMPUTE, vk_pso.pipeline);
    }

    fn bind_pipeline(&mut self, bind_point: vk::PipelineBindPoint, pipeline: vk::Pipeline) {
        unsafe {
            self.device
                .cmd_bind_pipeline(self.command_buffer, bind_point, pipeline);
        }
    }

    pub fn set_root_data(&mut self, root: GpuPtr<u8>) {
        let bytes = root.address.to_ne_bytes();
        let info = vk::PushDataInfoEXT::default()
            .offset(0)
            .data(vk::HostAddressRangeConstEXT::default().address(&bytes));
        unsafe {
            self.descriptor_heap_loader
                .cmd_push_data(self.command_buffer, &info);
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
        let info = vk::BindIndexBuffer3InfoKHR::default()
            .address_range(Self::range(indices, index_count as u64 * 4))
            .index_type(vk::IndexType::UINT32);
        unsafe {
            self.address_commands
                .cmd_bind_index_buffer3(self.command_buffer, &info);
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
        let info = vk::DispatchIndirect2InfoKHR::default()
            .address_range(Self::range(args, size_of::<DispatchIndirectArgs>() as u64));
        unsafe {
            self.address_commands
                .cmd_dispatch_indirect2(self.command_buffer, &info);
        }
    }

    pub fn draw_indirect(&mut self, args: GpuPtr<u8>) {
        let stride = size_of::<DrawIndirectArgs>() as u64;
        let info = vk::DrawIndirect2InfoKHR::default()
            .address_range(Self::strided_range(args, stride, stride))
            .draw_count(1);
        unsafe {
            self.address_commands
                .cmd_draw_indirect2(self.command_buffer, &info);
        }
    }

    pub fn draw_indexed_indirect(
        &mut self,
        indices: GpuPtr<u8>,
        _max_index_count: u32,
        args: GpuPtr<u8>,
    ) {
        let stride = size_of::<DrawIndexedIndirectArgs>() as u64;
        let index_info = vk::BindIndexBuffer3InfoKHR::default()
            .address_range(Self::range(indices, _max_index_count.max(1) as u64 * 4))
            .index_type(vk::IndexType::UINT32);
        let draw_info = vk::DrawIndirect2InfoKHR::default()
            .address_range(Self::strided_range(args, stride, stride))
            .draw_count(1);
        unsafe {
            self.address_commands
                .cmd_bind_index_buffer3(self.command_buffer, &index_info);
            self.address_commands
                .cmd_draw_indexed_indirect2(self.command_buffer, &draw_info);
        }
    }

    pub fn memcpy(&mut self, dst: GpuPtr<u8>, src: GpuPtr<u8>, size: u64) {
        if size == 0 {
            return;
        }
        let region = vk::DeviceMemoryCopyKHR::default()
            .src_range(Self::range(src, size))
            .dst_range(Self::range(dst, size));
        let info = vk::CopyDeviceMemoryInfoKHR::default().regions(std::slice::from_ref(&region));
        unsafe {
            self.address_commands
                .cmd_copy_memory(self.command_buffer, &info);
        }
    }

    pub fn copy_buffer_to_texture(
        &mut self,
        src: GpuPtr<u8>,
        texture: &Texture,
        region: ResolvedRegion,
    ) {
        let (image, aspect, src_range) =
            self.prepare_texture_copy(src, texture, region, "copy_buffer_to_texture");
        let copy = build_memory_image_region(
            src_range,
            aspect,
            region,
            IMAGE_LAYOUT,
        );
        let info = vk::CopyDeviceMemoryImageInfoKHR::default()
            .image(image)
            .regions(std::slice::from_ref(&copy));
        unsafe {
            self.address_commands
                .cmd_copy_memory_to_image(self.command_buffer, &info);
        }
    }

    pub fn copy_texture_to_buffer(
        &mut self,
        dst: GpuPtr<u8>,
        texture: &Texture,
        region: ResolvedRegion,
    ) {
        let (image, aspect, dst_range) =
            self.prepare_texture_copy(dst, texture, region, "copy_texture_to_buffer");
        let copy = build_memory_image_region(
            dst_range,
            aspect,
            region,
            IMAGE_LAYOUT,
        );
        let info = vk::CopyDeviceMemoryImageInfoKHR::default()
            .image(image)
            .regions(std::slice::from_ref(&copy));
        unsafe {
            self.address_commands
                .cmd_copy_image_to_memory(self.command_buffer, &info);
        }
    }

    pub fn copy_texture_to_texture(
        &mut self,
        src: &Texture,
        src_region: ResolvedRegion,
        dst: &Texture,
        dst_region: ResolvedRegion,
    ) {
        let (src_image, src_aspect) = (
            self.resolve_texture_info(src.id()).0,
            texture_aspect(src.desc().format),
        );
        let (dst_image, dst_aspect) = (
            self.resolve_texture_info(dst.id()).0,
            texture_aspect(dst.desc().format),
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
                IMAGE_LAYOUT,
                dst_image,
                IMAGE_LAYOUT,
                std::slice::from_ref(&copy),
            );
        }
    }

    /// Resolve the texture and the linear memory backing the copy, returning
    /// `(image, aspect, current layout, address range)`.
    fn prepare_texture_copy(
        &self,
        buffer_gpu: GpuPtr<u8>,
        texture: &Texture,
        region: ResolvedRegion,
        op: &'static str,
    ) -> (
        vk::Image,
        vk::ImageAspectFlags,
        vk::DeviceAddressRangeKHR,
    ) {
        let image = self.resolve_texture_info(texture.id()).0;
        let bpp = bytes_per_pixel(texture.desc().format)
            .unwrap_or_else(|| panic!("Unsupported texture format for {op}"));
        let (_, bytes_per_image) = region.linear_strides(bpp);
        let size = (bytes_per_image as u64) * (region.extent[2] as u64);
        (
            image,
            texture_aspect(texture.desc().format),
            Self::range(buffer_gpu, size),
        )
    }

    /// The no-hazard case: `to_vk_access_flags(empty)` is `NONE` and the descriptor-buffer path
    /// is skipped, leaving exactly the write-to-read dependency.
    pub fn barrier(&mut self, src: StageFlags, dst: StageFlags) {
        self.barrier_with_hazard(src, dst, HazardFlags::empty());
    }

    pub fn barrier_with_hazard(&mut self, src: StageFlags, dst: StageFlags, hazard: HazardFlags) {
        let use_descriptor_heap_hazard = hazard.contains(HazardFlags::DESCRIPTORS);
        let hazard_for_access = if use_descriptor_heap_hazard {
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

        if use_descriptor_heap_hazard {
            // Descriptor heap reads are not covered by MEMORY_READ.
            src_stage |= vk::PipelineStageFlags2::VERTEX_SHADER
                | vk::PipelineStageFlags2::FRAGMENT_SHADER
                | vk::PipelineStageFlags2::COMPUTE_SHADER;
            dst_stage |= vk::PipelineStageFlags2::VERTEX_SHADER
                | vk::PipelineStageFlags2::FRAGMENT_SHADER
                | vk::PipelineStageFlags2::COMPUTE_SHADER;
            dst_access |= vk::AccessFlags2::RESOURCE_HEAP_READ_EXT
                | vk::AccessFlags2::SAMPLER_HEAP_READ_EXT;
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
        let barrier = vk::ImageMemoryBarrier2::default()
            .old_layout(IMAGE_LAYOUT)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        unsafe {
            self.device.cmd_pipeline_barrier2(
                self.command_buffer,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(std::slice::from_ref(&barrier)),
            );
        }
    }

    pub fn set_meshlet_pipeline(&mut self, pso: &MeshletPso) {
        let vk_pso = backend_expect!(&pso.inner, crate::pipeline::MeshletPsoInner::Vulkan);
        self.retained_pipelines
            .push(RetainedPipeline::Meshlet(vk_pso.clone()));
        self.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, vk_pso.pipeline);
    }

    pub fn draw_meshlets(&mut self, x: u32, y: u32, z: u32) {
        unsafe {
            self.mesh_shader
                .cmd_draw_mesh_tasks(self.command_buffer, x, y, z);
        }
    }

    /// `args` points to one `VkDrawMeshTasksIndirectCommandEXT` (x, y, z: u32 = 12 bytes).
    pub fn draw_meshlets_indirect(&mut self, args: GpuPtr<u8>) {
        let stride = 12u64;
        let info = vk::DrawIndirect2InfoKHR::default()
            .address_range(Self::strided_range(args, stride, stride))
            .draw_count(1);
        unsafe {
            self.address_commands
                .cmd_draw_mesh_tasks_indirect2(self.command_buffer, &info);
        }
    }

    pub fn build_blas(&mut self, accel: &crate::accel::AccelerationStructure, desc: &BlasDesc) {
        let (accel_loader, vk_as, scratch_address) = self.resolve_accel(accel);

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
        let range_infos_ref: Option<&[vk::AccelerationStructureBuildRangeInfoKHR]> =
            Some(&range_infos);
        let build_range_infos: &[Option<&[vk::AccelerationStructureBuildRangeInfoKHR]>] =
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
        let (accel_loader, vk_as, scratch_address) = self.resolve_accel(accel);

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
        let build_range_infos: &[Option<&[vk::AccelerationStructureBuildRangeInfoKHR]>] =
            &[Some(&range_info)];

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
    ) -> (
        vk_accel_structure::Device,
        vk::AccelerationStructureKHR,
        u64,
    ) {
        let a = backend_expect!(&accel.inner, crate::accel::AccelInner::Vulkan);
        (
            self.acceleration_structure.clone(),
            a.acceleration_structure,
            a.scratch_address,
        )
    }
}

fn build_memory_image_region(
    address_range: vk::DeviceAddressRangeKHR,
    aspect: vk::ImageAspectFlags,
    region: ResolvedRegion,
    image_layout: vk::ImageLayout,
) -> vk::DeviceMemoryImageCopyKHR<'static> {
    // Zero row/image length means "tightly packed to the copy extent", which is the layout
    // `ResolvedRegion::linear_strides` reports to the caller.
    vk::DeviceMemoryImageCopyKHR::default()
        .address_range(address_range)
        .address_row_length(0)
        .address_image_height(0)
        .image_subresource(subresource_layers(aspect, region))
        .image_layout(image_layout)
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
