use std::rc::Rc;

use ash::vk;
use smallvec::SmallVec;

use super::accel::{blas_geometries, build_geometry_info, tlas_geometry};
use super::barrier::{to_vk_access_flags, to_vk_stage_flags};
use super::device::{IMAGE_LAYOUT, SharedTextures, VulkanDevice, VulkanLoaders};
use super::pipeline::VulkanPipeline;
use super::query::VulkanQueryPool;
use super::swapchain::VulkanSwapchain;
use super::texture::texture_aspect;
use crate::accel::AccelerationStructure;
use crate::barrier::{HazardFlags, StageFlags};
use crate::command::{
    CommandBuffer, DispatchIndirectArgs, DrawIndexedIndirectArgs, DrawIndirectArgs, LoadOp,
    RenderPassDesc, RenderTargetKind, StoreOp,
};
use crate::error::{RhiError, RhiResult};
use crate::pipeline::{ComputePso, GraphicsPso, MeshletPso};
use crate::texture::{ResolvedRegion, Texture, bytes_per_pixel};
use crate::types::{BlasDesc, GpuPtr, MAX_FRAMES_IN_FLIGHT, TextureId, TlasDesc};

/// Mip 0, layer 0 of a colour image: the whole of a swapchain image.
pub(crate) const COLOR_SUBRESOURCE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

/// Shader stages that read the descriptor heaps.
const HEAP_READING_STAGES: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::from_raw(
    vk::PipelineStageFlags2::VERTEX_SHADER.as_raw()
        | vk::PipelineStageFlags2::FRAGMENT_SHADER.as_raw()
        | vk::PipelineStageFlags2::COMPUTE_SHADER.as_raw(),
);

fn load_op_to_vk(op: LoadOp) -> vk::AttachmentLoadOp {
    match op {
        LoadOp::Load => vk::AttachmentLoadOp::LOAD,
        LoadOp::Clear => vk::AttachmentLoadOp::CLEAR,
        LoadOp::DontCare => vk::AttachmentLoadOp::DONT_CARE,
    }
}

fn store_op_to_vk(op: StoreOp) -> vk::AttachmentStoreOp {
    match op {
        StoreOp::Store => vk::AttachmentStoreOp::STORE,
        StoreOp::DontCare => vk::AttachmentStoreOp::DONT_CARE,
    }
}

/// Vulkan command buffer wrapper.
pub struct VulkanCommandBuffer {
    pub(crate) command_buffer: vk::CommandBuffer,
    /// Shared with the device rather than copied: see [`VulkanLoaders`].
    pub(crate) loaders: Rc<VulkanLoaders>,
    pub(crate) swapchain_image_views: Rc<[vk::ImageView]>,
    pub(crate) swapchain_images: Rc<[vk::Image]>,
    /// Set when the open pass pushed a label region for `end_render_pass` to pop.
    pub(crate) in_labelled_pass: bool,
    pub(crate) textures: SharedTextures,
    pub(crate) rendered_swapchain_images: SmallVec<[u32; 4]>,
    /// The pipeline bound by the last `set_*_pipeline`, so re-binding the same one does not
    /// push another `Rc` onto `retained_pipelines`.
    pub(crate) last_bound_pipeline: vk::Pipeline,
    /// Every pipeline bound into this buffer, kept alive until the submission retires even if the
    /// application drops it. Metal command buffers hold their own reference the same way.
    pub(crate) retained_pipelines: SmallVec<[Rc<VulkanPipeline>; 4]>,
    pub(crate) ended: bool,
}

impl VulkanCommandBuffer {
    pub(crate) fn finish(&mut self) -> RhiResult<()> {
        if self.ended {
            return Ok(());
        }
        // Here rather than in `submit_frame`, so it holds whether or not the caller ends first.
        for index in std::mem::take(&mut self.rendered_swapchain_images) {
            self.transition_to_present(index);
        }
        unsafe {
            self.loaders
                .device
                .end_command_buffer(self.command_buffer)
                .map_err(|error| RhiError::CommandBuffer(error.into()))?;
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
    fn strided_range(addr: GpuPtr<u8>, size: u64, stride: u64) -> vk::StridedDeviceAddressRangeKHR {
        vk::StridedDeviceAddressRangeKHR::default()
            .address(addr.address)
            .size(size)
            .stride(stride)
    }

    fn resolve_texture_info(&self, id: TextureId) -> (vk::Image, vk::ImageView) {
        self.textures
            .with(id.0, |tex| (tex.image, tex.image_view))
            .expect("Invalid texture ID")
    }

    pub fn begin_render_pass(&mut self, desc: &RenderPassDesc<'_>) {
        let cmd = self.command_buffer;

        // Opened before the attachment transitions so the capture attributes them to this pass.
        self.in_labelled_pass = false;
        if let (Some(loader), Some(label)) = (self.loaders.debug_labels.as_ref(), desc.label)
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
                    .subresource_range(COLOR_SUBRESOURCE);
                unsafe {
                    self.loaders.device.cmd_pipeline_barrier2(
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
                vk::RenderingAttachmentInfo::default()
                    .image_view(image_view)
                    .image_layout(IMAGE_LAYOUT)
                    .load_op(load_op_to_vk(ca.load_op))
                    .store_op(store_op_to_vk(ca.store_op))
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: ca.clear_color,
                        },
                    })
            })
            .collect();

        let depth_attachment = desc.depth_attachment.as_ref().map(|da| {
            let image_view = match da.target.kind() {
                RenderTargetKind::Texture(id) => self.resolve_texture_info(id).1,
                RenderTargetKind::SwapchainImage(_) => {
                    panic!("a swapchain image cannot be a depth attachment")
                }
            };
            vk::RenderingAttachmentInfo::default()
                .image_view(image_view)
                .image_layout(IMAGE_LAYOUT)
                .load_op(load_op_to_vk(da.load_op))
                .store_op(store_op_to_vk(da.store_op))
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
            self.loaders
                .device
                .cmd_begin_rendering(cmd, &rendering_info);

            self.loaders.device.cmd_set_depth_bias_enable(cmd, false);
        }
    }

    pub fn end_render_pass(&mut self) {
        unsafe {
            self.loaders.device.cmd_end_rendering(self.command_buffer);
        }
        if self.in_labelled_pass
            && let Some(loader) = self.loaders.debug_labels.as_ref()
        {
            unsafe { loader.cmd_end_debug_utils_label(self.command_buffer) };
            self.in_labelled_pass = false;
        }
    }

    pub fn set_graphics_pipeline(&mut self, pso: &GraphicsPso) {
        self.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, &pso.inner);
    }

    pub fn set_compute_pipeline(&mut self, pso: &ComputePso) {
        self.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &pso.inner);
    }

    pub fn set_meshlet_pipeline(&mut self, pso: &MeshletPso) {
        self.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, &pso.inner);
    }

    /// Bind `pso`, retaining it for this command buffer's submission unless it is already the
    /// one bound or already retained.
    fn bind_pipeline(&mut self, bind_point: vk::PipelineBindPoint, pso: &Rc<VulkanPipeline>) {
        if self.last_bound_pipeline != pso.pipeline {
            self.last_bound_pipeline = pso.pipeline;
            if !self
                .retained_pipelines
                .iter()
                .any(|retained| Rc::ptr_eq(retained, pso))
            {
                self.retained_pipelines.push(pso.clone());
            }
        }
        unsafe {
            self.loaders
                .device
                .cmd_bind_pipeline(self.command_buffer, bind_point, pso.pipeline);
        }
    }

    pub fn set_root_data(&mut self, root: GpuPtr<u8>) {
        let bytes = root.address.to_ne_bytes();
        let info = vk::PushDataInfoEXT::default()
            .offset(0)
            .data(vk::HostAddressRangeConstEXT::default().address(&bytes));
        unsafe {
            self.loaders
                .descriptor_heap
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
            self.loaders.device.cmd_draw(
                self.command_buffer,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
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
        let info = vk::BindIndexBuffer3InfoKHR::default()
            .address_range(Self::range(indices, index_count as u64 * 4))
            .index_type(vk::IndexType::UINT32);
        unsafe {
            self.loaders
                .address_commands
                .cmd_bind_index_buffer3(self.command_buffer, &info);
            // `first_index` is always 0: the bound range starts at the caller's pointer.
            self.loaders.device.cmd_draw_indexed(
                self.command_buffer,
                index_count,
                instance_count,
                0,
                vertex_offset,
                first_instance,
            );
        }
    }

    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        unsafe {
            self.loaders
                .device
                .cmd_dispatch(self.command_buffer, x, y, z);
        }
    }

    pub fn dispatch_indirect(&mut self, args: GpuPtr<u8>) {
        let info = vk::DispatchIndirect2InfoKHR::default()
            .address_range(Self::range(args, size_of::<DispatchIndirectArgs>() as u64));
        unsafe {
            self.loaders
                .address_commands
                .cmd_dispatch_indirect2(self.command_buffer, &info);
        }
    }

    pub fn draw_indirect(&mut self, args: GpuPtr<u8>) {
        let stride = size_of::<DrawIndirectArgs>() as u64;
        let info = vk::DrawIndirect2InfoKHR::default()
            .address_range(Self::strided_range(args, stride, stride))
            .draw_count(1);
        unsafe {
            self.loaders
                .address_commands
                .cmd_draw_indirect2(self.command_buffer, &info);
        }
    }

    pub fn draw_indexed_indirect(
        &mut self,
        indices: GpuPtr<u8>,
        max_index_count: u32,
        args: GpuPtr<u8>,
    ) {
        let stride = size_of::<DrawIndexedIndirectArgs>() as u64;
        let index_info = vk::BindIndexBuffer3InfoKHR::default()
            .address_range(Self::range(indices, max_index_count.max(1) as u64 * 4))
            .index_type(vk::IndexType::UINT32);
        let draw_info = vk::DrawIndirect2InfoKHR::default()
            .address_range(Self::strided_range(args, stride, stride))
            .draw_count(1);
        unsafe {
            self.loaders
                .address_commands
                .cmd_bind_index_buffer3(self.command_buffer, &index_info);
            self.loaders
                .address_commands
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
            self.loaders
                .address_commands
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
        let copy = build_memory_image_region(src_range, aspect, region, IMAGE_LAYOUT);
        let info = vk::CopyDeviceMemoryImageInfoKHR::default()
            .image(image)
            .regions(std::slice::from_ref(&copy));
        unsafe {
            self.loaders
                .address_commands
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
        let copy = build_memory_image_region(dst_range, aspect, region, IMAGE_LAYOUT);
        let info = vk::CopyDeviceMemoryImageInfoKHR::default()
            .image(image)
            .regions(std::slice::from_ref(&copy));
        unsafe {
            self.loaders
                .address_commands
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
            self.loaders.device.cmd_copy_image(
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
    /// `(image, aspect, address range)`.
    fn prepare_texture_copy(
        &self,
        buffer_gpu: GpuPtr<u8>,
        texture: &Texture,
        region: ResolvedRegion,
        op: &'static str,
    ) -> (vk::Image, vk::ImageAspectFlags, vk::DeviceAddressRangeKHR) {
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

    /// With an empty `hazard`, `to_vk_access_flags` is `NONE` and the descriptor-heap path is
    /// skipped, leaving exactly the write-to-read dependency.
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
            src_stage |= HEAP_READING_STAGES;
            dst_stage |= HEAP_READING_STAGES;
            dst_access |=
                vk::AccessFlags2::RESOURCE_HEAP_READ_EXT | vk::AccessFlags2::SAMPLER_HEAP_READ_EXT;
        }

        let memory_barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access);

        let dep_info =
            vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&memory_barrier));

        unsafe {
            self.loaders
                .device
                .cmd_pipeline_barrier2(self.command_buffer, &dep_info);
        }
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
            self.loaders
                .device
                .cmd_set_viewport(self.command_buffer, 0, &[viewport]);
        }
    }

    pub fn set_scissor(&mut self, x: i32, y: i32, width: u32, height: u32) {
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x, y },
            extent: vk::Extent2D { width, height },
        };
        unsafe {
            self.loaders
                .device
                .cmd_set_scissor(self.command_buffer, 0, &[scissor]);
        }
    }

    pub fn reset_queries(&mut self, pool: &VulkanQueryPool, count: u32) {
        unsafe {
            self.loaders
                .device
                .cmd_reset_query_pool(self.command_buffer, pool.pool, 0, count);
        }
    }

    pub fn write_timestamp(&mut self, pool: &VulkanQueryPool, query: u32) {
        // Bracket GPU time with bottom-of-pipe timestamps.
        unsafe {
            self.loaders.device.cmd_write_timestamp2(
                self.command_buffer,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                pool.pool,
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
            .subresource_range(COLOR_SUBRESOURCE);
        unsafe {
            self.loaders.device.cmd_pipeline_barrier2(
                self.command_buffer,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(std::slice::from_ref(&barrier)),
            );
        }
    }

    pub fn draw_meshlets(&mut self, x: u32, y: u32, z: u32) {
        unsafe {
            self.loaders
                .mesh_shader
                .cmd_draw_mesh_tasks(self.command_buffer, x, y, z);
        }
    }

    /// `args` points to one `VkDrawMeshTasksIndirectCommandEXT`.
    pub fn draw_meshlets_indirect(&mut self, args: GpuPtr<u8>) {
        let stride = size_of::<vk::DrawMeshTasksIndirectCommandEXT>() as u64;
        let info = vk::DrawIndirect2InfoKHR::default()
            .address_range(Self::strided_range(args, stride, stride))
            .draw_count(1);
        unsafe {
            self.loaders
                .address_commands
                .cmd_draw_mesh_tasks_indirect2(self.command_buffer, &info);
        }
    }

    pub fn build_blas(&mut self, accel: &AccelerationStructure, desc: &BlasDesc<'_>) {
        let (geometries, primitive_counts) = blas_geometries(desc);
        let ranges: SmallVec<[vk::AccelerationStructureBuildRangeInfoKHR; 4]> = primitive_counts
            .iter()
            .map(
                |&primitive_count| vk::AccelerationStructureBuildRangeInfoKHR {
                    primitive_count,
                    ..Default::default()
                },
            )
            .collect();
        self.build_accel(
            accel,
            build_geometry_info(
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                desc.flags,
                &geometries,
            ),
            &ranges,
        );
    }

    pub fn build_tlas(&mut self, accel: &AccelerationStructure, desc: &TlasDesc) {
        let geometries = [tlas_geometry(desc)];
        let ranges = [vk::AccelerationStructureBuildRangeInfoKHR {
            primitive_count: desc.instance_count,
            ..Default::default()
        }];
        self.build_accel(
            accel,
            build_geometry_info(
                vk::AccelerationStructureTypeKHR::TOP_LEVEL,
                desc.flags,
                &geometries,
            ),
            &ranges,
        );
    }

    fn build_accel(
        &mut self,
        accel: &AccelerationStructure,
        info: vk::AccelerationStructureBuildGeometryInfoKHR<'_>,
        ranges: &[vk::AccelerationStructureBuildRangeInfoKHR],
    ) {
        let accel = &accel.inner;
        let info = info
            .dst_acceleration_structure(accel.acceleration_structure)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: accel.scratch_address,
            });
        unsafe {
            self.loaders
                .acceleration_structure
                .cmd_build_acceleration_structures(
                    self.command_buffer,
                    std::slice::from_ref(&info),
                    &[Some(ranges)],
                );
        }
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

impl VulkanDevice {
    pub(crate) fn recycle_command_buffer(&self, command_buffer: vk::CommandBuffer) {
        self.queue.recycle_command_buffer(command_buffer)
    }

    pub fn create_command_buffer(&self) -> RhiResult<CommandBuffer> {
        self.begin_command_buffer(None)
    }

    /// Begin recording, optionally wired to a swapchain's images.
    ///
    /// `swapchain` is `Some` only for the frame's own command buffer, the one allowed to name
    /// [`RenderTarget::swapchain_image`](crate::RenderTarget::swapchain_image).
    fn begin_command_buffer(
        &self,
        swapchain: Option<&VulkanSwapchain>,
    ) -> RhiResult<CommandBuffer> {
        // Texture creation records initial-layout transitions into one reusable setup command
        // buffer. Flush the batch once before user work starts instead of queue-idling once per
        // texture.
        self.flush_setup_barriers()?;
        let cmd = self.queue.acquire_command_buffer()?;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        // SAFETY: `cmd` came from this device's pool and is not recording.
        unsafe {
            self.loaders
                .device
                .begin_command_buffer(cmd, &begin_info)
                .map_err(|e| RhiError::CommandBuffer(e.into()))?;
        }

        // Both heaps are bound once here and stay bound: they are the only ones the device owns.
        self.bind_descriptor_heaps(cmd);

        Ok(CommandBuffer::new(Box::new(VulkanCommandBuffer {
            command_buffer: cmd,
            loaders: self.loaders.clone(),
            swapchain_image_views: swapchain
                .map_or_else(|| Rc::from([]), |sc| sc.image_views.clone()),
            swapchain_images: swapchain.map_or_else(|| Rc::from([]), |sc| sc.images.clone()),
            in_labelled_pass: false,
            textures: self.textures.clone(),
            rendered_swapchain_images: SmallVec::new(),
            retained_pipelines: SmallVec::new(),
            last_bound_pipeline: vk::Pipeline::null(),
            ended: false,
        })))
    }

    /// Bind the resource and sampler heaps for the lifetime of `cmd`.
    fn bind_descriptor_heaps(&self, cmd: vk::CommandBuffer) {
        let heaps = &self.descriptor_heaps;
        unsafe {
            self.loaders
                .descriptor_heap
                .cmd_bind_resource_heap(cmd, &heaps.resource.bind_info());
            self.loaders
                .descriptor_heap
                .cmd_bind_sampler_heap(cmd, &heaps.sampler.bind_info());
        }
    }

    /// Create a command buffer pre-configured with swapchain image views for rendering.
    pub fn create_command_buffer_for_swapchain(
        &self,
        sc: &VulkanSwapchain,
        frame_index: usize,
    ) -> RhiResult<CommandBuffer> {
        if frame_index >= MAX_FRAMES_IN_FLIGHT {
            return Err(RhiError::CommandBuffer("invalid Vulkan frame index".into()));
        }
        self.begin_command_buffer(Some(sc))
    }
}
