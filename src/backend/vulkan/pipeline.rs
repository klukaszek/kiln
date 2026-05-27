use std::cell::RefCell;
use std::collections::HashMap;

use ash::vk;

use super::device::format_to_vk;
use crate::pipeline::{BlendAttachment, BlendState, ColorTarget};
use crate::types::{BlendFactor, BlendOp, ColorWriteMask, Cull, SampleCount, Topology};

/// Vulkan graphics pipeline state.
pub struct VulkanGraphicsPso {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pub(crate) root_constant_size: u32,
    pub(crate) device: ash::Device,
    pub(crate) desc: VulkanGraphicsPsoDesc,
    pub(crate) blend_pipelines: RefCell<HashMap<BlendState, vk::Pipeline>>,
}

/// Vulkan compute pipeline state.
pub struct VulkanComputePso {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pub(crate) root_constant_size: u32,
    #[allow(dead_code)]
    pub(crate) threads_per_threadgroup: [u32; 3],
}

pub struct VulkanGraphicsPsoDesc {
    pub(crate) vert_module: vk::ShaderModule,
    pub(crate) frag_module: vk::ShaderModule,
    pub(crate) vert_entry: std::ffi::CString,
    pub(crate) frag_entry: std::ffi::CString,
    pub(crate) topology: Topology,
    pub(crate) color_targets: Vec<ColorTarget>,
    pub(crate) depth_format: Option<vk::Format>,
    pub(crate) sample_count: SampleCount,
    pub(crate) alpha_to_coverage: bool,
    pub(crate) cull: Cull,
    pub(crate) stencil_format: vk::Format,
}

impl VulkanGraphicsPso {
    pub(crate) fn pipeline_for_blend(&self, blend: &BlendState) -> vk::Pipeline {
        if let Some(p) = self.blend_pipelines.borrow().get(blend) {
            return *p;
        }
        let pipeline = self.create_pipeline(blend);
        self.blend_pipelines
            .borrow_mut()
            .insert(blend.clone(), pipeline);
        pipeline
    }

    fn create_pipeline(&self, blend: &BlendState) -> vk::Pipeline {
        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(self.desc.vert_module)
                .name(&self.desc.vert_entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(self.desc.frag_module)
                .name(&self.desc.frag_entry),
        ];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(
            match self.desc.topology {
                Topology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
                Topology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
                Topology::TriangleFan => vk::PrimitiveTopology::TRIANGLE_FAN,
            },
        );

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        // Derive cull flags and front-face from the unified Cull value.
        // All variants use CCW as the implied front-face convention.
        let (cull_mode, front_face) = match self.desc.cull {
            Cull::None => (vk::CullModeFlags::NONE, vk::FrontFace::COUNTER_CLOCKWISE),
            Cull::Cw => (vk::CullModeFlags::BACK, vk::FrontFace::COUNTER_CLOCKWISE),
            Cull::Ccw => (vk::CullModeFlags::FRONT, vk::FrontFace::COUNTER_CLOCKWISE),
            Cull::All => (
                vk::CullModeFlags::FRONT_AND_BACK,
                vk::FrontFace::COUNTER_CLOCKWISE,
            ),
        };

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(cull_mode)
            .front_face(front_face);

        let samples = match self.desc.sample_count {
            SampleCount::S1 => vk::SampleCountFlags::TYPE_1,
            SampleCount::S2 => vk::SampleCountFlags::TYPE_2,
            SampleCount::S4 => vk::SampleCountFlags::TYPE_4,
            SampleCount::S8 => vk::SampleCountFlags::TYPE_8,
            SampleCount::S16 => vk::SampleCountFlags::TYPE_16,
        };
        let mut multisampling =
            vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(samples);
        if self.desc.alpha_to_coverage {
            multisampling = multisampling.alpha_to_coverage_enable(true);
        }

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS);

        let color_blend_attachments: Vec<vk::PipelineColorBlendAttachmentState> = self
            .desc
            .color_targets
            .iter()
            .enumerate()
            .map(|(i, target)| {
                let mut att = blend.attachments.get(i).cloned().unwrap_or_default();
                // AND the static PSO write mask (ColorTarget) with the dynamic blend mask.
                att.write_mask &= target.write_mask;
                blend_attachment_to_vk(att)
            })
            .collect();

        let color_blending =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);

        let dynamic_states = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::DEPTH_TEST_ENABLE,
            vk::DynamicState::DEPTH_WRITE_ENABLE,
            vk::DynamicState::DEPTH_COMPARE_OP,
            vk::DynamicState::STENCIL_TEST_ENABLE,
            vk::DynamicState::STENCIL_OP,
            vk::DynamicState::STENCIL_COMPARE_MASK,
            vk::DynamicState::STENCIL_WRITE_MASK,
            vk::DynamicState::STENCIL_REFERENCE,
            vk::DynamicState::DEPTH_BIAS_ENABLE,
            vk::DynamicState::DEPTH_BIAS,
        ];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let color_attachment_formats: Vec<vk::Format> = self
            .desc
            .color_targets
            .iter()
            .map(|t| format_to_vk(t.format))
            .collect();
        let depth_format = self.desc.depth_format.unwrap_or(vk::Format::UNDEFINED);

        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_attachment_formats)
            .depth_attachment_format(depth_format)
            .stencil_attachment_format(self.desc.stencil_format);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state_info)
            .layout(self.pipeline_layout)
            .push_next(&mut rendering_info);

        let pipelines = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .expect("Failed to create Vulkan pipeline variant")
        };

        pipelines[0]
    }
}

impl Drop for VulkanGraphicsPso {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            for (_, pipe) in self.blend_pipelines.borrow_mut().drain() {
                if pipe != self.pipeline {
                    self.device.destroy_pipeline(pipe, None);
                }
            }
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

fn blend_attachment_to_vk(att: BlendAttachment) -> vk::PipelineColorBlendAttachmentState {
    let mut state = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(color_write_mask_to_vk(att.write_mask))
        .blend_enable(att.blend_enable);

    if att.blend_enable {
        state = state
            .src_color_blend_factor(blend_factor_to_vk(att.src_color))
            .dst_color_blend_factor(blend_factor_to_vk(att.dst_color))
            .color_blend_op(blend_op_to_vk(att.color_op))
            .src_alpha_blend_factor(blend_factor_to_vk(att.src_alpha))
            .dst_alpha_blend_factor(blend_factor_to_vk(att.dst_alpha))
            .alpha_blend_op(blend_op_to_vk(att.alpha_op));
    }

    state
}

fn color_write_mask_to_vk(mask: ColorWriteMask) -> vk::ColorComponentFlags {
    let mut flags = vk::ColorComponentFlags::empty();
    if mask.contains(ColorWriteMask::R) {
        flags |= vk::ColorComponentFlags::R;
    }
    if mask.contains(ColorWriteMask::G) {
        flags |= vk::ColorComponentFlags::G;
    }
    if mask.contains(ColorWriteMask::B) {
        flags |= vk::ColorComponentFlags::B;
    }
    if mask.contains(ColorWriteMask::A) {
        flags |= vk::ColorComponentFlags::A;
    }
    flags
}

fn blend_factor_to_vk(factor: BlendFactor) -> vk::BlendFactor {
    match factor {
        BlendFactor::Zero => vk::BlendFactor::ZERO,
        BlendFactor::One => vk::BlendFactor::ONE,
        BlendFactor::SrcColor => vk::BlendFactor::SRC_COLOR,
        BlendFactor::OneMinusSrcColor => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        BlendFactor::DstColor => vk::BlendFactor::DST_COLOR,
        BlendFactor::OneMinusDstColor => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        BlendFactor::SrcAlpha => vk::BlendFactor::SRC_ALPHA,
        BlendFactor::OneMinusSrcAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        BlendFactor::DstAlpha => vk::BlendFactor::DST_ALPHA,
        BlendFactor::OneMinusDstAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
    }
}

fn blend_op_to_vk(op: BlendOp) -> vk::BlendOp {
    match op {
        BlendOp::Add => vk::BlendOp::ADD,
        BlendOp::Subtract => vk::BlendOp::SUBTRACT,
        BlendOp::ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
        BlendOp::Min => vk::BlendOp::MIN,
        BlendOp::Max => vk::BlendOp::MAX,
    }
}

// ---------------------------------------------------------------------------
// Vulkan mesh-shader pipeline — VK_EXT_mesh_shader
// ---------------------------------------------------------------------------

/// Vulkan meshlet (mesh shader) pipeline state.
///
/// Requires `VK_EXT_mesh_shader`.  The pipeline is built without a vertex or
/// geometry stage; the mesh stage replaces both.
pub struct VulkanMeshletPso {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pub(crate) root_constant_size: u32,
    pub(crate) device: ash::Device,
    /// Blend variants (same per-draw flyweight mechanism as graphics PSOs).
    pub(crate) desc: VulkanMeshletPsoDesc,
    pub(crate) blend_pipelines: RefCell<HashMap<BlendState, vk::Pipeline>>,
}

pub struct VulkanMeshletPsoDesc {
    pub(crate) mesh_module: vk::ShaderModule,
    pub(crate) frag_module: vk::ShaderModule,
    pub(crate) mesh_entry: std::ffi::CString,
    pub(crate) frag_entry: std::ffi::CString,
    pub(crate) color_targets: Vec<ColorTarget>,
    pub(crate) depth_format: Option<vk::Format>,
    pub(crate) stencil_format: vk::Format,
    pub(crate) sample_count: SampleCount,
    pub(crate) alpha_to_coverage: bool,
    pub(crate) cull: Cull,
}

impl VulkanMeshletPso {
    pub(crate) fn pipeline_for_blend(&self, blend: &BlendState) -> vk::Pipeline {
        if let Some(p) = self.blend_pipelines.borrow().get(blend) {
            return *p;
        }
        let pipeline = self.create_pipeline(blend);
        self.blend_pipelines
            .borrow_mut()
            .insert(blend.clone(), pipeline);
        pipeline
    }

    fn create_pipeline(&self, blend: &BlendState) -> vk::Pipeline {
        let mesh_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::MESH_EXT)
            .module(self.desc.mesh_module)
            .name(&self.desc.mesh_entry);
        let frag_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(self.desc.frag_module)
            .name(&self.desc.frag_entry);
        let stages = [mesh_stage, frag_stage];

        let (cull_mode, front_face) = match self.desc.cull {
            Cull::None => (vk::CullModeFlags::NONE, vk::FrontFace::COUNTER_CLOCKWISE),
            Cull::Cw => (vk::CullModeFlags::BACK, vk::FrontFace::COUNTER_CLOCKWISE),
            Cull::Ccw => (vk::CullModeFlags::FRONT, vk::FrontFace::COUNTER_CLOCKWISE),
            Cull::All => (
                vk::CullModeFlags::FRONT_AND_BACK,
                vk::FrontFace::COUNTER_CLOCKWISE,
            ),
        };
        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(cull_mode)
            .front_face(front_face);

        let samples = match self.desc.sample_count {
            SampleCount::S1 => vk::SampleCountFlags::TYPE_1,
            SampleCount::S2 => vk::SampleCountFlags::TYPE_2,
            SampleCount::S4 => vk::SampleCountFlags::TYPE_4,
            SampleCount::S8 => vk::SampleCountFlags::TYPE_8,
            SampleCount::S16 => vk::SampleCountFlags::TYPE_16,
        };
        let mut multisampling =
            vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(samples);
        if self.desc.alpha_to_coverage {
            multisampling = multisampling.alpha_to_coverage_enable(true);
        }

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS);

        let color_blend_attachments: Vec<vk::PipelineColorBlendAttachmentState> = self
            .desc
            .color_targets
            .iter()
            .enumerate()
            .map(|(i, target)| {
                let mut att = blend.attachments.get(i).cloned().unwrap_or_default();
                att.write_mask &= target.write_mask;
                blend_attachment_to_vk(att)
            })
            .collect();
        let color_blending =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);

        let dynamic_states = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::DEPTH_TEST_ENABLE,
            vk::DynamicState::DEPTH_WRITE_ENABLE,
            vk::DynamicState::DEPTH_COMPARE_OP,
            vk::DynamicState::STENCIL_TEST_ENABLE,
            vk::DynamicState::STENCIL_OP,
            vk::DynamicState::STENCIL_COMPARE_MASK,
            vk::DynamicState::STENCIL_WRITE_MASK,
            vk::DynamicState::STENCIL_REFERENCE,
            vk::DynamicState::DEPTH_BIAS_ENABLE,
            vk::DynamicState::DEPTH_BIAS,
        ];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let color_attachment_formats: Vec<vk::Format> = self
            .desc
            .color_targets
            .iter()
            .map(|t| format_to_vk(t.format))
            .collect();
        let depth_format = self.desc.depth_format.unwrap_or(vk::Format::UNDEFINED);
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_attachment_formats)
            .depth_attachment_format(depth_format)
            .stencil_attachment_format(self.desc.stencil_format);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state_info)
            .layout(self.pipeline_layout)
            .push_next(&mut rendering_info);

        let pipelines = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .expect("Failed to create Vulkan meshlet pipeline variant")
        };
        pipelines[0]
    }
}

impl Drop for VulkanMeshletPso {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            for (_, pipe) in self.blend_pipelines.borrow_mut().drain() {
                if pipe != self.pipeline {
                    self.device.destroy_pipeline(pipe, None);
                }
            }
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

// ---------------------------------------------------------------------------
// Vulkan ray tracing pipeline — VK_KHR_ray_tracing_pipeline
// ---------------------------------------------------------------------------

/// Vulkan ray tracing pipeline.
///
/// Requires `VK_KHR_ray_tracing_pipeline` and `VK_KHR_acceleration_structure`.
/// The shader binding table is owned by the caller — this struct holds only the
/// pipeline handle and layout.
pub struct VulkanRayTracingPso {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pub(crate) device: ash::Device,
    /// Opaque shader group handles (each `shader_group_handle_size` bytes).
    /// The caller copies these into a GPU buffer to construct the SBT.
    pub(crate) group_handles: Vec<u8>,
    /// Number of bytes per shader group handle (from `VkPhysicalDeviceRayTracingPipelinePropertiesKHR`).
    pub(crate) handle_size: u32,
    /// Required alignment for SBT record start addresses.
    pub(crate) handle_alignment: u32,
}

impl VulkanRayTracingPso {
    /// Return the opaque handle for group `index` as a byte slice.
    /// Copy this into a GPU-mapped SBT buffer, padded to `handle_alignment`.
    pub fn group_handle(&self, index: usize) -> &[u8] {
        let start = index * self.handle_size as usize;
        &self.group_handles[start..start + self.handle_size as usize]
    }

    /// Aligned handle size (stride for SBT records).
    pub fn aligned_handle_size(&self) -> u32 {
        (self.handle_size + self.handle_alignment - 1) & !(self.handle_alignment - 1)
    }
}

impl Drop for VulkanRayTracingPso {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}
