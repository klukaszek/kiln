use ash::vk;
use ash::vk::TaggedStructure as _;

use super::device::format_to_vk;
use crate::error::{RhiError, RhiResult};
use crate::pipeline::{BlendAttachment, BlendState, ColorTarget, DepthState};
use crate::types::{
    BlendFactor, BlendOp, ColorWriteMask, CompareOp, Cull, DepthFlags, SampleCount, Topology,
};

/// Depth state is baked into the pipeline on both backends; see `RasterPsoDesc`.
fn depth_stencil_create_info(
    depth: DepthState,
) -> vk::PipelineDepthStencilStateCreateInfo<'static> {
    vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(depth.mode.contains(DepthFlags::READ))
        .depth_write_enable(depth.mode.contains(DepthFlags::WRITE))
        .depth_compare_op(compare_op_to_vk(depth.compare))
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

/// Vulkan graphics pipeline state.
pub struct VulkanGraphicsPso {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) device: ash::Device,
    pub(crate) pipeline_cache: vk::PipelineCache,
    pub(crate) desc: VulkanGraphicsPsoDesc,
}

/// Vulkan compute pipeline state.
pub struct VulkanComputePso {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) device: ash::Device,
}

impl Drop for VulkanComputePso {
    /// Destroys immediately; the caller guarantees no in-flight submission still references it.
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
        }
    }
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
    pub(crate) depth: DepthState,
}

/// Raster state shared by the vertex and mesh pipeline paths.
struct RasterState<'a> {
    color_targets: &'a [ColorTarget],
    depth_format: Option<vk::Format>,
    depth: DepthState,
    sample_count: SampleCount,
    alpha_to_coverage: bool,
    cull: Cull,
}

/// Build a raster pipeline. `topology` is `Some` for the vertex path, which also needs vertex-input
/// and input-assembly state; the mesh path passes `None` and supplies neither.
fn create_raster_pipeline(
    device: &ash::Device,
    cache: vk::PipelineCache,
    stages: &[vk::PipelineShaderStageCreateInfo<'_>],
    topology: Option<Topology>,
    state: &RasterState<'_>,
    blend: &BlendState,
    what: &str,
) -> RhiResult<vk::Pipeline> {
    // Always empty: vertex data is read through pointers in the shader, never bound. The
    // struct is still required for a pipeline that has a vertex stage.
    let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly =
        vk::PipelineInputAssemblyStateCreateInfo::default().topology(match topology {
            Some(Topology::TriangleStrip) => vk::PrimitiveTopology::TRIANGLE_STRIP,
            _ => vk::PrimitiveTopology::TRIANGLE_LIST,
        });

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let (cull_mode, front_face) = cull_to_vk(state.cull);
    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(cull_mode)
        .front_face(front_face);

    let samples = match state.sample_count {
        SampleCount::S1 => vk::SampleCountFlags::TYPE_1,
        SampleCount::S2 => vk::SampleCountFlags::TYPE_2,
        SampleCount::S4 => vk::SampleCountFlags::TYPE_4,
        SampleCount::S8 => vk::SampleCountFlags::TYPE_8,
        SampleCount::S16 => vk::SampleCountFlags::TYPE_16,
    };
    let mut multisampling =
        vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(samples);
    if state.alpha_to_coverage {
        multisampling = multisampling.alpha_to_coverage_enable(true);
    }

    let depth_stencil = depth_stencil_create_info(state.depth);

    let color_blend_attachments: Vec<vk::PipelineColorBlendAttachmentState> = state
        .color_targets
        .iter()
        .enumerate()
        .map(|(i, target)| {
            let att = blend.attachments.get(i).copied().unwrap_or_default();
            blend_attachment_to_vk(att, target.write_mask)
        })
        .collect();
    let color_blending =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state_info =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let color_attachment_formats: Vec<vk::Format> = state
        .color_targets
        .iter()
        .map(|t| format_to_vk(t.format))
        .collect();
    let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&color_attachment_formats)
        .depth_attachment_format(state.depth_format.unwrap_or(vk::Format::UNDEFINED))
        .stencil_attachment_format(vk::Format::UNDEFINED);

    // Bindless goes through the descriptor heap, so the pipeline opts in and carries no layout.
    // The bit only exists on `PipelineCreateFlags2`, hence the pNext rather than `.flags()`.
    let mut flags2 = vk::PipelineCreateFlags2CreateInfo::default()
        .flags(vk::PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT);
    let mut pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(stages)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blending)
        .dynamic_state(&dynamic_state_info)
        .layout(vk::PipelineLayout::null())
        .push(&mut rendering_info)
        .push(&mut flags2);
    if topology.is_some() {
        pipeline_info = pipeline_info
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly);
    }

    let pipelines = unsafe {
        device
            .create_graphics_pipelines(cache, &[pipeline_info], None)
            .map_err(|(_, e)| {
                RhiError::PipelineCreation(format!("Vulkan {what} pipeline creation: {e:?}"))
            })?
    };
    Ok(pipelines[0])
}

impl VulkanGraphicsPso {
    pub(crate) fn create_pipeline(&self, blend: &BlendState) -> RhiResult<vk::Pipeline> {
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(self.desc.vert_module)
                .name(&self.desc.vert_entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(self.desc.frag_module)
                .name(&self.desc.frag_entry),
        ];
        create_raster_pipeline(
            &self.device,
            self.pipeline_cache,
            &stages,
            Some(self.desc.topology),
            &RasterState {
                color_targets: &self.desc.color_targets,
                depth_format: self.desc.depth_format,
                depth: self.desc.depth,
                sample_count: self.desc.sample_count,
                alpha_to_coverage: self.desc.alpha_to_coverage,
                cull: self.desc.cull,
            },
            blend,
            "graphics",
        )
    }
}

impl Drop for VulkanGraphicsPso {
    /// Destroys immediately; the caller guarantees no in-flight submission still references it.
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
        }
    }
}

/// Translate the unified `Cull` value into Vulkan's `(cull_mode_flags, front_face)`.
/// All variants use CCW as the implied front-face convention.
fn cull_to_vk(cull: Cull) -> (vk::CullModeFlags, vk::FrontFace) {
    let front = vk::FrontFace::COUNTER_CLOCKWISE;
    match cull {
        Cull::None => (vk::CullModeFlags::NONE, front),
        Cull::Cw => (vk::CullModeFlags::BACK, front),
        Cull::Ccw => (vk::CullModeFlags::FRONT, front),
        Cull::All => (vk::CullModeFlags::FRONT_AND_BACK, front),
    }
}

fn blend_attachment_to_vk(
    att: BlendAttachment,
    write_mask: ColorWriteMask,
) -> vk::PipelineColorBlendAttachmentState {
    let mut state = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(color_write_mask_to_vk(write_mask))
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

/// Vulkan meshlet (mesh shader) pipeline state. Requires `VK_EXT_mesh_shader`; built without a
/// vertex or geometry stage, since the mesh stage replaces both.
pub struct VulkanMeshletPso {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) device: ash::Device,
    pub(crate) pipeline_cache: vk::PipelineCache,
    pub(crate) desc: VulkanMeshletPsoDesc,
}

pub struct VulkanMeshletPsoDesc {
    pub(crate) mesh_module: vk::ShaderModule,
    pub(crate) frag_module: vk::ShaderModule,
    pub(crate) mesh_entry: std::ffi::CString,
    pub(crate) frag_entry: std::ffi::CString,
    pub(crate) color_targets: Vec<ColorTarget>,
    pub(crate) depth_format: Option<vk::Format>,
    pub(crate) depth: DepthState,
    pub(crate) sample_count: SampleCount,
    pub(crate) alpha_to_coverage: bool,
    pub(crate) cull: Cull,
}

impl VulkanMeshletPso {
    pub(crate) fn create_pipeline(&self, blend: &BlendState) -> RhiResult<vk::Pipeline> {
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::MESH_EXT)
                .module(self.desc.mesh_module)
                .name(&self.desc.mesh_entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(self.desc.frag_module)
                .name(&self.desc.frag_entry),
        ];
        create_raster_pipeline(
            &self.device,
            self.pipeline_cache,
            &stages,
            None,
            &RasterState {
                color_targets: &self.desc.color_targets,
                depth_format: self.desc.depth_format,
                depth: self.desc.depth,
                sample_count: self.desc.sample_count,
                alpha_to_coverage: self.desc.alpha_to_coverage,
                cull: self.desc.cull,
            },
            blend,
            "meshlet",
        )
    }
}

impl Drop for VulkanMeshletPso {
    /// Destroys immediately; the caller guarantees no in-flight submission still references it.
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
        }
    }
}
