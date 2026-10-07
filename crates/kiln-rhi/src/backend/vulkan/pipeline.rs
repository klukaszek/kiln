use ash::vk;
use ash::vk::TaggedStructure as _;

use super::device::{VulkanDevice, spirv_words};
use super::device::{compare_op_to_vk, format_to_vk};
use super::shader::VulkanShaderModule;
use crate::error::{RhiError, RhiResult};
use crate::pipeline::{BlendAttachment, ColorTarget, DepthState};
use crate::pipeline::{
    ComputePso, ComputePsoDesc, GraphicsPso, GraphicsPsoDesc, MeshletPso, MeshletPsoDesc,
};
use crate::shader::{ShaderModule, ShaderModuleDesc};
use crate::types::{BlendFactor, BlendOp, ColorWriteMask, Cull, DepthFlags, SampleCount, Topology};
use smallvec::SmallVec;
use std::ffi::CString;

/// Depth state is baked into the pipeline on both backends; see `RasterPsoDesc`.
fn depth_stencil_create_info(
    depth: DepthState,
) -> vk::PipelineDepthStencilStateCreateInfo<'static> {
    vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(depth.mode.contains(DepthFlags::READ))
        .depth_write_enable(depth.mode.contains(DepthFlags::WRITE))
        .depth_compare_op(compare_op_to_vk(depth.compare))
}

/// Vulkan graphics pipeline state.
///
/// Holds the pipeline and nothing else: the descriptor it was built from is consumed by
/// `create_raster_pipeline` and never needed again, and keeping it would mean keeping
/// `vk::ShaderModule` handles that dangle once the caller drops the `ShaderModule`.
pub struct VulkanGraphicsPso {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) device: ash::Device,
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

/// Raster state shared by the vertex and mesh pipeline paths.
pub(crate) struct RasterState<'a> {
    pub(crate) color_targets: &'a [ColorTarget],
    pub(crate) depth_format: Option<vk::Format>,
    pub(crate) depth: DepthState,
    pub(crate) sample_count: SampleCount,
    pub(crate) alpha_to_coverage: bool,
    pub(crate) cull: Cull,
    pub(crate) blend: &'a [BlendAttachment],
}

/// Build a raster pipeline. `topology` is `Some` for the vertex path, which also needs vertex-input
/// and input-assembly state; the mesh path passes `None` and supplies neither.
pub(crate) fn create_raster_pipeline(
    device: &ash::Device,
    cache: vk::PipelineCache,
    stages: &[vk::PipelineShaderStageCreateInfo<'_>],
    topology: Option<Topology>,
    state: &RasterState<'_>,
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

    // `VkSampleCountFlagBits` is the sample count itself as a single bit.
    let samples = vk::SampleCountFlags::from_raw(state.sample_count.count());
    let mut multisampling =
        vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(samples);
    if state.alpha_to_coverage {
        multisampling = multisampling.alpha_to_coverage_enable(true);
    }

    let depth_stencil = depth_stencil_create_info(state.depth);

    let color_blend_attachments: SmallVec<[vk::PipelineColorBlendAttachmentState; 4]> = state
        .color_targets
        .iter()
        .enumerate()
        .map(|(i, target)| {
            let att = state.blend.get(i).copied().unwrap_or_default();
            blend_attachment_to_vk(att, target.write_mask)
        })
        .collect();
    let color_blending =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state_info =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let color_attachment_formats: SmallVec<[vk::Format; 4]> = state
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
                RhiError::PipelineCreation(format!("Vulkan {what} pipeline creation: {e:?}").into())
            })?
    };
    Ok(pipelines[0])
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
}

impl Drop for VulkanMeshletPso {
    /// Destroys immediately; the caller guarantees no in-flight submission still references it.
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
        }
    }
}

impl VulkanDevice {
    pub fn create_shader_module(&self, desc: &ShaderModuleDesc) -> RhiResult<ShaderModule> {
        let code = spirv_words(desc.code)?;

        let shader_info = vk::ShaderModuleCreateInfo::default().code(&code);

        let module = unsafe {
            self.loaders
                .device
                .create_shader_module(&shader_info, None)
                .map_err(|e| RhiError::ShaderCompilation(e.into()))?
        };

        let entry_point = CString::new(desc.entry_point).map_err(|e| {
            RhiError::ShaderCompilation(crate::error::ErrorDetail::with_source(
                format!(
                    "entry point {:?} contains an interior NUL",
                    desc.entry_point
                ),
                e,
            ))
        })?;

        Ok(ShaderModule {
            inner: Box::new(VulkanShaderModule::new(
                self.loaders.device.clone(),
                module,
                entry_point,
            )),
            stage: desc.stage,
            threads_per_threadgroup: desc.threads_per_threadgroup,
            _owner: None,
        })
    }

    pub fn create_graphics_pso(
        &self,
        desc: &GraphicsPsoDesc,
        vert_module: &VulkanShaderModule,
        frag_module: &VulkanShaderModule,
    ) -> RhiResult<GraphicsPso> {
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module.module)
                .name(&vert_module.entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module.module)
                .name(&frag_module.entry_point),
        ];
        let pipeline = super::pipeline::create_raster_pipeline(
            &self.loaders.device,
            self.pipeline_cache,
            &stages,
            Some(desc.topology),
            &super::pipeline::RasterState {
                color_targets: desc.color_targets,
                depth_format: desc.depth_format.map(format_to_vk),
                depth: desc.depth,
                sample_count: desc.sample_count,
                alpha_to_coverage: desc.alpha_to_coverage,
                cull: desc.cull,
                blend: desc.blend,
            },
            "graphics",
        )?;
        if let Some(label) = desc.label {
            self.set_object_name(pipeline, label);
        }

        Ok(GraphicsPso {
            inner: std::rc::Rc::new(VulkanGraphicsPso {
                pipeline,
                device: self.loaders.device.clone(),
            }),
            _owner: None,
        })
    }

    /// `_threads` is the frontend's resolved `[numthreads]`, which Metal needs at dispatch.
    /// Vulkan takes it from the SPIR-V, so the size is validated by the caller and dropped here.
    pub fn create_compute_pso(
        &self,
        desc: &ComputePsoDesc,
        shader: &VulkanShaderModule,
        _threads: Option<[u32; 3]>,
    ) -> RhiResult<ComputePso> {
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader.module)
            .name(&shader.entry_point);

        let mut flags2 = vk::PipelineCreateFlags2CreateInfo::default()
            .flags(vk::PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT);
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(vk::PipelineLayout::null())
            .push(&mut flags2);

        let pipelines = unsafe {
            self.loaders
                .device
                .create_compute_pipelines(self.pipeline_cache, &[pipeline_info], None)
                .map_err(|e| RhiError::PipelineCreation(format!("{e:?}").into()))?
        };

        if let Some(label) = desc.label {
            self.set_object_name(pipelines[0], label);
        }

        Ok(ComputePso {
            inner: std::rc::Rc::new(VulkanComputePso {
                pipeline: pipelines[0],
                device: self.loaders.device.clone(),
            }),
            _owner: None,
        })
    }

    pub fn create_meshlet_pso(
        &self,
        desc: &MeshletPsoDesc,
        mesh_module: &VulkanShaderModule,
        frag_module: &VulkanShaderModule,
        // Vulkan reads the mesh threadgroup size from the SPIR-V.
        _threads_per_mesh_group: Option<[u32; 3]>,
    ) -> RhiResult<MeshletPso> {
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::MESH_EXT)
                .module(mesh_module.module)
                .name(&mesh_module.entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module.module)
                .name(&frag_module.entry_point),
        ];
        // `None` topology: the mesh stage replaces vertex input and input assembly.
        let pipeline = super::pipeline::create_raster_pipeline(
            &self.loaders.device,
            self.pipeline_cache,
            &stages,
            None,
            &super::pipeline::RasterState {
                color_targets: desc.color_targets,
                depth_format: desc.depth_format.map(format_to_vk),
                depth: desc.depth,
                sample_count: desc.sample_count,
                alpha_to_coverage: desc.alpha_to_coverage,
                cull: desc.cull,
                blend: desc.blend,
            },
            "meshlet",
        )?;
        if let Some(label) = desc.label {
            self.set_object_name(pipeline, label);
        }

        Ok(MeshletPso {
            inner: std::rc::Rc::new(VulkanMeshletPso {
                pipeline,
                device: self.loaders.device.clone(),
            }),
            _owner: None,
        })
    }
}
