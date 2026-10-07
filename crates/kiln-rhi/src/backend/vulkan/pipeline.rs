use std::ffi::CString;
use std::rc::Rc;

use ash::vk;
use ash::vk::TaggedStructure as _;
use smallvec::SmallVec;

use super::device::{VulkanDevice, compare_op_to_vk, format_to_vk, spirv_words};
use super::shader::VulkanShaderModule;
use crate::error::{ErrorDetail, RhiError, RhiResult};
use crate::pipeline::{
    BlendAttachment, ComputePso, ComputePsoDesc, GraphicsPso, MeshletPso, RasterPsoDesc,
};
use crate::shader::{ShaderModule, ShaderModuleDesc};
use crate::types::{BlendFactor, BlendOp, ColorWriteMask, Cull, DepthFlags, Topology};

/// A graphics, compute or mesh pipeline. Command buffers hold an `Rc` to every pipeline they
/// bind, so dropping the application's handle never destroys one still in use.
pub struct VulkanPipeline {
    pub(crate) pipeline: vk::Pipeline,
    device: ash::Device,
}

impl Drop for VulkanPipeline {
    fn drop(&mut self) {
        // SAFETY: the last `Rc` is gone, so no recorded or in-flight command buffer binds it.
        unsafe { self.device.destroy_pipeline(self.pipeline, None) };
    }
}

fn stage_info(
    stage: vk::ShaderStageFlags,
    module: &VulkanShaderModule,
) -> vk::PipelineShaderStageCreateInfo<'_> {
    vk::PipelineShaderStageCreateInfo::default()
        .stage(stage)
        .module(module.module)
        .name(&module.entry_point)
}

/// Bindless goes through the descriptor heap, so pipelines opt in and carry no layout. The bit
/// only exists on `PipelineCreateFlags2`, hence a pNext rather than `.flags()`.
fn descriptor_heap_flags() -> vk::PipelineCreateFlags2CreateInfo<'static> {
    vk::PipelineCreateFlags2CreateInfo::default()
        .flags(vk::PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT)
}

fn cull_to_vk(cull: Cull) -> (vk::CullModeFlags, vk::FrontFace) {
    // Kiln's front face is always counter-clockwise.
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

impl VulkanDevice {
    pub fn create_shader_module(&self, desc: &ShaderModuleDesc) -> RhiResult<ShaderModule> {
        let code = spirv_words(desc.code)?;
        let shader_info = vk::ShaderModuleCreateInfo::default().code(&code);
        let module = unsafe { self.loaders.device.create_shader_module(&shader_info, None) }
            .map_err(|e| RhiError::ShaderCompilation(e.into()))?;

        let entry_point = CString::new(desc.entry_point).map_err(|e| {
            RhiError::ShaderCompilation(ErrorDetail::with_source(
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

    fn wrap_pipeline(&self, pipeline: vk::Pipeline, label: Option<&str>) -> Rc<VulkanPipeline> {
        if let Some(label) = label {
            self.set_object_name(pipeline, label);
        }
        Rc::new(VulkanPipeline {
            pipeline,
            device: self.loaders.device.clone(),
        })
    }

    /// Build a raster pipeline. The vertex path passes `vertex_input`, and with it the vertex-input
    /// and input-assembly state; the mesh stage replaces both.
    fn create_raster_pipeline(
        &self,
        desc: &RasterPsoDesc,
        stages: &[vk::PipelineShaderStageCreateInfo<'_>],
        vertex_input: bool,
        what: &str,
    ) -> RhiResult<Rc<VulkanPipeline>> {
        // Always empty: vertex data is read through pointers in the shader, never bound.
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly =
            vk::PipelineInputAssemblyStateCreateInfo::default().topology(match desc.topology {
                Topology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
                Topology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
            });

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let (cull_mode, front_face) = cull_to_vk(desc.cull);
        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(cull_mode)
            .front_face(front_face);

        // `VkSampleCountFlagBits` is the sample count itself as a single bit.
        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::from_raw(desc.sample_count.count()))
            .alpha_to_coverage_enable(desc.alpha_to_coverage);

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(desc.depth.mode.contains(DepthFlags::READ))
            .depth_write_enable(desc.depth.mode.contains(DepthFlags::WRITE))
            .depth_compare_op(compare_op_to_vk(desc.depth.compare));

        let blend_attachments: SmallVec<[vk::PipelineColorBlendAttachmentState; 4]> = desc
            .color_targets
            .iter()
            .enumerate()
            .map(|(i, target)| {
                blend_attachment_to_vk(
                    desc.blend.get(i).copied().unwrap_or_default(),
                    target.write_mask,
                )
            })
            .collect();
        let color_blending =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let color_formats: SmallVec<[vk::Format; 4]> = desc
            .color_targets
            .iter()
            .map(|t| format_to_vk(t.format))
            .collect();
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(
                desc.depth_format
                    .map_or(vk::Format::UNDEFINED, format_to_vk),
            )
            .stencil_attachment_format(vk::Format::UNDEFINED);

        let mut flags2 = descriptor_heap_flags();
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
        if vertex_input {
            pipeline_info = pipeline_info
                .vertex_input_state(&vertex_input_info)
                .input_assembly_state(&input_assembly);
        }

        let pipelines = unsafe {
            self.loaders.device.create_graphics_pipelines(
                self.pipeline_cache,
                &[pipeline_info],
                None,
            )
        }
        .map_err(|(_, e)| {
            RhiError::PipelineCreation(format!("Vulkan {what} pipeline creation: {e:?}").into())
        })?;
        Ok(self.wrap_pipeline(pipelines[0], desc.label))
    }

    pub fn create_graphics_pso(
        &self,
        desc: &RasterPsoDesc,
        vertex: &VulkanShaderModule,
        fragment: &VulkanShaderModule,
    ) -> RhiResult<GraphicsPso> {
        let stages = [
            stage_info(vk::ShaderStageFlags::VERTEX, vertex),
            stage_info(vk::ShaderStageFlags::FRAGMENT, fragment),
        ];
        Ok(GraphicsPso {
            inner: self.create_raster_pipeline(desc, &stages, true, "graphics")?,
            _owner: None,
        })
    }

    pub fn create_meshlet_pso(
        &self,
        desc: &RasterPsoDesc,
        mesh: &VulkanShaderModule,
        fragment: &VulkanShaderModule,
        // Vulkan reads the mesh threadgroup size from the SPIR-V.
        _threads_per_mesh_group: Option<[u32; 3]>,
    ) -> RhiResult<MeshletPso> {
        let stages = [
            stage_info(vk::ShaderStageFlags::MESH_EXT, mesh),
            stage_info(vk::ShaderStageFlags::FRAGMENT, fragment),
        ];
        Ok(MeshletPso {
            inner: self.create_raster_pipeline(desc, &stages, false, "meshlet")?,
            _owner: None,
        })
    }

    /// Vulkan reads `[numthreads]` from the SPIR-V; `_threads` is for Metal, which cannot.
    pub fn create_compute_pso(
        &self,
        desc: &ComputePsoDesc,
        shader: &VulkanShaderModule,
        _threads: Option<[u32; 3]>,
    ) -> RhiResult<ComputePso> {
        let mut flags2 = descriptor_heap_flags();
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info(vk::ShaderStageFlags::COMPUTE, shader))
            .layout(vk::PipelineLayout::null())
            .push(&mut flags2);
        let pipelines = unsafe {
            self.loaders.device.create_compute_pipelines(
                self.pipeline_cache,
                &[pipeline_info],
                None,
            )
        }
        .map_err(|(_, e)| {
            RhiError::PipelineCreation(format!("Vulkan compute pipeline creation: {e:?}").into())
        })?;
        Ok(ComputePso {
            inner: self.wrap_pipeline(pipelines[0], desc.label),
            _owner: None,
        })
    }
}
