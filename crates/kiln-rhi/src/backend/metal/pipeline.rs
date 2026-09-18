use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4AlphaToCoverageState, MTL4BlendState, MTL4Compiler, MTL4IndirectCommandBufferSupportState,
    MTL4LibraryFunctionDescriptor, MTL4PipelineDescriptor,
    MTL4RenderPipelineColorAttachmentDescriptor, MTL4RenderPipelineDescriptor, MTLBlendFactor,
    MTLBlendOperation, MTLColorWriteMask, MTLCullMode, MTLDevice, MTLLibrary, MTLPrimitiveType,
    MTLRenderPipelineState, MTLWinding,
};

use super::device::MetalDevice;
use super::device::cull_to_mtl;
use super::shader::MetalShaderModule;
use super::texture::format_to_mtl;
use crate::error::{RhiError, RhiResult};
use crate::pipeline::{BlendAttachment, DepthState};
use crate::pipeline::{
    ComputePso, ComputePsoDesc, GraphicsPso, GraphicsPsoDesc, MeshletPso, MeshletPsoDesc,
};
use crate::shader::{ShaderModule, ShaderModuleDesc};
use crate::types::Topology;
use crate::types::{BlendFactor, BlendOp, ColorWriteMask, CompareOp, DepthFlags};
use objc2_metal::MTL4ComputePipelineDescriptor;
use smallvec::SmallVec;

pub(crate) fn make_depth_stencil_state(
    device: &ProtocolObject<dyn MTLDevice>,
    depth: DepthState,
) -> RhiResult<Retained<ProtocolObject<dyn objc2_metal::MTLDepthStencilState>>> {
    let desc = objc2_metal::MTLDepthStencilDescriptor::new();
    desc.setDepthCompareFunction(if depth.mode.contains(DepthFlags::READ) {
        compare_op_to_mtl(depth.compare)
    } else {
        objc2_metal::MTLCompareFunction::Always
    });
    desc.setDepthWriteEnabled(depth.mode.contains(DepthFlags::WRITE));
    device
        .newDepthStencilStateWithDescriptor(&desc)
        .ok_or_else(|| {
            RhiError::PipelineCreation("Metal depth/stencil state creation failed".into())
        })
}

pub(crate) fn compare_op_to_mtl(op: CompareOp) -> objc2_metal::MTLCompareFunction {
    use objc2_metal::MTLCompareFunction as F;
    match op {
        CompareOp::Never => F::Never,
        CompareOp::Less => F::Less,
        CompareOp::Equal => F::Equal,
        CompareOp::LessOrEqual => F::LessEqual,
        CompareOp::Greater => F::Greater,
        CompareOp::NotEqual => F::NotEqual,
        CompareOp::GreaterOrEqual => F::GreaterEqual,
        CompareOp::Always => F::Always,
    }
}

pub struct MetalGraphicsPso {
    pub(crate) pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    pub(crate) depth_stencil: Retained<ProtocolObject<dyn objc2_metal::MTLDepthStencilState>>,
    pub(crate) depth_bias: (f32, f32, f32),
    pub(crate) cull_mode: MTLCullMode,
    pub(crate) winding: MTLWinding,
    pub(crate) topology: MTLPrimitiveType,
}

pub struct MetalComputePso {
    pub(crate) pipeline: Retained<ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>,
    pub(crate) threads_per_threadgroup: [u32; 3],
    /// Owned: outlives the descriptor, and names every encoder this pipeline binds to.
    pub(crate) label: Option<String>,
}

/// One shader stage of a Metal render pipeline: the library plus the entry point in it.
pub(crate) struct MetalStage<'a> {
    pub(crate) library: &'a ProtocolObject<dyn MTLLibrary>,
    pub(crate) entry_point: &'a str,
}

impl MetalStage<'_> {
    fn function_descriptor(&self) -> Retained<MTL4LibraryFunctionDescriptor> {
        let desc = MTL4LibraryFunctionDescriptor::new();
        desc.setName(Some(&NSString::from_str(self.entry_point)));
        desc.setLibrary(Some(self.library));
        desc
    }
}

/// Everything `compile_pipeline_state` needs beyond the two shader stages.
pub(crate) struct MetalRasterState<'a> {
    pub(crate) color_formats: &'a [objc2_metal::MTLPixelFormat],
    pub(crate) color_write_masks: &'a [ColorWriteMask],
    pub(crate) sample_count: usize,
    pub(crate) alpha_to_coverage: bool,
    pub(crate) blend: &'a [BlendAttachment],
    pub(crate) label: Option<&'a str>,
}

impl MetalGraphicsPso {
    pub(crate) fn compile_pipeline_state(
        compiler: &ProtocolObject<dyn MTL4Compiler>,
        vertex: MetalStage<'_>,
        fragment: MetalStage<'_>,
        state: &MetalRasterState<'_>,
    ) -> RhiResult<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
        let vertex_desc = vertex.function_descriptor();
        let fragment_desc = fragment.function_descriptor();

        let pso_desc = MTL4RenderPipelineDescriptor::new();
        pso_desc.setVertexFunctionDescriptor(Some(vertex_desc.as_ref()));
        pso_desc.setFragmentFunctionDescriptor(Some(fragment_desc.as_ref()));
        if let Some(label) = state.label {
            let base: &MTL4PipelineDescriptor = pso_desc.as_ref();
            base.setLabel(Some(&NSString::from_str(label)));
        }

        let color_attachments = pso_desc.colorAttachments();
        for (i, fmt) in state.color_formats.iter().enumerate() {
            let att = unsafe { color_attachments.objectAtIndexedSubscript(i) };
            att.setPixelFormat(*fmt);
            let blend_att = state.blend.get(i).copied().unwrap_or_default();
            let write_mask = state
                .color_write_masks
                .get(i)
                .copied()
                .unwrap_or(ColorWriteMask::ALL);
            apply_blend_to_attachment(att.as_ref(), blend_att, write_mask);
        }

        // Metal 4 supplies depth/stencil formats when the render pass is created, not here.

        unsafe {
            pso_desc.setRasterSampleCount(state.sample_count);
        }
        pso_desc.setAlphaToCoverageState(if state.alpha_to_coverage {
            MTL4AlphaToCoverageState::Enabled
        } else {
            MTL4AlphaToCoverageState::Disabled
        });
        pso_desc.setSupportIndirectCommandBuffers(MTL4IndirectCommandBufferSupportState::Enabled);

        let base_desc: &MTL4PipelineDescriptor = pso_desc.as_ref();
        compiler
            .newRenderPipelineStateWithDescriptor_compilerTaskOptions_error(base_desc, None)
            .map_err(|e| {
                RhiError::PipelineCreation(
                    format!("Metal 4 graphics PSO creation failed: {e}").into(),
                )
            })
    }
}

pub(crate) fn apply_blend_to_attachment(
    att: &MTL4RenderPipelineColorAttachmentDescriptor,
    blend: BlendAttachment,
    write_mask: ColorWriteMask,
) {
    att.setBlendingState(if blend.blend_enable {
        MTL4BlendState::Enabled
    } else {
        MTL4BlendState::Disabled
    });
    att.setWriteMask(color_write_mask_to_mtl(write_mask));
    if blend.blend_enable {
        att.setSourceRGBBlendFactor(blend_factor_to_mtl(blend.src_color));
        att.setDestinationRGBBlendFactor(blend_factor_to_mtl(blend.dst_color));
        att.setRgbBlendOperation(blend_op_to_mtl(blend.color_op));
        att.setSourceAlphaBlendFactor(blend_factor_to_mtl(blend.src_alpha));
        att.setDestinationAlphaBlendFactor(blend_factor_to_mtl(blend.dst_alpha));
        att.setAlphaBlendOperation(blend_op_to_mtl(blend.alpha_op));
    }
}

fn color_write_mask_to_mtl(mask: ColorWriteMask) -> MTLColorWriteMask {
    let mut flags = MTLColorWriteMask::empty();
    if mask.contains(ColorWriteMask::R) {
        flags |= MTLColorWriteMask::Red;
    }
    if mask.contains(ColorWriteMask::G) {
        flags |= MTLColorWriteMask::Green;
    }
    if mask.contains(ColorWriteMask::B) {
        flags |= MTLColorWriteMask::Blue;
    }
    if mask.contains(ColorWriteMask::A) {
        flags |= MTLColorWriteMask::Alpha;
    }
    flags
}

fn blend_factor_to_mtl(factor: BlendFactor) -> MTLBlendFactor {
    match factor {
        BlendFactor::Zero => MTLBlendFactor::Zero,
        BlendFactor::One => MTLBlendFactor::One,
        BlendFactor::SrcColor => MTLBlendFactor::SourceColor,
        BlendFactor::OneMinusSrcColor => MTLBlendFactor::OneMinusSourceColor,
        BlendFactor::DstColor => MTLBlendFactor::DestinationColor,
        BlendFactor::OneMinusDstColor => MTLBlendFactor::OneMinusDestinationColor,
        BlendFactor::SrcAlpha => MTLBlendFactor::SourceAlpha,
        BlendFactor::OneMinusSrcAlpha => MTLBlendFactor::OneMinusSourceAlpha,
        BlendFactor::DstAlpha => MTLBlendFactor::DestinationAlpha,
        BlendFactor::OneMinusDstAlpha => MTLBlendFactor::OneMinusDestinationAlpha,
    }
}

fn blend_op_to_mtl(op: BlendOp) -> MTLBlendOperation {
    match op {
        BlendOp::Add => MTLBlendOperation::Add,
        BlendOp::Subtract => MTLBlendOperation::Subtract,
        BlendOp::ReverseSubtract => MTLBlendOperation::ReverseSubtract,
        BlendOp::Min => MTLBlendOperation::Min,
        BlendOp::Max => MTLBlendOperation::Max,
    }
}

/// Metal meshlet (mesh shader) pipeline state. Mesh shaders need an Apple GPU family that
/// supports them; `create_meshlet_pso` fails at PSO compilation otherwise.
pub struct MetalMeshletPso {
    pub(crate) cull_mode: MTLCullMode,
    pub(crate) winding: MTLWinding,
    pub(crate) depth_stencil: Retained<ProtocolObject<dyn objc2_metal::MTLDepthStencilState>>,
    pub(crate) depth_bias: (f32, f32, f32),
    pub(crate) default_pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
}

impl MetalDevice {
    pub fn create_shader_module(&self, desc: &ShaderModuleDesc) -> RhiResult<ShaderModule> {
        // For Metal, `code` should be a compiled .metallib binary.
        if desc.code.is_empty() {
            return Err(RhiError::ShaderCompilation(
                "Metal shader module needs a non-empty metallib".into(),
            ));
        }
        let ptr = std::ptr::NonNull::new(desc.code.as_ptr().cast::<std::ffi::c_void>().cast_mut())
            .ok_or_else(|| RhiError::ShaderCompilation("shader code pointer is null".into()))?;
        let dispatch_data = unsafe {
            dispatch2::DispatchData::new(ptr, desc.code.len(), None, std::ptr::null_mut())
        };

        let library = self
            .shared
            .device
            .newLibraryWithData_error(&dispatch_data)
            .map_err(|e| {
                RhiError::ShaderCompilation(format!("Metal library creation failed: {e}").into())
            })?;

        Ok(ShaderModule {
            inner: Box::new(MetalShaderModule {
                library,
                entry_point: desc.entry_point.to_string(),
            }),
            stage: desc.stage,
            threads_per_threadgroup: desc.threads_per_threadgroup,
            _owner: None,
        })
    }

    pub fn create_graphics_pso(
        &self,
        desc: &GraphicsPsoDesc,
        vert_module: &MetalShaderModule,
        frag_module: &MetalShaderModule,
    ) -> RhiResult<GraphicsPso> {
        // SmallVec: a pipeline with more than four colour targets is rare enough that the
        // inline capacity covers every real case without touching the heap.
        let color_formats: SmallVec<[_; 4]> = desc
            .color_targets
            .iter()
            .map(|target| format_to_mtl(target.format))
            .collect();
        let color_write_masks: SmallVec<[_; 4]> = desc
            .color_targets
            .iter()
            .map(|target| target.write_mask)
            .collect();
        let pipeline_state = MetalGraphicsPso::compile_pipeline_state(
            self.compiler.as_ref(),
            super::pipeline::MetalStage {
                library: vert_module.library.as_ref(),
                entry_point: &vert_module.entry_point,
            },
            super::pipeline::MetalStage {
                library: frag_module.library.as_ref(),
                entry_point: &frag_module.entry_point,
            },
            &super::pipeline::MetalRasterState {
                color_formats: &color_formats,
                color_write_masks: &color_write_masks,
                sample_count: desc.sample_count.count() as usize,
                alpha_to_coverage: desc.alpha_to_coverage,
                blend: desc.blend,
                label: desc.label,
            },
        )?;

        let (cull_mode, winding) = cull_to_mtl(desc.cull);

        let topology = match desc.topology {
            Topology::TriangleList => objc2_metal::MTLPrimitiveType::Triangle,
            Topology::TriangleStrip => objc2_metal::MTLPrimitiveType::TriangleStrip,
        };

        Ok(GraphicsPso {
            inner: Box::new(MetalGraphicsPso {
                pipeline: pipeline_state,
                depth_stencil: super::pipeline::make_depth_stencil_state(
                    self.shared.device.as_ref(),
                    desc.depth,
                )?,
                depth_bias: (
                    desc.depth.bias,
                    desc.depth.bias_slope,
                    desc.depth.bias_clamp,
                ),
                cull_mode,
                winding,
                topology,
            }),
            _owner: None,
        })
    }

    pub fn create_compute_pso(
        &self,
        desc: &ComputePsoDesc,
        compute_module: &MetalShaderModule,
        threads_per_threadgroup: Option<[u32; 3]>,
    ) -> RhiResult<ComputePso> {
        // Unlike Vulkan, Metal cannot recover `[numthreads]` from the compiled library: it has to
        // be told, both to size the dispatch and to set `requiredThreadsPerThreadgroup`.
        let threads_per_threadgroup = threads_per_threadgroup.ok_or_else(|| {
            RhiError::PipelineCreation(
                "Metal needs the compute shader's threadgroup size, and this module carries no \
                 reflection for it: set ComputePsoDesc::threads_per_threadgroup"
                    .into(),
            )
        })?;
        let fn_name = NSString::from_str(&compute_module.entry_point);
        let func_desc = MTL4LibraryFunctionDescriptor::new();
        func_desc.setName(Some(&fn_name));
        func_desc.setLibrary(Some(&compute_module.library));

        let pipeline_desc = MTL4ComputePipelineDescriptor::new();
        pipeline_desc.setComputeFunctionDescriptor(Some(&func_desc));
        if let Some(label) = desc.label {
            let base: &MTL4PipelineDescriptor = pipeline_desc.as_ref();
            base.setLabel(Some(&NSString::from_str(label)));
        }
        // `Device::create_compute_pso` already settled and range-checked this.
        let tg = objc2_metal::MTLSize {
            width: threads_per_threadgroup[0] as usize,
            height: threads_per_threadgroup[1] as usize,
            depth: threads_per_threadgroup[2] as usize,
        };
        pipeline_desc.setRequiredThreadsPerThreadgroup(tg);

        let pipeline_state = self
            .compiler
            .newComputePipelineStateWithDescriptor_compilerTaskOptions_error(&pipeline_desc, None)
            .map_err(|e| {
                RhiError::PipelineCreation(format!("Metal compute PSO creation failed: {e}").into())
            })?;

        // A threadgroup wider than the pipeline's register budget is undefined in Metal: the
        // dispatch is dropped and the frame comes out black, with nothing raised anywhere. Vulkan
        // rejects the same mistake at validation, so report it here rather than let it render.
        let max_threads = {
            use objc2_metal::MTLComputePipelineState;
            pipeline_state.maxTotalThreadsPerThreadgroup()
        };
        let requested = tg.width * tg.height * tg.depth;
        if requested > max_threads {
            return Err(RhiError::PipelineCreation(
                format!(
                    "Metal compute PSO {:?} requested {requested} threads per threadgroup, but the \
                 compiled shader's register use allows at most {max_threads}",
                    desc.label.unwrap_or("<unlabelled>"),
                )
                .into(),
            ));
        }

        Ok(ComputePso {
            inner: Box::new(MetalComputePso {
                pipeline: pipeline_state,
                threads_per_threadgroup,
                label: desc.label.map(str::to_owned),
            }),
            _owner: None,
        })
    }

    pub fn create_meshlet_pso(
        &self,
        desc: &MeshletPsoDesc,
        mesh_module: &MetalShaderModule,
        frag_module: &MetalShaderModule,
    ) -> RhiResult<MeshletPso> {
        use super::pipeline::MetalMeshletPso;
        use objc2_metal::MTL4MeshRenderPipelineDescriptor;

        let mesh_fn_name = NSString::from_str(&mesh_module.entry_point);
        let mesh_func_desc = MTL4LibraryFunctionDescriptor::new();
        mesh_func_desc.setName(Some(&mesh_fn_name));
        mesh_func_desc.setLibrary(Some(&mesh_module.library));

        let frag_fn_name = NSString::from_str(&frag_module.entry_point);
        let frag_func_desc = MTL4LibraryFunctionDescriptor::new();
        frag_func_desc.setName(Some(&frag_fn_name));
        frag_func_desc.setLibrary(Some(&frag_module.library));

        let pipeline_desc = MTL4MeshRenderPipelineDescriptor::new();
        pipeline_desc.setMeshFunctionDescriptor(Some(&mesh_func_desc));
        pipeline_desc.setFragmentFunctionDescriptor(Some(&frag_func_desc));
        if let Some(label) = desc.label {
            let base: &MTL4PipelineDescriptor = pipeline_desc.as_ref();
            base.setLabel(Some(&NSString::from_str(label)));
        }

        unsafe {
            pipeline_desc.setRasterSampleCount(desc.sample_count.count() as usize);
            if desc.alpha_to_coverage {
                pipeline_desc
                    .setAlphaToCoverageState(objc2_metal::MTL4AlphaToCoverageState::Enabled);
            }
        }

        for (i, target) in desc.color_targets.iter().enumerate() {
            let att = unsafe { pipeline_desc.colorAttachments().objectAtIndexedSubscript(i) };
            att.setPixelFormat(super::texture::format_to_mtl(target.format));
            let blend_att = desc.blend.get(i).copied().unwrap_or_default();
            super::pipeline::apply_blend_to_attachment(att.as_ref(), blend_att, target.write_mask);
        }

        let base_desc: &MTL4PipelineDescriptor = pipeline_desc.as_ref();
        let default_pipeline = self
            .compiler
            .newRenderPipelineStateWithDescriptor_compilerTaskOptions_error(base_desc, None)
            .map_err(|e| RhiError::PipelineCreation(format!("Mesh PSO: {e}").into()))?;

        let (cull_mode, winding) = cull_to_mtl(desc.cull);

        Ok(MeshletPso {
            inner: Box::new(MetalMeshletPso {
                cull_mode,
                winding,
                depth_stencil: super::pipeline::make_depth_stencil_state(
                    self.shared.device.as_ref(),
                    desc.depth,
                )?,
                depth_bias: (
                    desc.depth.bias,
                    desc.depth.bias_slope,
                    desc.depth.bias_clamp,
                ),
                default_pipeline,
            }),
            _owner: None,
        })
    }
}
