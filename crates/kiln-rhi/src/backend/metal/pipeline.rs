use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4AlphaToCoverageState, MTL4BlendState, MTL4Compiler, MTL4ComputePipelineDescriptor,
    MTL4IndirectCommandBufferSupportState, MTL4MeshRenderPipelineDescriptor,
    MTL4PipelineDescriptor, MTL4RenderPipelineColorAttachmentDescriptorArray,
    MTL4RenderPipelineDescriptor, MTLBlendFactor, MTLBlendOperation, MTLColorWriteMask,
    MTLCompareFunction, MTLComputePipelineState, MTLCullMode, MTLDepthStencilDescriptor,
    MTLDepthStencilState, MTLDevice, MTLPrimitiveType, MTLRenderPipelineState, MTLSize, MTLWinding,
};

use super::device::{MetalDevice, cull_to_mtl};
use super::shader::MetalShaderModule;
use super::texture::format_to_mtl;
use crate::error::{RhiError, RhiResult};
use crate::pipeline::{
    BlendAttachment, ColorTarget, ComputePso, ComputePsoDesc, DepthState, GraphicsPso, MeshletPso,
    RasterPsoDesc,
};
use crate::shader::{ShaderModule, ShaderModuleDesc};
use crate::types::{BlendFactor, BlendOp, ColorWriteMask, CompareOp, DepthFlags, Topology};

/// One SIMD group, for a mesh module whose threadgroup size is unknown.
const DEFAULT_THREADS_PER_MESH_GROUP: MTLSize = MTLSize {
    width: 32,
    height: 1,
    depth: 1,
};

/// The encoder state a graphics or mesh pipeline sets when bound.
pub(crate) struct MetalRasterState {
    pub(crate) pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    pub(crate) depth_stencil: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
    pub(crate) depth_bias: (f32, f32, f32),
    pub(crate) cull_mode: MTLCullMode,
    pub(crate) winding: MTLWinding,
}

pub struct MetalGraphicsPso {
    pub(crate) raster: MetalRasterState,
    pub(crate) topology: MTLPrimitiveType,
}

/// Needs an Apple GPU family with mesh shaders; pipeline compilation fails otherwise.
pub struct MetalMeshletPso {
    pub(crate) raster: MetalRasterState,
    /// The mesh shader's `[numthreads]`. Metal drops a draw dispatched at any other size.
    pub(crate) threads_per_mesh_group: MTLSize,
}

pub struct MetalComputePso {
    pub(crate) pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(crate) threads_per_threadgroup: [u32; 3],
    /// Names every encoder this pipeline opens.
    pub(crate) label: Option<String>,
}

pub(crate) fn compare_op_to_mtl(op: CompareOp) -> MTLCompareFunction {
    match op {
        CompareOp::Never => MTLCompareFunction::Never,
        CompareOp::Less => MTLCompareFunction::Less,
        CompareOp::Equal => MTLCompareFunction::Equal,
        CompareOp::LessOrEqual => MTLCompareFunction::LessEqual,
        CompareOp::Greater => MTLCompareFunction::Greater,
        CompareOp::NotEqual => MTLCompareFunction::NotEqual,
        CompareOp::GreaterOrEqual => MTLCompareFunction::GreaterEqual,
        CompareOp::Always => MTLCompareFunction::Always,
    }
}

fn depth_stencil_state(
    device: &ProtocolObject<dyn MTLDevice>,
    depth: DepthState,
) -> RhiResult<Retained<ProtocolObject<dyn MTLDepthStencilState>>> {
    let desc = MTLDepthStencilDescriptor::new();
    desc.setDepthCompareFunction(if depth.mode.contains(DepthFlags::READ) {
        compare_op_to_mtl(depth.compare)
    } else {
        MTLCompareFunction::Always
    });
    desc.setDepthWriteEnabled(depth.mode.contains(DepthFlags::WRITE));
    device
        .newDepthStencilStateWithDescriptor(&desc)
        .ok_or_else(|| {
            RhiError::PipelineCreation("Metal depth/stencil state creation failed".into())
        })
}

/// Depth formats are given to Metal 4 by the render pass, not the pipeline.
fn set_color_targets(
    attachments: &MTL4RenderPipelineColorAttachmentDescriptorArray,
    targets: &[ColorTarget],
    blend: &[BlendAttachment],
) {
    for (i, target) in targets.iter().enumerate() {
        let att = unsafe { attachments.objectAtIndexedSubscript(i) };
        att.setPixelFormat(format_to_mtl(target.format));
        att.setWriteMask(color_write_mask_to_mtl(target.write_mask));
        let blend = blend.get(i).copied().unwrap_or_default();
        if blend.blend_enable {
            att.setBlendingState(MTL4BlendState::Enabled);
            att.setSourceRGBBlendFactor(blend_factor_to_mtl(blend.src_color));
            att.setDestinationRGBBlendFactor(blend_factor_to_mtl(blend.dst_color));
            att.setRgbBlendOperation(blend_op_to_mtl(blend.color_op));
            att.setSourceAlphaBlendFactor(blend_factor_to_mtl(blend.src_alpha));
            att.setDestinationAlphaBlendFactor(blend_factor_to_mtl(blend.dst_alpha));
            att.setAlphaBlendOperation(blend_op_to_mtl(blend.alpha_op));
        } else {
            att.setBlendingState(MTL4BlendState::Disabled);
        }
    }
}

fn alpha_to_coverage(enabled: bool) -> MTL4AlphaToCoverageState {
    if enabled {
        MTL4AlphaToCoverageState::Enabled
    } else {
        MTL4AlphaToCoverageState::Disabled
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

fn set_label(desc: &MTL4PipelineDescriptor, label: Option<&str>) {
    if let Some(label) = label {
        desc.setLabel(Some(&NSString::from_str(label)));
    }
}

impl MetalDevice {
    pub fn create_shader_module(&self, desc: &ShaderModuleDesc) -> RhiResult<ShaderModule> {
        if desc.code.is_empty() {
            return Err(RhiError::ShaderCompilation(
                "Metal shader module needs a non-empty metallib".into(),
            ));
        }
        let ptr = std::ptr::NonNull::new(desc.code.as_ptr().cast::<std::ffi::c_void>().cast_mut())
            .ok_or_else(|| RhiError::ShaderCompilation("shader code pointer is null".into()))?;
        // SAFETY: `ptr` covers `desc.code`, which outlives this call; with no destructor, the
        // dispatch data copies it.
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

    /// Compile a render pipeline and the depth state that goes with it.
    fn raster_state(
        &self,
        desc: &RasterPsoDesc,
        pipeline_desc: &MTL4PipelineDescriptor,
    ) -> RhiResult<MetalRasterState> {
        let pipeline = self
            .compiler
            .newRenderPipelineStateWithDescriptor_compilerTaskOptions_error(pipeline_desc, None)
            .map_err(|e| {
                RhiError::PipelineCreation(format!("Metal render PSO creation failed: {e}").into())
            })?;
        let (cull_mode, winding) = cull_to_mtl(desc.cull);
        Ok(MetalRasterState {
            pipeline,
            depth_stencil: depth_stencil_state(&self.shared.device, desc.depth)?,
            depth_bias: (
                desc.depth.bias,
                desc.depth.bias_slope,
                desc.depth.bias_clamp,
            ),
            cull_mode,
            winding,
        })
    }

    pub fn create_graphics_pso(
        &self,
        desc: &RasterPsoDesc,
        vertex: &MetalShaderModule,
        fragment: &MetalShaderModule,
    ) -> RhiResult<GraphicsPso> {
        let pipeline_desc = MTL4RenderPipelineDescriptor::new();
        pipeline_desc.setVertexFunctionDescriptor(Some(&vertex.function_descriptor()));
        pipeline_desc.setFragmentFunctionDescriptor(Some(&fragment.function_descriptor()));
        set_color_targets(
            &pipeline_desc.colorAttachments(),
            desc.color_targets,
            desc.blend,
        );
        unsafe { pipeline_desc.setRasterSampleCount(desc.sample_count.count() as usize) };
        pipeline_desc.setAlphaToCoverageState(alpha_to_coverage(desc.alpha_to_coverage));
        pipeline_desc
            .setSupportIndirectCommandBuffers(MTL4IndirectCommandBufferSupportState::Enabled);
        set_label(&pipeline_desc, desc.label);

        Ok(GraphicsPso {
            inner: Box::new(MetalGraphicsPso {
                raster: self.raster_state(desc, &pipeline_desc)?,
                topology: match desc.topology {
                    Topology::TriangleList => MTLPrimitiveType::Triangle,
                    Topology::TriangleStrip => MTLPrimitiveType::TriangleStrip,
                },
            }),
            _owner: None,
        })
    }

    pub fn create_meshlet_pso(
        &self,
        desc: &RasterPsoDesc,
        mesh: &MetalShaderModule,
        fragment: &MetalShaderModule,
        threads_per_mesh_group: Option<[u32; 3]>,
    ) -> RhiResult<MeshletPso> {
        let pipeline_desc = MTL4MeshRenderPipelineDescriptor::new();
        pipeline_desc.setMeshFunctionDescriptor(Some(&mesh.function_descriptor()));
        pipeline_desc.setFragmentFunctionDescriptor(Some(&fragment.function_descriptor()));
        set_color_targets(
            &pipeline_desc.colorAttachments(),
            desc.color_targets,
            desc.blend,
        );
        unsafe { pipeline_desc.setRasterSampleCount(desc.sample_count.count() as usize) };
        pipeline_desc.setAlphaToCoverageState(alpha_to_coverage(desc.alpha_to_coverage));
        set_label(&pipeline_desc, desc.label);

        let threads_per_mesh_group = match threads_per_mesh_group {
            Some([width, height, depth]) => {
                let size = MTLSize {
                    width: width as usize,
                    height: height as usize,
                    depth: depth as usize,
                };
                pipeline_desc.setRequiredThreadsPerMeshThreadgroup(size);
                size
            }
            None => DEFAULT_THREADS_PER_MESH_GROUP,
        };

        Ok(MeshletPso {
            inner: Box::new(MetalMeshletPso {
                raster: self.raster_state(desc, &pipeline_desc)?,
                threads_per_mesh_group,
            }),
            _owner: None,
        })
    }

    pub fn create_compute_pso(
        &self,
        desc: &ComputePsoDesc,
        module: &MetalShaderModule,
        threads_per_threadgroup: Option<[u32; 3]>,
    ) -> RhiResult<ComputePso> {
        // Metal cannot read `[numthreads]` back out of a library, so it has to be supplied.
        let threads_per_threadgroup = threads_per_threadgroup.ok_or_else(|| {
            RhiError::PipelineCreation(
                "Metal needs the compute shader's threadgroup size, and this module carries no \
                 reflection for it: set ComputePsoDesc::threads_per_threadgroup"
                    .into(),
            )
        })?;
        let [width, height, depth] = threads_per_threadgroup;
        let threads = MTLSize {
            width: width as usize,
            height: height as usize,
            depth: depth as usize,
        };

        let pipeline_desc = MTL4ComputePipelineDescriptor::new();
        pipeline_desc.setComputeFunctionDescriptor(Some(&module.function_descriptor()));
        pipeline_desc.setRequiredThreadsPerThreadgroup(threads);
        set_label(&pipeline_desc, desc.label);

        let pipeline = self
            .compiler
            .newComputePipelineStateWithDescriptor_compilerTaskOptions_error(&pipeline_desc, None)
            .map_err(|e| {
                RhiError::PipelineCreation(format!("Metal compute PSO creation failed: {e}").into())
            })?;

        // Metal silently drops a dispatch wider than the pipeline's register budget allows.
        let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
        let requested = threads.width * threads.height * threads.depth;
        if requested > max_threads {
            return Err(RhiError::PipelineCreation(
                format!(
                    "Metal compute PSO {:?} requested {requested} threads per threadgroup, but \
                     the compiled shader's register use allows at most {max_threads}",
                    desc.label.unwrap_or("<unlabelled>"),
                )
                .into(),
            ));
        }

        Ok(ComputePso {
            inner: Box::new(MetalComputePso {
                pipeline,
                threads_per_threadgroup,
                label: desc.label.map(str::to_owned),
            }),
            _owner: None,
        })
    }
}
