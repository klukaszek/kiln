use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4AlphaToCoverageState, MTL4BlendState, MTL4Compiler, MTL4IndirectCommandBufferSupportState,
    MTL4LibraryFunctionDescriptor, MTL4PipelineDescriptor,
    MTL4RenderPipelineColorAttachmentDescriptor, MTL4RenderPipelineDescriptor, MTLBlendFactor,
    MTLBlendOperation, MTLColorWriteMask, MTLCullMode, MTLLibrary, MTLPrimitiveType,
    MTLRenderPipelineState, MTLWinding,
};

use crate::error::{RhiError, RhiResult};
use crate::pipeline::{BlendAttachment, BlendState};
use crate::types::{BlendFactor, BlendOp, ColorWriteMask};

pub struct MetalGraphicsPso {
    pub(crate) pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    pub(crate) cull_mode: MTLCullMode,
    pub(crate) winding: MTLWinding,
    pub(crate) topology: MTLPrimitiveType,
}

pub struct MetalComputePso {
    pub(crate) pipeline: Retained<ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>,
    pub(crate) threads_per_threadgroup: [u32; 3],
    pub(crate) label: Option<String>,
}

impl MetalGraphicsPso {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compile_pipeline_state(
        compiler: &ProtocolObject<dyn MTL4Compiler>,
        vertex_library: &ProtocolObject<dyn MTLLibrary>,
        vertex_entry_point: &str,
        fragment_library: &ProtocolObject<dyn MTLLibrary>,
        fragment_entry_point: &str,
        color_formats: &[objc2_metal::MTLPixelFormat],
        color_write_masks: &[ColorWriteMask],
        sample_count: usize,
        alpha_to_coverage: bool,
        blend: &BlendState,
        label: Option<&str>,
    ) -> RhiResult<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
        let vertex_name = NSString::from_str(vertex_entry_point);
        let fragment_name = NSString::from_str(fragment_entry_point);

        let vertex_desc = MTL4LibraryFunctionDescriptor::new();
        vertex_desc.setName(Some(&vertex_name));
        vertex_desc.setLibrary(Some(vertex_library));

        let fragment_desc = MTL4LibraryFunctionDescriptor::new();
        fragment_desc.setName(Some(&fragment_name));
        fragment_desc.setLibrary(Some(fragment_library));

        let pso_desc = MTL4RenderPipelineDescriptor::new();
        pso_desc.setVertexFunctionDescriptor(Some(vertex_desc.as_ref()));
        pso_desc.setFragmentFunctionDescriptor(Some(fragment_desc.as_ref()));
        if let Some(label) = label {
            let base: &MTL4PipelineDescriptor = pso_desc.as_ref();
            base.setLabel(Some(&NSString::from_str(label)));
        }

        let color_attachments = pso_desc.colorAttachments();
        for (i, fmt) in color_formats.iter().enumerate() {
            let att = unsafe { color_attachments.objectAtIndexedSubscript(i) };
            att.setPixelFormat(*fmt);
            let blend_att = blend.attachments.get(i).cloned().unwrap_or_default();
            let write_mask = color_write_masks
                .get(i)
                .copied()
                .unwrap_or(ColorWriteMask::ALL);
            apply_blend_to_attachment(att.as_ref(), blend_att, write_mask);
        }

        // Metal 4 supplies depth/stencil formats when the render pass is created, not here.

        unsafe {
            pso_desc.setRasterSampleCount(sample_count);
        }
        pso_desc.setAlphaToCoverageState(if alpha_to_coverage {
            MTL4AlphaToCoverageState::Enabled
        } else {
            MTL4AlphaToCoverageState::Disabled
        });
        pso_desc.setSupportIndirectCommandBuffers(MTL4IndirectCommandBufferSupportState::Enabled);

        let base_desc: &MTL4PipelineDescriptor = pso_desc.as_ref();
        compiler
            .newRenderPipelineStateWithDescriptor_compilerTaskOptions_error(base_desc, None)
            .map_err(|e| {
                RhiError::PipelineCreation(format!("Metal 4 graphics PSO creation failed: {e}"))
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
    pub(crate) default_pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
}
