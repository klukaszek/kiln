use std::cell::RefCell;
use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4AlphaToCoverageState, MTL4BlendState, MTL4Compiler, MTL4IndirectCommandBufferSupportState,
    MTL4LibraryFunctionDescriptor, MTL4PipelineDescriptor, MTL4PipelineOptions,
    MTL4RenderPipelineColorAttachmentDescriptor, MTL4RenderPipelineDescriptor,
    MTL4ShaderReflection, MTLBlendFactor, MTLBlendOperation, MTLColorWriteMask, MTLCullMode,
    MTLLibrary, MTLPrimitiveType, MTLRenderPipelineState, MTLWinding,
};

use crate::pipeline::{BlendAttachment, BlendState};
use crate::types::{BlendFactor, BlendOp, ColorWriteMask};

pub struct MetalGraphicsPso {
    pub(crate) cull_mode: MTLCullMode,
    pub(crate) winding: MTLWinding,
    pub(crate) topology: MTLPrimitiveType,
    pub(crate) compiler: Retained<ProtocolObject<dyn MTL4Compiler>>,
    pub(crate) vertex_library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pub(crate) vertex_entry_point: String,
    pub(crate) fragment_library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pub(crate) fragment_entry_point: String,
    pub(crate) color_formats: Vec<objc2_metal::MTLPixelFormat>,
    /// Stored for render-pass construction; not baked into the MTL4 PSO at compile time.
    #[allow(dead_code)]
    pub(crate) depth_format: objc2_metal::MTLPixelFormat,
    /// Stored for render-pass construction; not baked into the MTL4 PSO at compile time.
    #[allow(dead_code)]
    pub(crate) stencil_format: objc2_metal::MTLPixelFormat,
    pub(crate) sample_count: usize,
    pub(crate) alpha_to_coverage: bool,
    pub(crate) root_constant_size: u32,
    pub(crate) graphics_argument_buffer_slots: Vec<usize>,
    pub(crate) blend_pipelines:
        RefCell<HashMap<BlendState, Retained<ProtocolObject<dyn MTLRenderPipelineState>>>>,
}

pub struct MetalComputePso {
    pub(crate) pipeline: Retained<ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>,
    pub(crate) threads_per_threadgroup: [u32; 3],
    pub(crate) root_constant_size: u32,
    pub(crate) compute_argument_buffer_slots: Vec<usize>,
}

impl MetalGraphicsPso {
    pub(crate) fn pipeline_for_blend(
        &self,
        blend: &BlendState,
    ) -> Retained<ProtocolObject<dyn MTLRenderPipelineState>> {
        if let Some(pso) = self.blend_pipelines.borrow().get(blend) {
            return pso.clone();
        }
        let pso = self.create_pipeline(blend);
        self.blend_pipelines
            .borrow_mut()
            .insert(blend.clone(), pso.clone());
        pso
    }

    fn create_pipeline(
        &self,
        blend: &BlendState,
    ) -> Retained<ProtocolObject<dyn MTLRenderPipelineState>> {
        Self::compile_pipeline_state(
            self.compiler.as_ref(),
            self.vertex_library.as_ref(),
            &self.vertex_entry_point,
            self.fragment_library.as_ref(),
            &self.fragment_entry_point,
            &self.color_formats,
            self.sample_count,
            self.alpha_to_coverage,
            blend,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compile_pipeline_state(
        compiler: &ProtocolObject<dyn MTL4Compiler>,
        vertex_library: &ProtocolObject<dyn MTLLibrary>,
        vertex_entry_point: &str,
        fragment_library: &ProtocolObject<dyn MTLLibrary>,
        fragment_entry_point: &str,
        color_formats: &[objc2_metal::MTLPixelFormat],
        sample_count: usize,
        alpha_to_coverage: bool,
        blend: &BlendState,
    ) -> Retained<ProtocolObject<dyn MTLRenderPipelineState>> {
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

        let color_attachments = pso_desc.colorAttachments();
        for (i, fmt) in color_formats.iter().enumerate() {
            let att = unsafe { color_attachments.objectAtIndexedSubscript(i) };
            att.setPixelFormat(*fmt);
            let blend_att = blend.attachments.get(i).cloned().unwrap_or_default();
            apply_blend_to_attachment(att.as_ref(), blend_att);
        }

        // Note: MTL4RenderPipelineDescriptor does not expose depth/stencil attachment format
        // setters — Metal 4 decouples PSO compilation from attachment formats. Formats are
        // stored on MetalGraphicsPso for render-pass construction but are not baked into the PSO.

        unsafe {
            pso_desc.setRasterSampleCount(sample_count);
        }
        pso_desc.setAlphaToCoverageState(if alpha_to_coverage {
            MTL4AlphaToCoverageState::Enabled
        } else {
            MTL4AlphaToCoverageState::Disabled
        });
        pso_desc.setSupportIndirectCommandBuffers(MTL4IndirectCommandBufferSupportState::Enabled);

        let options = MTL4PipelineOptions::new();
        options.setShaderReflection(
            MTL4ShaderReflection::BindingInfo | MTL4ShaderReflection::BufferTypeInfo,
        );
        pso_desc.setOptions(Some(&options));

        let base_desc: &MTL4PipelineDescriptor = pso_desc.as_ref();
        compiler
            .newRenderPipelineStateWithDescriptor_compilerTaskOptions_error(base_desc, None)
            .expect("Metal 4 graphics PSO creation failed")
    }
}

fn apply_blend_to_attachment(
    att: &MTL4RenderPipelineColorAttachmentDescriptor,
    blend: BlendAttachment,
) {
    att.setBlendingState(if blend.blend_enable {
        MTL4BlendState::Enabled
    } else {
        MTL4BlendState::Disabled
    });
    att.setWriteMask(color_write_mask_to_mtl(blend.write_mask));
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

// ---------------------------------------------------------------------------
// Metal mesh shader pipeline — MTLMeshRenderPipelineDescriptor
// ---------------------------------------------------------------------------

/// Metal meshlet (mesh shader) pipeline state.
///
/// Requires the Metal 4 mesh render pipeline path.
/// On unsupported hardware `create_meshlet_pso` returns `RhiError::Unsupported`.
pub struct MetalMeshletPso {
    pub(crate) cull_mode: MTLCullMode,
    pub(crate) winding: MTLWinding,
    #[allow(dead_code)]
    pub(crate) sample_count: usize,
    #[allow(dead_code)]
    pub(crate) alpha_to_coverage: bool,
    #[allow(dead_code)]
    pub(crate) color_formats: Vec<objc2_metal::MTLPixelFormat>,
    #[allow(dead_code)]
    pub(crate) depth_format: objc2_metal::MTLPixelFormat,
    #[allow(dead_code)]
    pub(crate) stencil_format: objc2_metal::MTLPixelFormat,
    pub(crate) root_constant_size: u32,
    /// Buffer-index slots the mesh+fragment shader declares for bindless heap pointers.
    /// Used to drive selective heap refresh (texture=1, sampler=2).
    pub(crate) argument_buffer_slots: Vec<usize>,
    /// Blend pipeline variants (same flyweight mechanism as graphics PSOs).
    #[allow(dead_code)]
    pub(crate) blend_pipelines:
        RefCell<HashMap<BlendState, Retained<ProtocolObject<dyn MTLRenderPipelineState>>>>,
    // Hold the compiled pipeline state (default blend variant).
    pub(crate) default_pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
}
