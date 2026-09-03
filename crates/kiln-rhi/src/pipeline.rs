//! Pipeline state objects: graphics, compute, and mesh-shader PSOs.

use crate::types::*;

/// Per-color-attachment entry in a graphics PSO.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ColorTarget {
    pub format: Format,
    /// Static write mask baked into the PSO; set empty to dead-code-eliminate unused outputs.
    pub write_mask: ColorWriteMask,
}

impl ColorTarget {
    pub fn new(format: Format) -> Self {
        Self {
            format,
            write_mask: ColorWriteMask::ALL,
        }
    }
}

/// Description for creating a rasterization pipeline state object.
///
/// Minimal PSO — topology, formats, MSAA, cull, write masks and blend baked. DepthStencil stays
/// dynamic (`set_depth_stencil_state`); blend cannot, since both backends bake it in. Shaders are
/// arguments to `create_graphics_pso` / `create_meshlet_pso`. The vertex and mesh paths take the
/// same raster state, hence the [`GraphicsPsoDesc`] / [`MeshletPsoDesc`] aliases.
#[derive(Clone, Debug)]
pub struct RasterPsoDesc {
    pub topology: Topology,
    pub color_targets: Vec<ColorTarget>,
    /// `None` = no depth.
    pub depth_format: Option<Format>,
    /// Separate from `depth_format`; `None` = no stencil.
    pub stencil_format: Option<Format>,
    pub sample_count: SampleCount,
    pub alpha_to_coverage: bool,
    pub cull: Cull,
    /// `None` = opaque.
    pub blendstate: Option<BlendState>,
    pub label: Option<String>,
}

/// Raster state for a vertex/pixel pipeline. See [`RasterPsoDesc`].
pub type GraphicsPsoDesc = RasterPsoDesc;

/// Raster state for a mesh/pixel pipeline. The mesh shader replaces the vertex shader;
/// amplification shaders aren't exposed. Requires `VK_EXT_mesh_shader` on Vulkan.
/// See [`RasterPsoDesc`].
pub type MeshletPsoDesc = RasterPsoDesc;

impl Default for RasterPsoDesc {
    fn default() -> Self {
        Self {
            topology: Topology::TriangleList,
            color_targets: vec![ColorTarget::new(Format::B8G8R8A8Srgb)],
            depth_format: Some(Format::D32Float),
            sample_count: SampleCount::S1,
            alpha_to_coverage: false,
            cull: Cull::None,
            stencil_format: None,
            blendstate: None,
            label: None,
        }
    }
}

/// Opaque graphics pipeline state object handle.
pub struct GraphicsPso {
    pub(crate) inner: GraphicsPsoInner,
    pub(crate) _owner: Option<std::rc::Rc<crate::device::DeviceInner>>,
}

pub(crate) enum GraphicsPsoInner {
    #[cfg(feature = "vulkan")]
    Vulkan(Box<crate::backend::vulkan::pipeline::VulkanGraphicsPso>),
    #[cfg(feature = "metal")]
    Metal(Box<crate::backend::metal::pipeline::MetalGraphicsPso>),
}

/// Description for creating a compute pipeline.
#[derive(Clone, Debug)]
pub struct ComputePsoDesc {
    /// Metal only; Vulkan takes it from the shader.
    pub threads_per_threadgroup: [u32; 3],
    pub label: Option<String>,
}

impl Default for ComputePsoDesc {
    fn default() -> Self {
        Self {
            threads_per_threadgroup: [1, 1, 1],
            label: None,
        }
    }
}

/// Opaque compute pipeline state object handle.
pub struct ComputePso {
    pub(crate) inner: ComputePsoInner,
    pub(crate) _owner: Option<std::rc::Rc<crate::device::DeviceInner>>,
}

pub(crate) enum ComputePsoInner {
    #[cfg(feature = "vulkan")]
    Vulkan(Box<crate::backend::vulkan::pipeline::VulkanComputePso>),
    #[cfg(feature = "metal")]
    Metal(Box<crate::backend::metal::pipeline::MetalComputePso>),
}

/// Per-face stencil operation descriptor.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StencilDesc {
    pub test: CompareOp,
    pub fail_op: StencilOp,
    /// Stencil passed, depth passed.
    pub pass_op: StencilOp,
    /// Stencil passed, depth failed.
    pub depth_fail_op: StencilOp,
    pub reference: u8,
}

impl Default for StencilDesc {
    fn default() -> Self {
        Self {
            test: CompareOp::Always,
            fail_op: StencilOp::Keep,
            pass_op: StencilOp::Keep,
            depth_fail_op: StencilOp::Keep,
            reference: 0,
        }
    }
}

/// Dynamic depth-stencil state, set via `set_depth_stencil_state`.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthStencilState {
    /// Empty = disabled; `READ` = test only; `READ|WRITE` = both.
    pub depth_mode: DepthFlags,
    pub depth_test: CompareOp,
    pub depth_bias: f32,
    pub depth_bias_slope_factor: f32,
    pub depth_bias_clamp: f32,
    /// Stencil is off entirely while both masks are zero.
    pub stencil_read_mask: u8,
    pub stencil_write_mask: u8,
    pub stencil_front: StencilDesc,
    pub stencil_back: StencilDesc,
}

impl Default for DepthStencilState {
    fn default() -> Self {
        Self {
            depth_mode: DepthFlags::empty(),
            depth_test: CompareOp::Always,
            depth_bias: 0.0,
            depth_bias_slope_factor: 0.0,
            depth_bias_clamp: 0.0,
            stencil_read_mask: 0,
            stencil_write_mask: 0,
            stencil_front: StencilDesc::default(),
            stencil_back: StencilDesc::default(),
        }
    }
}

impl DepthStencilState {
    pub fn stencil_enabled(&self) -> bool {
        self.stencil_read_mask != 0 || self.stencil_write_mask != 0
    }
}

/// Per-attachment blend descriptor. The write mask lives on [`ColorTarget`], so it applies
/// whether or not blending is enabled.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BlendAttachment {
    pub blend_enable: bool,
    pub src_color: BlendFactor,
    pub dst_color: BlendFactor,
    pub color_op: BlendOp,
    pub src_alpha: BlendFactor,
    pub dst_alpha: BlendFactor,
    pub alpha_op: BlendOp,
}

impl Default for BlendAttachment {
    fn default() -> Self {
        Self {
            blend_enable: false,
            src_color: BlendFactor::One,
            dst_color: BlendFactor::Zero,
            color_op: BlendOp::Add,
            src_alpha: BlendFactor::One,
            dst_alpha: BlendFactor::Zero,
            alpha_op: BlendOp::Add,
        }
    }
}

/// Blend state, baked into a PSO via `RasterPsoDesc::blendstate`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BlendState {
    pub attachments: Vec<BlendAttachment>,
}

impl Default for BlendState {
    fn default() -> Self {
        Self {
            attachments: vec![BlendAttachment::default()],
        }
    }
}

/// Opaque meshlet pipeline state object handle.
pub struct MeshletPso {
    pub(crate) inner: MeshletPsoInner,
    pub(crate) _owner: Option<std::rc::Rc<crate::device::DeviceInner>>,
}

pub(crate) enum MeshletPsoInner {
    #[cfg(feature = "vulkan")]
    Vulkan(Box<crate::backend::vulkan::pipeline::VulkanMeshletPso>),
    #[cfg(feature = "metal")]
    Metal(Box<crate::backend::metal::pipeline::MetalMeshletPso>),
}
