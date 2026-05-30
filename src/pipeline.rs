use crate::types::*;

/// Per-color-attachment entry in a graphics PSO.
///
/// Matches Aaltonen's `ColorTarget { FORMAT format; uint8 writeMask; }`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ColorTarget {
    pub format: Format,
    /// Static write mask baked into the PSO for dead-code elimination.
    /// Disable unused outputs (e.g. set to empty) to allow the compiler to
    /// eliminate dead pixel shader outputs, reducing PSO permutations.
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

/// Description for creating a graphics pipeline state object.
///
/// Minimal PSO — only topology, color/depth formats, MSAA, cull, and write masks baked.
/// DepthStencil and Blend are separate flyweight objects (`set_depth_stencil_state` /
/// `set_blend_state`) to minimise PSO permutations.
///
/// Matches Aaltonen's `GpuRasterDesc`.
///
/// Shaders are not part of this desc — they are passed as `&ShaderModule` arguments to
/// `create_graphics_pso`, matching the spec's `gpuCreateGraphicsPipeline(vertexIR, pixelIR, desc)`.
#[derive(Clone, Debug)]
pub struct GraphicsPsoDesc {
    /// Primitive topology.
    pub topology: Topology,
    /// Color render targets. Each entry bakes the format and static write mask.
    pub color_targets: Vec<ColorTarget>,
    /// Depth attachment format (None = no depth).
    pub depth_format: Option<Format>,
    /// MSAA sample count.
    pub sample_count: SampleCount,
    /// Enable alpha-to-coverage.
    pub alpha_to_coverage: bool,
    /// Size of root constants in bytes (root table: base + stride).
    pub root_constant_size: u32,
    /// Cull mode. Encodes cull direction and implied front-face winding (`Cull::Cw` = standard back-face culling).
    pub cull: Cull,
    /// Separate stencil attachment format (None = no stencil). Distinct from `depth_format`.
    pub stencil_format: Option<Format>,
    /// Enable dual-source blending (requires `blendstate` with two outputs).
    pub support_dual_source_blending: bool,
    /// Optional pre-baked blend state. When `Some`, this variant is compiled into the PSO
    /// at creation time and used as the default — matching Aaltonen's `GpuRasterDesc.blendstate`.
    /// When `None`, blend state is supplied per-draw via `cmd.set_blend_state(...)`.
    pub blendstate: Option<BlendState>,
    pub label: Option<String>,
}

impl Default for GraphicsPsoDesc {
    fn default() -> Self {
        Self {
            topology: Topology::TriangleList,
            color_targets: vec![ColorTarget::new(Format::B8G8R8A8Srgb)],
            depth_format: Some(Format::D32Float),
            sample_count: SampleCount::S1,
            alpha_to_coverage: false,
            root_constant_size: (std::mem::size_of::<GpuAddress>() * 4) as u32,
            cull: Cull::None,
            stencil_format: None,
            support_dual_source_blending: false,
            blendstate: None,
            label: None,
        }
    }
}

/// Opaque graphics pipeline state object handle.
pub struct GraphicsPso {
    pub(crate) inner: GraphicsPsoInner,
}

pub(crate) enum GraphicsPsoInner {
    #[cfg(feature = "vulkan")]
    Vulkan(Box<crate::backend::vulkan::pipeline::VulkanGraphicsPso>),
    #[cfg(feature = "metal")]
    Metal(Box<crate::backend::metal::pipeline::MetalGraphicsPso>),
}

/// Description for creating a compute pipeline.
///
/// The compute shader is passed as a `&ShaderModule` argument to `create_compute_pso`,
/// matching the spec's `gpuCreateComputePipeline(computeIR)`.
#[derive(Clone, Debug)]
pub struct ComputePsoDesc {
    /// Size of root constants in bytes.
    pub root_constant_size: u32,
    /// Threads per threadgroup (Metal dispatch requires this).
    /// Vulkan ignores this value.
    pub threads_per_threadgroup: [u32; 3],
    pub label: Option<String>,
}

impl Default for ComputePsoDesc {
    fn default() -> Self {
        Self {
            root_constant_size: std::mem::size_of::<GpuAddress>() as u32,
            threads_per_threadgroup: [1, 1, 1],
            label: None,
        }
    }
}

/// Opaque compute pipeline state object handle.
pub struct ComputePso {
    pub(crate) inner: ComputePsoInner,
}

pub(crate) enum ComputePsoInner {
    #[cfg(feature = "vulkan")]
    Vulkan(crate::backend::vulkan::pipeline::VulkanComputePso),
    #[cfg(feature = "metal")]
    Metal(crate::backend::metal::pipeline::MetalComputePso),
}

/// Per-face stencil operation descriptor.
///
/// Matches Aaltonen's `Stencil` struct.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StencilDesc {
    /// Comparison function applied against the stencil buffer.
    pub test: CompareOp,
    /// Action when stencil test fails.
    pub fail_op: StencilOp,
    /// Action when stencil test passes and depth test passes.
    pub pass_op: StencilOp,
    /// Action when stencil test passes but depth test fails.
    pub depth_fail_op: StencilOp,
    /// Reference value compared against the stencil buffer.
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

/// Separate depth-stencil state (flyweight object).
///
/// Matches Aaltonen's `GpuDepthStencilDesc`. Set dynamically via `set_depth_stencil_state`.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthStencilState {
    /// Depth read/write mode. `0` = disabled; `READ` = test only; `READ|WRITE` = full.
    pub depth_mode: DepthFlags,
    /// Depth compare function. Applied when `depth_mode` has `READ`.
    pub depth_test: CompareOp,
    /// Constant depth bias added to each fragment's depth.
    pub depth_bias: f32,
    /// Slope-scaled depth bias.
    pub depth_bias_slope_factor: f32,
    /// Clamp applied to the total depth bias.
    pub depth_bias_clamp: f32,
    /// Stencil buffer read mask (ANDed with the stored stencil value before comparison).
    pub stencil_read_mask: u8,
    /// Stencil buffer write mask (ANDed with the written stencil value).
    pub stencil_write_mask: u8,
    /// Front-face stencil operations. Only active when stencil_read/write_mask != 0.
    pub stencil_front: StencilDesc,
    /// Back-face stencil operations.
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
            stencil_read_mask: 0xff,
            stencil_write_mask: 0xff,
            stencil_front: StencilDesc::default(),
            stencil_back: StencilDesc::default(),
        }
    }
}

impl DepthStencilState {
    /// Returns true if stencil testing/writing is active (either mask is non-zero).
    pub fn stencil_enabled(&self) -> bool {
        self.stencil_read_mask != 0 || self.stencil_write_mask != 0
    }
}

/// Per-attachment blend descriptor.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BlendAttachment {
    pub blend_enable: bool,
    pub src_color: BlendFactor,
    pub dst_color: BlendFactor,
    pub color_op: BlendOp,
    pub src_alpha: BlendFactor,
    pub dst_alpha: BlendFactor,
    pub alpha_op: BlendOp,
    pub write_mask: ColorWriteMask,
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
            write_mask: ColorWriteMask::ALL,
        }
    }
}

/// Separate blend state (flyweight object).
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

// ---------------------------------------------------------------------------
// Meshlet (mesh shader) pipeline — gpuCreateGraphicsMeshletPipeline
// ---------------------------------------------------------------------------

/// Description for creating a mesh-shader graphics pipeline.
///
/// Matches Aaltonen's `gpuCreateGraphicsMeshletPipeline(meshletIR, pixelIR, desc)`.
/// The mesh shader replaces the vertex shader entirely; amplification shaders
/// are not exposed (use a compute prepass or a root-pointer-addressed amplification
/// in the mesh shader itself).
///
/// On Vulkan, requires `VK_EXT_mesh_shader`.
/// On Metal, this backend targets the Metal 4 mesh render pipeline path.
///
/// Mesh and pixel shaders are passed as `&ShaderModule` arguments to `create_meshlet_pso`,
/// matching the spec's `gpuCreateGraphicsMeshletPipeline(meshletIR, pixelIR, desc)`.
#[derive(Clone, Debug)]
pub struct MeshletPsoDesc {
    /// Rasterizer state — same fields as `GraphicsPsoDesc`.
    pub topology: Topology,
    pub color_targets: Vec<ColorTarget>,
    pub depth_format: Option<Format>,
    pub stencil_format: Option<Format>,
    pub sample_count: SampleCount,
    pub alpha_to_coverage: bool,
    pub cull: Cull,
    pub support_dual_source_blending: bool,
    /// Optional pre-baked blend state.
    pub blendstate: Option<BlendState>,
    /// Root constant size in bytes (passed via the mesh shader's root pointer).
    pub root_constant_size: u32,
    pub label: Option<String>,
}

impl Default for MeshletPsoDesc {
    fn default() -> Self {
        Self {
            topology: Topology::TriangleList,
            color_targets: vec![ColorTarget::new(Format::B8G8R8A8Srgb)],
            depth_format: Some(Format::D32Float),
            stencil_format: None,
            sample_count: SampleCount::S1,
            alpha_to_coverage: false,
            cull: Cull::None,
            support_dual_source_blending: false,
            blendstate: None,
            root_constant_size: (std::mem::size_of::<crate::types::GpuAddress>() * 2) as u32,
            label: None,
        }
    }
}

/// Opaque meshlet pipeline state object handle.
pub struct MeshletPso {
    pub(crate) inner: MeshletPsoInner,
}

pub(crate) enum MeshletPsoInner {
    #[cfg(feature = "vulkan")]
    Vulkan(Box<crate::backend::vulkan::pipeline::VulkanMeshletPso>),
    #[cfg(feature = "metal")]
    Metal(Box<crate::backend::metal::pipeline::MetalMeshletPso>),
}
