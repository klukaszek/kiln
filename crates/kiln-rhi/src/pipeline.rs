//! Pipeline state objects: graphics, compute, and mesh-shader PSOs.

use crate::types::{
    BlendFactor, BlendOp, ColorWriteMask, CompareOp, Cull, DepthFlags, Format, SampleCount,
    Topology,
};

/// Per-color-attachment entry in a graphics PSO.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ColorTarget {
    pub format: Format,
    /// Static write mask baked into the PSO; set empty to dead-code-eliminate unused outputs.
    pub write_mask: ColorWriteMask,
}

impl ColorTarget {
    /// Write every channel.
    pub const fn new(format: Format) -> Self {
        Self::new_const(format)
    }

    /// `new` in const context; `ColorWriteMask::all()` is not const, so the bits are named here.
    pub(crate) const fn new_const(format: Format) -> Self {
        Self {
            format,
            write_mask: ColorWriteMask::ALL,
        }
    }
}

/// Description for creating a rasterization pipeline state object.
///
/// Minimal PSO — topology, formats, MSAA, cull, write masks, blend and depth all baked. Shaders
/// are arguments to `create_graphics_pso` / `create_meshlet_pso`; the vertex and mesh paths share
/// this state, hence the [`GraphicsPsoDesc`] / [`MeshletPsoDesc`] aliases.
#[derive(Clone, Copy, Debug)]
pub struct RasterPsoDesc<'a> {
    pub topology: Topology,
    /// Borrowed like `label`: describing a pipeline should not allocate.
    pub color_targets: &'a [ColorTarget],
    /// `None` = no depth attachment and no depth test.
    pub depth_format: Option<Format>,
    pub depth: DepthState,
    pub sample_count: SampleCount,
    pub alpha_to_coverage: bool,
    pub cull: Cull,
    /// Per-target blending, in the same order as `color_targets`. Empty means opaque, and a
    /// target past the end of this slice is opaque too.
    pub blend: &'a [BlendAttachment],
    pub label: Option<&'a str>,
}

/// Raster state for a vertex/pixel pipeline. See [`RasterPsoDesc`].
pub type GraphicsPsoDesc<'a> = RasterPsoDesc<'a>;

/// Raster state for a mesh/pixel pipeline. The mesh shader replaces the vertex shader;
/// amplification shaders aren't exposed. Requires `VK_EXT_mesh_shader` on Vulkan.
/// See [`RasterPsoDesc`].
pub type MeshletPsoDesc<'a> = RasterPsoDesc<'a>;

/// A `const` item so [`RasterPsoDesc::default`] can borrow it for `'static`.
const DEFAULT_COLOR_TARGETS: &[ColorTarget] = &[ColorTarget::new_const(Format::B8G8R8A8Srgb)];

impl Default for RasterPsoDesc<'_> {
    /// Opaque triangles into one sRGB BGRA target, no depth attachment and no depth test.
    ///
    /// The depth default used to declare a `D32Float` attachment while leaving [`DepthState`]
    /// disabled, which reads as "depth on" and behaves as "depth off". A depth attachment is
    /// something you opt into along with the state that uses it.
    fn default() -> Self {
        Self {
            topology: Topology::TriangleList,
            color_targets: DEFAULT_COLOR_TARGETS,
            depth_format: None,
            depth: DepthState::default(),
            sample_count: SampleCount::S1,
            alpha_to_coverage: false,
            cull: Cull::None,
            blend: &[],
            label: None,
        }
    }
}

/// Opaque graphics pipeline state object handle.
pub struct GraphicsPso {
    pub(crate) inner: GraphicsPsoInner,
    pub(crate) _owner: Option<std::rc::Rc<crate::device::DeviceInner>>,
}

backend_enum!(GraphicsPsoInner { vulkan: std::rc::Rc<crate::backend::vulkan::pipeline::VulkanGraphicsPso>, metal: Box<crate::backend::metal::pipeline::MetalGraphicsPso> });

/// Description for creating a compute pipeline.
#[derive(Clone, Debug, Default)]
pub struct ComputePsoDesc<'a> {
    /// The shader's `[numthreads]` size, or `None` to take it from the
    /// [`ShaderModule`](crate::ShaderModule)'s reflection. Metal needs it explicitly (a
    /// `metallib` does not carry it); disagreeing with the module is an error.
    pub threads_per_threadgroup: Option<[u32; 3]>,
    /// Borrowed: a descriptor is an argument, and both backends copy the name into their own
    /// string type before the call returns.
    pub label: Option<&'a str>,
}

/// Opaque compute pipeline state object handle.
pub struct ComputePso {
    pub(crate) inner: ComputePsoInner,
    pub(crate) _owner: Option<std::rc::Rc<crate::device::DeviceInner>>,
}

backend_enum!(ComputePsoInner { vulkan: std::rc::Rc<crate::backend::vulkan::pipeline::VulkanComputePso>, metal: Box<crate::backend::metal::pipeline::MetalComputePso> });

/// Depth test and bias, baked into the pipeline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthState {
    /// Empty = disabled; `READ` = test only; `READ | WRITE` = both.
    pub mode: DepthFlags,
    pub compare: CompareOp,
    pub bias: f32,
    pub bias_slope: f32,
    pub bias_clamp: f32,
}

impl Default for DepthState {
    fn default() -> Self {
        Self {
            mode: DepthFlags::empty(),
            compare: CompareOp::Always,
            bias: 0.0,
            bias_slope: 0.0,
            bias_clamp: 0.0,
        }
    }
}

impl DepthState {
    /// Depth test enabled, writing enabled, `compare` as given.
    pub fn read_write(compare: CompareOp) -> Self {
        Self {
            mode: DepthFlags::READ | DepthFlags::WRITE,
            compare,
            ..Default::default()
        }
    }
}

/// Per-attachment blend descriptor. The write mask lives on [`ColorTarget`], so it applies
/// whether or not blending is enabled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

/// Opaque meshlet pipeline state object handle.
pub struct MeshletPso {
    pub(crate) inner: MeshletPsoInner,
    pub(crate) _owner: Option<std::rc::Rc<crate::device::DeviceInner>>,
}

backend_enum!(MeshletPsoInner { vulkan: std::rc::Rc<crate::backend::vulkan::pipeline::VulkanMeshletPso>, metal: Box<crate::backend::metal::pipeline::MetalMeshletPso> });
