//! Shader module loading and stage types.

/// Shader stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    Vertex,
    Pixel,
    Compute,
    Mesh,
}

/// A compiled shader module.
///
/// Passed by reference to `create_*_pso`: shaders are arguments to pipeline creation, not fields
/// of [`RasterPsoDesc`](crate::RasterPsoDesc). Each `create_*_pso` checks [`stage`](Self::stage)
/// against the slot it was passed in, so swapping the vertex and pixel arguments is an error
/// rather than a driver-level surprise.
pub struct ShaderModule {
    pub(crate) inner: ShaderModuleInner,
    pub(crate) stage: ShaderStage,
    pub(crate) threads_per_threadgroup: Option<[u32; 3]>,
    pub(crate) _owner: Option<std::rc::Rc<crate::device::DeviceInner>>,
}

backend_enum!(ShaderModuleInner { vulkan: Box<crate::backend::vulkan::shader::VulkanShaderModule>, metal: Box<crate::backend::metal::shader::MetalShaderModule> });

impl ShaderModule {
    pub fn stage(&self) -> ShaderStage {
        self.stage
    }

    /// The threadgroup size a compute entry point declared with `[numthreads]`, when it is known.
    ///
    /// [`compiler::compile`](crate::compiler::compile) fills this from slangc's reflection, so a
    /// [`ComputePsoDesc`](crate::ComputePsoDesc) built from such a module does not have to repeat
    /// it. `None` for a module built straight from bytes without
    /// [`ShaderModuleDesc::threads_per_threadgroup`].
    pub fn threads_per_threadgroup(&self) -> Option<[u32; 3]> {
        self.threads_per_threadgroup
    }
}

/// Description for creating a shader module.
pub struct ShaderModuleDesc<'a> {
    /// SPIR-V bytecode (Vulkan) or MSL source/metallib (Metal).
    pub code: &'a [u8],
    /// Entry point function name.
    pub entry_point: &'a str,
    /// Shader stage.
    pub stage: ShaderStage,
    /// The `[numthreads]` size this compute entry point declares, if known.
    ///
    /// Metal cannot recover it from a compiled `metallib`, so it has to be supplied out of band;
    /// Vulkan takes it from the SPIR-V. Setting it here lets `create_compute_pso` work the same
    /// way on both, and lets it reject a [`ComputePsoDesc`](crate::ComputePsoDesc) that disagrees
    /// with the shader. `None` means the caller will state it on the PSO instead.
    pub threads_per_threadgroup: Option<[u32; 3]>,
    pub label: Option<&'a str>,
}
