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
/// of [`RasterPsoDesc`](crate::RasterPsoDesc). The stage is checked against the slot.
pub struct ShaderModule {
    pub(crate) inner: ShaderModuleInner,
    pub(crate) stage: ShaderStage,
    pub(crate) _owner: Option<std::rc::Rc<crate::device::DeviceInner>>,
}

pub(crate) enum ShaderModuleInner {
    #[cfg(feature = "vulkan")]
    Vulkan(Box<crate::backend::vulkan::shader::VulkanShaderModule>),
    #[cfg(feature = "metal")]
    Metal(Box<crate::backend::metal::shader::MetalShaderModule>),
}

impl ShaderModule {
    pub fn stage(&self) -> ShaderStage {
        self.stage
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
    pub label: Option<&'a str>,
}
