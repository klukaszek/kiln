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
/// Owns its backend resource and is passed by reference to `create_*_pso`
/// (matching `gpuCreateGraphicsPipeline(vertexIR, pixelIR, desc)` in the spec — shaders
/// are arguments to pipeline creation, not fields of the raster desc).
pub struct ShaderModule {
    pub(crate) inner: ShaderModuleInner,
    pub(crate) stage: ShaderStage,
}

pub(crate) enum ShaderModuleInner {
    #[cfg(feature = "vulkan")]
    Vulkan(crate::backend::vulkan::shader::VulkanShaderModule),
    #[cfg(feature = "metal")]
    Metal(crate::backend::metal::shader::MetalShaderModule),
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
