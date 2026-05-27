use ash::vk;

/// Vulkan shader module wrapper.
pub struct VulkanShaderModule {
    pub(crate) module: vk::ShaderModule,
    pub(crate) entry_point: std::ffi::CString,
}
