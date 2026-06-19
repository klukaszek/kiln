use ash::vk;

/// Vulkan shader module wrapper.
///
/// Owns its `vk::ShaderModule` and destroys it on drop — the frontend `ShaderModule`
/// is the single owner (no device-side registry). A module only needs to outlive the
/// `create_*_pso` call that consumes it; the created pipeline keeps its own copy.
pub struct VulkanShaderModule {
    pub(crate) module: vk::ShaderModule,
    pub(crate) entry_point: std::ffi::CString,
    device: ash::Device,
}

impl VulkanShaderModule {
    pub(crate) fn new(
        device: ash::Device,
        module: vk::ShaderModule,
        entry_point: std::ffi::CString,
    ) -> Self {
        Self {
            module,
            entry_point,
            device,
        }
    }
}

impl Drop for VulkanShaderModule {
    fn drop(&mut self) {
        // SAFETY: `module` was created from `device` and is not used after drop; pipelines
        // built from it retain their own copy of the shader, so this does not dangle.
        unsafe { self.device.destroy_shader_module(self.module, None) };
    }
}
