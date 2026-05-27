use ash::vk;

/// Vulkan acceleration structure entry (BLAS or TLAS).
///
/// `acceleration_structure` is an opaque `VkAccelerationStructureKHR`.
/// `buffer` / `buffer_memory` hold the backing storage for the AS data.
/// `device_address` is the GPU address passed into TLAS instance descriptors
/// and into the shader for `gpuSetActiveTextureHeapPtr`-style root data.
pub struct VulkanAccelerationStructure {
    pub(crate) acceleration_structure: vk::AccelerationStructureKHR,
    pub(crate) buffer: vk::Buffer,
    pub(crate) buffer_memory: vk::DeviceMemory,
    /// The GPU-visible address of this acceleration structure.
    /// Use this value in `TlasInstance::acceleration_structure_reference`
    /// and in root structs where the shader accesses it via `TraceRayInline`.
    pub(crate) device_address: u64,
    pub(crate) device: ash::Device,
    /// Extension loader for `VK_KHR_acceleration_structure`.
    pub(crate) accel_loader: ash::khr::acceleration_structure::Device,
}

impl Drop for VulkanAccelerationStructure {
    fn drop(&mut self) {
        unsafe {
            self.accel_loader
                .destroy_acceleration_structure(self.acceleration_structure, None);
            self.device.free_memory(self.buffer_memory, None);
            self.device.destroy_buffer(self.buffer, None);
        }
    }
}
