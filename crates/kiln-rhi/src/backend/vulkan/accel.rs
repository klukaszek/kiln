use ash::vk;

use super::device::{SharedPendingRetirements, VulkanRetiredResource};

/// Vulkan acceleration structure entry (BLAS or TLAS).
///
/// `acceleration_structure` is an opaque `VkAccelerationStructureKHR`.
/// `buffer` / `buffer_memory` hold the backing storage for the AS data.
pub struct VulkanAccelerationStructure {
    pub(crate) acceleration_structure: vk::AccelerationStructureKHR,
    pub(crate) buffer: vk::Buffer,
    pub(crate) buffer_memory: vk::DeviceMemory,
    /// The GPU-visible address of this acceleration structure.
    /// Use this value in `TlasInstance::acceleration_structure_reference`
    /// and in root structs where the shader accesses it via `TraceRayInline`.
    pub(crate) device_address: u64,
    /// Build scratch storage, owned by the structure so it outlives the GPU build:
    /// `build_blas`/`build_tlas` only *record* the build into a command buffer that
    /// executes later, so the scratch must stay alive until then. `scratch_address`
    /// is aligned to the device's `minAccelerationStructureScratchOffsetAlignment`.
    pub(crate) scratch_buffer: vk::Buffer,
    pub(crate) scratch_memory: vk::DeviceMemory,
    pub(crate) scratch_address: u64,
    /// Extension loader for `VK_KHR_acceleration_structure`.
    pub(crate) accel_loader: ash::khr::acceleration_structure::Device,
    pub(crate) pending_retired_resources: SharedPendingRetirements,
}

impl Drop for VulkanAccelerationStructure {
    fn drop(&mut self) {
        self.pending_retired_resources
            .lock()
            .expect("pending retired resource lock poisoned")
            .push(VulkanRetiredResource::AccelerationStructure {
                acceleration_structure: self.acceleration_structure,
                buffer: self.buffer,
                buffer_memory: self.buffer_memory,
                scratch_buffer: self.scratch_buffer,
                scratch_memory: self.scratch_memory,
                accel_loader: self.accel_loader.clone(),
            });
    }
}
