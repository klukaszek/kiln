use crate::types::GpuAddress;
use ash::vk;

/// Vulkan buffer with buffer_device_address support.
pub struct VulkanBuffer {
    pub(crate) buffer: vk::Buffer,
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) size: u64,
    pub(crate) mapped_ptr: Option<*mut u8>,
    pub(crate) gpu_address: GpuAddress,
}

// SAFETY: VulkanBuffer's raw pointer is only used for CPU-side uploads
// and the underlying Vulkan memory is externally synchronized.
unsafe impl Send for VulkanBuffer {}
unsafe impl Sync for VulkanBuffer {}

impl VulkanBuffer {
    pub fn mapped_ptr(&self) -> Option<*mut u8> {
        self.mapped_ptr
    }

    pub fn gpu_address(&self) -> GpuAddress {
        self.gpu_address
    }

    pub fn size(&self) -> u64 {
        self.size
    }
}
