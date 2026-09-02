use ash::vk;

use crate::{RhiError, RhiResult};

/// Vulkan timeline semaphore wrapper.
pub struct VulkanTimelineSemaphore {
    pub(crate) semaphore: vk::Semaphore,
    pub(crate) device: ash::Device,
}

impl VulkanTimelineSemaphore {
    pub fn value(&self) -> RhiResult<u64> {
        unsafe { self.device.get_semaphore_counter_value(self.semaphore) }
            .map_err(|error| RhiError::SyncError(error.to_string()))
    }

    pub fn wait(&self, value: u64, timeout_ns: u64) -> RhiResult<bool> {
        let semaphores = [self.semaphore];
        let values = [value];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        match unsafe { self.device.wait_semaphores(&wait_info, timeout_ns) } {
            Ok(()) => Ok(true),
            Err(vk::Result::TIMEOUT) => Ok(false),
            Err(error) => Err(RhiError::SyncError(error.to_string())),
        }
    }
}
