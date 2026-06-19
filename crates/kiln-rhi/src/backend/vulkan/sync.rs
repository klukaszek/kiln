use ash::vk;

/// Vulkan timeline semaphore wrapper.
pub struct VulkanTimelineSemaphore {
    pub(crate) semaphore: vk::Semaphore,
    pub(crate) device: ash::Device,
}

impl VulkanTimelineSemaphore {
    pub fn value(&self) -> u64 {
        unsafe {
            self.device
                .get_semaphore_counter_value(self.semaphore)
                .unwrap_or(0)
        }
    }

    pub fn wait(&self, value: u64, timeout_ns: u64) {
        let semaphores = [self.semaphore];
        let values = [value];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        unsafe {
            let _ = self.device.wait_semaphores(&wait_info, timeout_ns);
        }
    }
}
