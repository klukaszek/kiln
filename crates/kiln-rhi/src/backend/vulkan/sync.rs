use ash::vk;
use ash::vk::TaggedStructure as _;

use super::device::VulkanDevice;
use crate::sync::TimelineSemaphore;
use crate::{RhiError, RhiResult};

/// Vulkan timeline semaphore wrapper.
pub struct VulkanTimelineSemaphore {
    pub(crate) semaphore: vk::Semaphore,
    pub(crate) device: ash::Device,
}

impl VulkanTimelineSemaphore {
    pub fn value(&self) -> RhiResult<u64> {
        unsafe { self.device.get_semaphore_counter_value(self.semaphore) }
            .map_err(|error| RhiError::SyncError(error.into()))
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
            Err(error) => Err(RhiError::SyncError(error.into())),
        }
    }
}

impl VulkanDevice {
    pub fn create_timeline_semaphore(&self, initial_value: u64) -> RhiResult<TimelineSemaphore> {
        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(initial_value);

        let semaphore_info = vk::SemaphoreCreateInfo::default().push(&mut type_info);

        let semaphore = unsafe {
            self.loaders
                .device
                .create_semaphore(&semaphore_info, None)
                .map_err(|e| RhiError::SyncError(e.into()))?
        };
        self.timeline_semaphores.borrow_mut().push(semaphore);

        Ok(TimelineSemaphore {
            inner: Box::new(VulkanTimelineSemaphore {
                semaphore,
                device: self.loaders.device.clone(),
            }),
            _owner: None,
        })
    }
}
