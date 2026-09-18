//! Timeline semaphore for CPU/GPU and frame synchronization.

use crate::RhiResult;

/// Timeline semaphore for frame synchronization.
pub struct TimelineSemaphore {
    pub(crate) inner: TimelineSemaphoreInner,
    pub(crate) _owner: Option<std::rc::Rc<crate::device::DeviceInner>>,
}

backend_enum!(TimelineSemaphoreInner { vulkan: Box<crate::backend::vulkan::sync::VulkanTimelineSemaphore>, metal: Box<crate::backend::metal::sync::MetalTimelineSemaphore> });

impl TimelineSemaphore {
    pub fn value(&self) -> RhiResult<u64> {
        {
            let s = &self.inner;
            s.value()
        }
    }

    /// `Ok(true)` if the value was reached, `Ok(false)` on timeout. Backend failures are `Err`
    /// rather than being mistaken for a successful wait.
    pub fn wait(&self, value: u64, timeout_ns: u64) -> RhiResult<bool> {
        {
            let s = &self.inner;
            s.wait(value, timeout_ns)
        }
    }
}
