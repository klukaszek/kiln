//! Timeline semaphore for CPU/GPU and frame synchronization.

use crate::RhiResult;

/// Timeline semaphore for frame synchronization.
pub struct TimelineSemaphore {
    pub(crate) inner: TimelineSemaphoreInner,
    pub(crate) _owner: Option<std::rc::Rc<crate::device::DeviceInner>>,
}

pub(crate) enum TimelineSemaphoreInner {
    #[cfg(feature = "vulkan")]
    Vulkan(Box<crate::backend::vulkan::sync::VulkanTimelineSemaphore>),
    #[cfg(feature = "metal")]
    Metal(Box<crate::backend::metal::sync::MetalTimelineSemaphore>),
}

impl TimelineSemaphore {
    /// Get the current signaled value.
    pub fn value(&self) -> RhiResult<u64> {
        backend_dispatch!(&self.inner, TimelineSemaphoreInner, s => s.value())
    }

    /// CPU-side wait until the semaphore reaches `value`.
    ///
    /// Returns `Ok(true)` when the value was reached and `Ok(false)` when the timeout expired.
    /// Backend failures are returned instead of being mistaken for a successful wait.
    pub fn wait(&self, value: u64, timeout_ns: u64) -> RhiResult<bool> {
        backend_dispatch!(&self.inner, TimelineSemaphoreInner, s => s.wait(value, timeout_ns))
    }
}
