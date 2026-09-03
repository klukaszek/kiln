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
    pub fn value(&self) -> RhiResult<u64> {
        backend_dispatch!(&self.inner, TimelineSemaphoreInner, s => s.value())
    }

    /// `Ok(true)` if the value was reached, `Ok(false)` on timeout. Backend failures are `Err`
    /// rather than being mistaken for a successful wait.
    pub fn wait(&self, value: u64, timeout_ns: u64) -> RhiResult<bool> {
        backend_dispatch!(&self.inner, TimelineSemaphoreInner, s => s.wait(value, timeout_ns))
    }
}
