/// Timeline semaphore for frame synchronization.
pub struct TimelineSemaphore {
    pub(crate) inner: TimelineSemaphoreInner,
}

pub(crate) enum TimelineSemaphoreInner {
    #[cfg(feature = "vulkan")]
    Vulkan(Box<crate::backend::vulkan::sync::VulkanTimelineSemaphore>),
    #[cfg(feature = "metal")]
    Metal(Box<crate::backend::metal::sync::MetalTimelineSemaphore>),
}

impl TimelineSemaphore {
    /// Get the current signaled value.
    pub fn value(&self) -> u64 {
        backend_dispatch!(&self.inner, TimelineSemaphoreInner, s => s.value())
    }

    /// CPU-side wait until the semaphore reaches `value`.
    pub fn wait(&self, value: u64, timeout_ns: u64) {
        backend_dispatch!(&self.inner, TimelineSemaphoreInner, s => s.wait(value, timeout_ns))
    }
}
