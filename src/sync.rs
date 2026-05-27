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
        match &self.inner {
            #[cfg(feature = "vulkan")]
            TimelineSemaphoreInner::Vulkan(s) => s.value(),
            #[cfg(feature = "metal")]
            TimelineSemaphoreInner::Metal(s) => s.value(),
        }
    }

    /// CPU-side wait until the semaphore reaches `value`.
    pub fn wait(&self, value: u64, timeout_ns: u64) {
        match &self.inner {
            #[cfg(feature = "vulkan")]
            TimelineSemaphoreInner::Vulkan(s) => s.wait(value, timeout_ns),
            #[cfg(feature = "metal")]
            TimelineSemaphoreInner::Metal(s) => s.wait(value, timeout_ns),
        }
    }
}
