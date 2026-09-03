//! GPU timestamp queries.
//!
//! `cmd.write_timestamp` records into a pool; once the work has completed,
//! `device.read_timestamps` returns raw ticks and `device.timestamp_period_ns` scales them.

/// A pool of GPU timestamp query slots.
pub struct QueryPool {
    pub(crate) inner: QueryPoolInner,
    pub(crate) count: u32,
    pub(crate) _owner: Option<std::rc::Rc<crate::device::DeviceInner>>,
}

pub(crate) enum QueryPoolInner {
    #[cfg(feature = "vulkan")]
    Vulkan(crate::backend::vulkan::query::VulkanQueryPool),
    #[cfg(feature = "metal")]
    Metal(crate::backend::metal::query::MetalQueryPool),
}

impl QueryPool {
    pub fn count(&self) -> u32 {
        self.count
    }
}
