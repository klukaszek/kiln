//! GPU timestamp queries for profiling (e.g. per-frame GPU time).
//!
//! Record timestamps into a [`QueryPool`] on a command buffer with
//! [`CommandBuffer::write_timestamp`](crate::CommandBuffer::write_timestamp), then once the GPU
//! work has completed (e.g. the frame slot's fence has been waited) read the raw tick values back
//! with [`Device::read_timestamps`](crate::Device::read_timestamps) and convert tick deltas to
//! nanoseconds with [`Device::timestamp_period_ns`](crate::Device::timestamp_period_ns).
//!
//! On Vulkan this wraps a `VkQueryPool` (`VK_QUERY_TYPE_TIMESTAMP`); on Metal an `MTL4CounterHeap`
//! of type `Timestamp`. Like other RHI resources, pools are freed explicitly with
//! [`Device::destroy_query_pool`](crate::Device::destroy_query_pool).

/// A pool of GPU timestamp query slots.
pub struct QueryPool {
    pub(crate) inner: QueryPoolInner,
    pub(crate) count: u32,
}

pub(crate) enum QueryPoolInner {
    #[cfg(feature = "vulkan")]
    Vulkan(crate::backend::vulkan::query::VulkanQueryPool),
    #[cfg(feature = "metal")]
    Metal(crate::backend::metal::query::MetalQueryPool),
}

impl QueryPool {
    /// Number of timestamp slots in this pool.
    pub fn count(&self) -> u32 {
        self.count
    }
}
