//! Vulkan timestamp query pool (`VK_QUERY_TYPE_TIMESTAMP`).

use ash::vk;

use super::device::VulkanDevice;
use crate::error::{RhiError, RhiResult};
use crate::query::QueryPool;

/// Backing for a [`crate::QueryPool`] on Vulkan. The device registry owns the native lifetime.
pub struct VulkanQueryPool {
    pub(crate) pool: vk::QueryPool,
}

impl VulkanDevice {
    pub fn create_query_pool(&self, count: u32) -> RhiResult<QueryPool> {
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(count);
        let pool = unsafe { self.loaders.device.create_query_pool(&info, None) }
            .map_err(|e| RhiError::Backend(format!("create_query_pool: {e}").into()))?;
        // Vulkan requires a query to be reset before its first use.
        unsafe { self.loaders.device.reset_query_pool(pool, 0, count) };
        self.query_pools.borrow_mut().push(pool);
        Ok(QueryPool {
            inner: VulkanQueryPool { pool },
            count,
            _owner: None,
        })
    }

    pub fn destroy_query_pool(&self, pool: VulkanQueryPool) {
        self.query_pools
            .borrow_mut()
            .retain(|&entry| entry != pool.pool);
        unsafe { self.loaders.device.destroy_query_pool(pool.pool, None) };
    }

    pub fn timestamp_period_ns(&self) -> f64 {
        self.timestamp_period as f64
    }

    /// Resolve into the caller's slice. `vkGetQueryPoolResults` writes straight into it, so this
    /// is the primitive and the `Vec` form is the convenience.
    pub fn read_timestamps_into(
        &self,
        pool: &VulkanQueryPool,
        count: u32,
        out: &mut [u64],
    ) -> RhiResult<()> {
        let vk_pool = pool.pool;
        let slots = &mut out[..count as usize];
        // The frame fence has already completed; unwritten slots remain zero.
        // SAFETY: `vk_pool` is this device's, and `slots` has one `u64` per query.
        let result = unsafe {
            self.loaders.device.get_query_pool_results(
                vk_pool,
                0,
                slots,
                vk::QueryResultFlags::TYPE_64,
            )
        };
        match result {
            Ok(()) => Ok(()),
            Err(vk::Result::NOT_READY) => {
                slots.fill(0);
                Ok(())
            }
            Err(e) => Err(RhiError::Backend(
                format!("get_query_pool_results: {e}").into(),
            )),
        }
    }
}
