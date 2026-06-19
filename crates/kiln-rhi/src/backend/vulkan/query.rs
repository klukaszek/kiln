//! Vulkan timestamp query pool (`VK_QUERY_TYPE_TIMESTAMP`).

use ash::vk;

/// Backing for a [`crate::QueryPool`] on Vulkan. Freed explicitly via
/// `Device::destroy_query_pool` (RHI resources are not RAII for device-owned storage).
pub struct VulkanQueryPool {
    pub(crate) pool: vk::QueryPool,
}
