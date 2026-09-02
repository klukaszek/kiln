//! Vulkan timestamp query pool (`VK_QUERY_TYPE_TIMESTAMP`).

use ash::vk;

/// Backing for a [`crate::QueryPool`] on Vulkan. The device registry owns the native lifetime.
pub struct VulkanQueryPool {
    pub(crate) pool: vk::QueryPool,
}
