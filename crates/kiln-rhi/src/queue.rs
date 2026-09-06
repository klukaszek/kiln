//! GPU queue for command submission and swapchain presentation.

use crate::command::CommandBuffer;
use crate::error::RhiResult;
use crate::swapchain::{AcquiredImage, Swapchain};
use crate::sync::TimelineSemaphore;

/// GPU queue for submission and presentation.
pub struct Queue {
    pub(crate) inner: QueueInner,
}

pub(crate) enum QueueInner {
    #[cfg(feature = "vulkan")]
    Vulkan(Box<crate::backend::vulkan::queue::VulkanQueue>),
    #[cfg(feature = "metal")]
    Metal(Box<crate::backend::metal::device::MetalQueue>),
}

#[derive(Default)]
pub struct SubmitDesc<'a> {
    pub wait_semaphores: &'a [(TimelineSemaphore, u64)],
    pub signal_semaphores: &'a [(TimelineSemaphore, u64)],
}

impl Queue {
    pub fn submit(&self, cmd: CommandBuffer) -> RhiResult<()> {
        self.submit_with_desc(cmd, &SubmitDesc::default())
    }

    pub fn submit_with_desc(&self, cmd: CommandBuffer, desc: &SubmitDesc<'_>) -> RhiResult<()> {
        match (&self.inner, cmd.inner) {
            #[cfg(feature = "vulkan")]
            (QueueInner::Vulkan(q), crate::command::CommandBufferInner::Vulkan(cmd)) => {
                q.submit_with_desc(*cmd, desc)
            }
            #[cfg(feature = "metal")]
            (QueueInner::Metal(q), crate::command::CommandBufferInner::Metal(cmd)) => {
                q.submit_with_desc(*cmd, desc)
            }
            #[allow(unreachable_patterns)]
            _ => unreachable!("mismatched backend types"),
        }
    }

    /// Wait for the frame slot and acquire its image. If recording is abandoned before
    /// submission, acquiring the same slot again reuses its outstanding image. Recreate the
    /// swapchain to discard outstanding acquisitions (for example after a resize).
    pub fn acquire_image(
        &self,
        swapchain: &Swapchain,
        frame_index: usize,
    ) -> RhiResult<AcquiredImage> {
        match (&self.inner, &swapchain.inner) {
            #[cfg(feature = "vulkan")]
            (QueueInner::Vulkan(q), crate::swapchain::SwapchainInner::Vulkan(sc)) => {
                q.acquire_image(sc, frame_index)
            }
            #[cfg(feature = "metal")]
            (QueueInner::Metal(q), crate::swapchain::SwapchainInner::Metal(sc)) => {
                q.acquire_image(sc, frame_index)
            }
            #[allow(unreachable_patterns)]
            _ => unreachable!("mismatched backend types"),
        }
    }

    /// Submits and presents `image_index`.
    pub fn submit_frame(
        &self,
        cmd: CommandBuffer,
        swapchain: &Swapchain,
        frame_index: usize,
        image_index: u32,
    ) -> RhiResult<()> {
        match (&self.inner, cmd.inner, &swapchain.inner) {
            #[cfg(feature = "vulkan")]
            (
                QueueInner::Vulkan(q),
                crate::command::CommandBufferInner::Vulkan(cmd),
                crate::swapchain::SwapchainInner::Vulkan(sc),
            ) => q.submit_frame(*cmd, sc, frame_index, image_index),
            #[cfg(feature = "metal")]
            (
                QueueInner::Metal(q),
                crate::command::CommandBufferInner::Metal(cmd),
                crate::swapchain::SwapchainInner::Metal(sc),
            ) => q.submit_frame(*cmd, sc, frame_index, image_index),
            #[allow(unreachable_patterns)]
            _ => unreachable!("mismatched backend types"),
        }
    }

    pub fn wait_idle(&self) {
        match &self.inner {
            #[cfg(feature = "vulkan")]
            QueueInner::Vulkan(q) => q.wait_idle(),
            #[cfg(feature = "metal")]
            QueueInner::Metal(q) => q.wait_idle(),
        }
    }
}
