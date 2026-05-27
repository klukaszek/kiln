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
    Vulkan(Box<crate::backend::vulkan::device::VulkanQueue>),
    #[cfg(feature = "metal")]
    Metal(Box<crate::backend::metal::device::MetalQueue>),
}

/// What to wait/signal when submitting.
#[derive(Default)]
pub struct SubmitDesc<'a> {
    /// Timeline semaphores to wait on before execution.
    pub wait_semaphores: &'a [(TimelineSemaphore, u64)],
    /// Timeline semaphores to signal after execution.
    pub signal_semaphores: &'a [(TimelineSemaphore, u64)],
}

impl Queue {
    /// Submit a command buffer for execution.
    /// The command buffer is consumed (transient, auto-reclaimed).
    pub fn submit(&self, cmd: CommandBuffer) -> RhiResult<()> {
        self.submit_with_desc(cmd, &SubmitDesc::default())
    }

    /// Submit a command buffer with explicit timeline wait/signal dependencies.
    /// The command buffer is consumed (transient, auto-reclaimed).
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

    /// Acquire the next swapchain image for rendering.
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

    /// Present a rendered swapchain image.
    pub fn present(
        &self,
        swapchain: &Swapchain,
        image_index: u32,
        frame_index: usize,
    ) -> RhiResult<()> {
        match (&self.inner, &swapchain.inner) {
            #[cfg(feature = "vulkan")]
            (QueueInner::Vulkan(q), crate::swapchain::SwapchainInner::Vulkan(sc)) => {
                q.present(sc, image_index, frame_index)
            }
            #[cfg(feature = "metal")]
            (QueueInner::Metal(q), crate::swapchain::SwapchainInner::Metal(sc)) => {
                q.present(sc, image_index, frame_index)
            }
            #[allow(unreachable_patterns)]
            _ => unreachable!("mismatched backend types"),
        }
    }

    /// Submit a command buffer for frame presentation.
    /// Handles semaphore waits/signals and fence signaling for the frame loop.
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

    /// Wait for the queue to be idle.
    pub fn wait_idle(&self) {
        match &self.inner {
            #[cfg(feature = "vulkan")]
            QueueInner::Vulkan(q) => q.wait_idle(),
            #[cfg(feature = "metal")]
            QueueInner::Metal(q) => q.wait_idle(),
        }
    }
}
