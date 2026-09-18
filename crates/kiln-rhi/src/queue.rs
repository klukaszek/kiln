//! GPU queue for command submission and swapchain presentation.

use crate::command::CommandBuffer;
use crate::error::RhiResult;
use crate::swapchain::Swapchain;
use crate::sync::TimelineSemaphore;

/// GPU queue for submission and presentation.
pub struct Queue {
    pub(crate) inner: QueueInner,
}

backend_enum!(QueueInner {
    vulkan: std::rc::Rc<crate::backend::vulkan::queue::VulkanQueue>,
    metal: std::rc::Rc<crate::backend::metal::queue::MetalQueue>,
});

#[derive(Default)]
pub struct SubmitDesc<'a> {
    pub wait_semaphores: &'a [(&'a TimelineSemaphore, u64)],
    pub signal_semaphores: &'a [(&'a TimelineSemaphore, u64)],
}

impl Queue {
    pub fn submit(&self, cmd: CommandBuffer) -> RhiResult<()> {
        self.submit_with_desc(cmd, &SubmitDesc::default())
    }

    pub fn submit_with_desc(&self, cmd: CommandBuffer, desc: &SubmitDesc<'_>) -> RhiResult<()> {
        {
            let q = &self.inner;
            let cmd = cmd.inner;
            q.submit_with_desc(*cmd, desc)
        }
    }

    /// Wait for the frame slot and acquire its image. If recording is abandoned before
    /// submission, acquiring the same slot again reuses its outstanding image. Recreate the
    /// swapchain to discard outstanding acquisitions (for example after a resize).
    /// Returns the index of the acquired image, for
    /// [`RenderTarget::swapchain_image`](crate::RenderTarget::swapchain_image) and
    /// [`submit_frame`](Self::submit_frame).
    pub fn acquire_image(&self, swapchain: &Swapchain, frame_index: usize) -> RhiResult<u32> {
        {
            let q = &self.inner;
            let sc = &swapchain.inner;
            q.acquire_image(sc, frame_index)
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
        {
            let q = &self.inner;
            let cmd = cmd.inner;
            let sc = &swapchain.inner;
            q.submit_frame(*cmd, sc, frame_index, image_index)
        }
    }

    pub fn wait_idle(&self) {
        match &self.inner {
            #[cfg(feature = "vulkan")]
            q => q.wait_idle(),
            #[cfg(feature = "metal")]
            q => q.wait_idle(),
        }
    }
}
