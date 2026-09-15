//! Vulkan queue: submission, frame acquisition, presentation, and resource retirement.

use std::cell::RefCell;
use std::collections::VecDeque;

use ash::khr::swapchain;
use ash::{Device, vk};
use smallvec::SmallVec;

use super::command::RetainedPipeline;
use super::memory::{SharedBufferPool, VulkanBuffer};
use super::swapchain::VulkanSwapchain;
use super::texture::VulkanTexture;
use crate::error::{RhiError, RhiResult};
use crate::queue::SubmitDesc;
use crate::swapchain::AcquiredImage;
use crate::sync::{TimelineSemaphore, TimelineSemaphoreInner};
use crate::types::MAX_FRAMES_IN_FLIGHT;
use crate::types::{SamplerId, TextureId};

use super::command::VulkanCommandBuffer;
use super::device::{SharedSamplerFreeIds, SharedTextureFreeIds};

/// Vulkan queue wrapper.
pub struct VulkanQueue {
    pub(crate) queue: vk::Queue,
    pub(crate) device: Device,
    pub(crate) swapchain_loader: swapchain::Device,
    pub(crate) command_pool: vk::CommandPool,
    pub(crate) buffer_pool: SharedBufferPool,
    /// Generic command buffers are returned to the pool once their queue timeline value is
    /// complete. Swapchain command buffers have their own frame fences and are intentionally not
    /// placed in this list.
    /// Submitted buffers awaiting their timeline value, each holding the pipelines it bound so
    /// they outlive the GPU work even if the application dropped them.
    pub(crate) pending_commands: RefCell<VecDeque<PendingCommand>>,
    pub(crate) available_commands: RefCell<Vec<vk::CommandBuffer>>,
    pub(crate) completion_semaphore: vk::Semaphore,
    pub(crate) next_completion_value: RefCell<u64>,
    pub(crate) frame_completion_values: RefCell<[u64; MAX_FRAMES_IN_FLIGHT]>,
    /// Whether each frame slot's fence has a submission pending to signal it. A submit that fails
    /// after `reset_fences` would otherwise leave the fence unsignaled forever, and the next
    /// acquire on that slot would block on it for good.
    pub(crate) frame_fence_armed: RefCell<[bool; MAX_FRAMES_IN_FLIGHT]>,
    /// Resources destroyed by the app, each tagged with the timeline value that must retire
    /// before its storage and its bindless slot can be reused. See `release_resource`.
    pub(crate) retired_resources: RefCell<VecDeque<(u64, VulkanRetiredResource)>>,
    pub(crate) free_texture_ids: SharedTextureFreeIds,
    pub(crate) free_sampler_ids: SharedSamplerFreeIds,
}

pub(crate) struct PendingCommand {
    pub(crate) command_buffer: vk::CommandBuffer,
    pub(crate) completion_value: u64,
    /// Held, never read: released when this entry is reclaimed.
    #[allow(dead_code)]
    pub(crate) retained_pipelines: SmallVec<[RetainedPipeline; 4]>,
}

pub(crate) enum VulkanRetiredResource {
    Buffer(VulkanBuffer),
    Texture {
        id: TextureId,
        texture: VulkanTexture,
    },
    /// No Vulkan object to free: under `VK_EXT_descriptor_heap` a sampler is only a descriptor
    /// in the heap, so retirement exists purely to keep an in-flight slot from being reused.
    Sampler { id: SamplerId },
}

impl VulkanQueue {
    pub(crate) fn acquire_command_buffer(&self) -> RhiResult<vk::CommandBuffer> {
        if let Some(command_buffer) = self.available_commands.borrow_mut().pop() {
            unsafe {
                self.device
                    .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                    .map_err(|e| RhiError::CommandBuffer(format!("Reset command buffer: {e}")))?;
            }
            return Ok(command_buffer);
        }

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY);
        unsafe {
            self.device
                .allocate_command_buffers(&alloc_info)
                .map(|buffers| buffers[0])
                .map_err(|e| RhiError::CommandBuffer(e.to_string()))
        }
    }

    pub(crate) fn recycle_command_buffer(&self, command_buffer: vk::CommandBuffer) {
        self.available_commands.borrow_mut().push(command_buffer);
    }

    /// Release a destroyed resource's storage immediately: destroy the native handles, hand any
    /// bindless ID back to the free list, and return the buffer range to the pool.
    ///
    /// Hold a resource until every submission issued so far has retired.
    pub(crate) fn release_resource(&self, resource: VulkanRetiredResource) {
        let pending_until = *self.next_completion_value.borrow();
        if pending_until == 0 {
            self.free_resource(resource);
            return;
        }
        self.retired_resources
            .borrow_mut()
            .push_back((pending_until, resource));
    }

    /// Values are pushed in issue order, so a prefix drain is enough.
    fn collect_retired(&self, completed: u64) {
        loop {
            let Some(resource) = ({
                let mut retired = self.retired_resources.borrow_mut();
                match retired.front() {
                    Some((value, _)) if *value <= completed => {
                        retired.pop_front().map(|(_, resource)| resource)
                    }
                    _ => None,
                }
            }) else {
                return;
            };
            self.free_resource(resource);
        }
    }

    fn free_resource(&self, resource: VulkanRetiredResource) {
        unsafe {
            match &resource {
                // The block owns the mapping and the memory; the buffer only returns its range.
                // Nothing to destroy: the range simply returns to its block.
                VulkanRetiredResource::Buffer(buffer) => {
                    self.buffer_pool.borrow_mut().release(
                        buffer.block_index,
                        buffer.block_offset,
                        buffer.block_range,
                    );
                }
                VulkanRetiredResource::Texture { texture, .. } => {
                    self.device.destroy_image_view(texture.image_view, None);
                    if !texture.is_view {
                        self.device.destroy_image(texture.image, None);
                    }
                }
                // Descriptor-only; the slot is recycled below.
                VulkanRetiredResource::Sampler { .. } => {}
            }
        }
        match resource {
            VulkanRetiredResource::Texture { id, .. } => {
                self.free_texture_ids.borrow_mut().push(id);
            }
            VulkanRetiredResource::Sampler { id, .. } => {
                self.free_sampler_ids.borrow_mut().push(id);
            }
            VulkanRetiredResource::Buffer(_) => {}
        }
    }

    fn completed_submission_value(&self) -> u64 {
        // Reading the counter only fails on a lost device. Reporting 0 instead would silently
        // stall every reclamation path, turning that into an unexplained leak.
        unsafe {
            self.device
                .get_semaphore_counter_value(self.completion_semaphore)
                .expect("read Vulkan completion timeline")
        }
    }

    fn next_completion_value(&self) -> RhiResult<u64> {
        let mut next = self.next_completion_value.borrow_mut();
        *next = next
            .checked_add(1)
            .ok_or_else(|| RhiError::QueueSubmit("Vulkan completion timeline exhausted".into()))?;
        Ok(*next)
    }

    pub fn submit_with_desc(
        &self,
        mut cmd: VulkanCommandBuffer,
        desc: &SubmitDesc<'_>,
    ) -> RhiResult<()> {
        self.reclaim_completed_commands();
        if let Err(error) = cmd.finish() {
            unsafe {
                self.device
                    .free_command_buffers(self.command_pool, &[cmd.command_buffer]);
            }
            return Err(error);
        }
        let waits = timeline_waits(desc.wait_semaphores);
        let mut signals = timeline_waits(desc.signal_semaphores);
        let completion_value = self.next_completion_value()?;
        signals.push(semaphore_submit(
            self.completion_semaphore,
            completion_value,
        ));
        if let Err(err) =
            self.submit_timeline(cmd.command_buffer, &waits, &signals, vk::Fence::null())
        {
            unsafe {
                self.device
                    .free_command_buffers(self.command_pool, &[cmd.command_buffer]);
            }
            return Err(err);
        }
        self.pending_commands.borrow_mut().push_back(PendingCommand {
            command_buffer: cmd.command_buffer,
            completion_value,
            retained_pipelines: std::mem::take(&mut cmd.retained_pipelines),
        });
        Ok(())
    }

    /// Reclaim completed generic command buffers without waiting. Submission stays asynchronous;
    /// the next submission pays one timeline-counter query and then reclaims a prefix of work.
    fn reclaim_completed_commands(&self) {
        let idle =
            self.pending_commands.borrow().is_empty() && self.retired_resources.borrow().is_empty();
        if idle {
            return;
        }
        let completed = self.completed_submission_value();
        self.collect_retired(completed);
        let mut pending = self.pending_commands.borrow_mut();
        while pending
            .front()
            .is_some_and(|entry| entry.completion_value <= completed)
        {
            let entry = pending
                .pop_front()
                .expect("pending command queue front disappeared");
            // Dropping `entry` releases this submission's hold on its pipelines.
            self.recycle_command_buffer(entry.command_buffer);
        }
    }

    /// Encode a `vkQueueSubmit2`. Each `SemaphoreSubmitInfo` carries its own value and stage
    /// mask, so binary and timeline semaphores go through one array with no `pNext` chain.
    /// Pass `vk::Fence::null()` when no completion fence is needed.
    fn submit_timeline(
        &self,
        cmd: vk::CommandBuffer,
        waits: &[vk::SemaphoreSubmitInfo<'_>],
        signals: &[vk::SemaphoreSubmitInfo<'_>],
        fence: vk::Fence,
    ) -> RhiResult<()> {
        let command_buffers = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
        let submit_info = vk::SubmitInfo2::default()
            .wait_semaphore_infos(waits)
            .command_buffer_infos(&command_buffers)
            .signal_semaphore_infos(signals);
        unsafe {
            self.device
                .queue_submit2(self.queue, &[submit_info], fence)
                .map_err(|e| RhiError::QueueSubmit(e.to_string()))?;
        }
        Ok(())
    }

    pub fn acquire_image(
        &self,
        sc: &VulkanSwapchain,
        frame_index: usize,
    ) -> RhiResult<AcquiredImage> {
        let armed = {
            let mut flags = self.frame_fence_armed.borrow_mut();
            std::mem::replace(&mut flags[frame_index], false)
        };
        unsafe {
            let fence = sc.in_flight_fences[frame_index];
            if armed {
                self.device
                    .wait_for_fences(&[fence], true, u64::MAX)
                    .map_err(|e| RhiError::SyncError(e.to_string()))?;
            }
            self.reclaim_completed_commands();

            {
                let mut cmd_buffers = sc.in_flight_cmd_buffers.borrow_mut();
                if let Some(prev) = cmd_buffers.get_mut(frame_index)
                    && *prev != vk::CommandBuffer::null()
                {
                    self.recycle_command_buffer(*prev);
                    *prev = vk::CommandBuffer::null();
                }
            }

            let semaphore = sc.present_complete_semaphores[frame_index];
            let mut acquired = sc.acquired_images.borrow_mut();
            let image_index = match acquired[frame_index] {
                Some(index) => index,
                None => {
                    let (index, _) = self
                        .swapchain_loader
                        .acquire_next_image(sc.swapchain, u64::MAX, semaphore, vk::Fence::null())
                        .map_err(|e| match e {
                            vk::Result::ERROR_OUT_OF_DATE_KHR => RhiError::SwapchainOutOfDate,
                            _ => RhiError::SwapchainCreation(e.to_string()),
                        })?;
                    acquired[frame_index] = Some(index);
                    index
                }
            };

            Ok(AcquiredImage {
                index: image_index,
                format: sc.format,
                width: sc.extent.width,
                height: sc.extent.height,
            })
        }
    }

    pub fn present(
        &self,
        sc: &VulkanSwapchain,
        image_index: u32,
        _frame_index: usize,
    ) -> RhiResult<()> {
        let image_index_usize = image_index as usize;
        let Some(&wait_semaphore) = sc.rendering_complete_semaphores.get(image_index_usize) else {
            return Err(RhiError::PresentFailed("invalid Vulkan image index".into()));
        };
        let wait_semaphores = [wait_semaphore];
        let swapchains = [sc.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe {
            self.swapchain_loader
                .queue_present(self.queue, &present_info)
                .map_err(|e| match e {
                    vk::Result::ERROR_OUT_OF_DATE_KHR => RhiError::SwapchainOutOfDate,
                    _ => RhiError::PresentFailed(e.to_string()),
                })?;
        }
        Ok(())
    }

    pub fn submit_frame(
        &self,
        mut cmd: super::command::VulkanCommandBuffer,
        sc: &super::swapchain::VulkanSwapchain,
        frame_index: usize,
        image_index: u32,
    ) -> RhiResult<()> {
        // Acquire waits gate color writes; timeline waits cover all commands.
        let image_index = image_index as usize;
        if sc.acquired_images.borrow()[frame_index] != Some(image_index as u32) {
            return Err(RhiError::QueueSubmit(
                "frame does not own this acquired image".into(),
            ));
        }
        if let Err(error) = cmd.finish() {
            unsafe {
                self.device
                    .free_command_buffers(self.command_pool, &[cmd.command_buffer]);
            }
            return Err(error);
        }
        let waits: SmallVec<[vk::SemaphoreSubmitInfo<'_>; 4]> = SmallVec::from_elem(
            semaphore_submit(sc.present_complete_semaphores[frame_index], 0)
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
            1,
        );
        let mut signals: SmallVec<[vk::SemaphoreSubmitInfo<'_>; 4]> = SmallVec::new();
        signals.push(semaphore_submit(
            sc.rendering_complete_semaphores[image_index],
            0,
        ));
        let completion_value = self.next_completion_value()?;
        signals.push(semaphore_submit(
            self.completion_semaphore,
            completion_value,
        ));
        let fence = sc.in_flight_fences[frame_index];
        let raw_cmd = cmd.command_buffer;
        // Reset late: a frame that never submits would otherwise wedge this slot's next acquire.
        unsafe {
            self.device
                .reset_fences(&[fence])
                .map_err(|e| RhiError::SyncError(e.to_string()))?;
        }
        self.submit_timeline(raw_cmd, &waits, &signals, fence)?;
        self.frame_fence_armed.borrow_mut()[frame_index] = true;
        self.frame_completion_values.borrow_mut()[frame_index] = completion_value;
        sc.acquired_images.borrow_mut()[frame_index] = None;

        if let Some(slot) = sc.in_flight_cmd_buffers.borrow_mut().get_mut(frame_index) {
            *slot = raw_cmd;
        }

        // `submit_frame` owns presentation on both backends. A stale swapchain is rebuilt on the
        // next acquire.
        match self.present(sc, image_index as u32, frame_index) {
            Ok(()) | Err(RhiError::SwapchainOutOfDate) => Ok(()),
            Err(e) => Err(e),
        }
    }

    pub fn wait_for_frame(&self, frame_index: usize) {
        let value = self.frame_completion_values.borrow()[frame_index];
        if value == 0 {
            return;
        }
        let semaphores = [self.completion_semaphore];
        let values = [value];
        let wait = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        unsafe { self.device.wait_semaphores(&wait, u64::MAX) }
            .expect("Failed to wait for Vulkan frame completion");
    }

    pub fn wait_idle(&self) {
        unsafe {
            self.device
                .queue_wait_idle(self.queue)
                .expect("Failed to wait for Vulkan queue idle");
        }
        self.reclaim_completed_commands();
        // Blocks are only freed here, never on the destroy path, so this cannot land mid-frame.
        self.buffer_pool.borrow_mut().trim(&self.device);
    }
}

/// One wait or signal entry. `value` is ignored for binary semaphores.
fn semaphore_submit<'a>(semaphore: vk::Semaphore, value: u64) -> vk::SemaphoreSubmitInfo<'a> {
    vk::SemaphoreSubmitInfo::default()
        .semaphore(semaphore)
        .value(value)
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
}

fn timeline_waits<'a>(
    pairs: &[(TimelineSemaphore, u64)],
) -> SmallVec<[vk::SemaphoreSubmitInfo<'a>; 4]> {
    pairs
        .iter()
        .map(|(sem, value)| {
            semaphore_submit(
                backend_expect!(&sem.inner, TimelineSemaphoreInner::Vulkan).semaphore,
                *value,
            )
        })
        .collect()
}
