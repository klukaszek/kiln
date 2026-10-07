//! Vulkan queue: submission, frame acquisition, presentation, and resource retirement.

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::rc::Rc;

use ash::khr::swapchain;
use ash::{Device, vk};
use smallvec::SmallVec;

use super::accel::VulkanAccelerationStructure;
use super::command::VulkanCommandBuffer;
use super::device::{SharedSamplerIds, SharedTextures};
use super::memory::{SharedBufferPool, VulkanBuffer};
use super::pipeline::VulkanPipeline;
use super::swapchain::VulkanSwapchain;
use super::texture::VulkanTexture;
use crate::backend::retire::RetirementQueue;
use crate::error::{RhiError, RhiResult};
use crate::queue::SubmitDesc;
use crate::sync::TimelineSemaphore;
use crate::types::{MAX_FRAMES_IN_FLIGHT, SamplerId, TextureId};

/// Vulkan queue wrapper.
pub struct VulkanQueue {
    pub(crate) queue: vk::Queue,
    pub(crate) device: Device,
    pub(crate) swapchain_loader: swapchain::Device,
    pub(crate) command_pool: vk::CommandPool,
    pub(crate) buffer_pool: SharedBufferPool,
    /// Submitted command buffers awaiting their timeline value, each holding the pipelines it
    /// bound so they outlive the GPU work even if the application dropped them. Entries are
    /// returned to `available_commands` once their value is complete. Swapchain command buffers
    /// have their own frame fences and are intentionally not placed in this list.
    pub(crate) pending_commands: RefCell<VecDeque<PendingCommand>>,
    pub(crate) available_commands: RefCell<Vec<vk::CommandBuffer>>,
    pub(crate) completion_semaphore: vk::Semaphore,
    pub(crate) next_completion_value: Cell<u64>,
    /// The last frame submission on each in-flight slot. Lives here rather than on the swapchain
    /// because a slot outlives the swapchain it last presented to.
    pub(crate) frames: RefCell<[FrameSubmission; MAX_FRAMES_IN_FLIGHT]>,
    /// Resources destroyed by the app, each tagged with the timeline value that must retire
    /// before its storage and its bindless slot can be reused.
    pub(crate) retired_resources: RetirementQueue<VulkanRetiredResource>,
    pub(crate) textures: SharedTextures,
    pub(crate) samplers: SharedSamplerIds,
}

/// What the queue remembers about one frame slot's last submission.
#[derive(Clone, Copy, Default)]
pub(crate) struct FrameSubmission {
    /// Completion timeline value; 0 means the slot has never been submitted.
    pub(crate) value: u64,
    /// Whether the slot's fence has a submission pending to signal it. A submit that fails after
    /// `reset_fences` would otherwise leave the fence unsignaled forever, and the next acquire on
    /// that slot would block on it for good.
    pub(crate) armed: bool,
}

pub(crate) struct PendingCommand {
    pub(crate) command_buffer: vk::CommandBuffer,
    pub(crate) completion_value: u64,
    /// Pipelines the application may have dropped while this submission still bound them.
    /// Released by [`VulkanQueue::reclaim_completed_commands`] once the submission retires.
    pub(crate) retained_pipelines: SmallVec<[Rc<VulkanPipeline>; 4]>,
}

pub(crate) enum VulkanRetiredResource {
    Buffer(VulkanBuffer),
    /// The `VkAccelerationStructureKHR` plus its backing and scratch ranges. Destroying any of
    /// them while a frame is still tracing against the structure is a use-after-free.
    Accel(Box<VulkanAccelerationStructure>),
    Texture {
        id: TextureId,
        texture: VulkanTexture,
    },
    /// No Vulkan object to free: under `VK_EXT_descriptor_heap` a sampler is only a descriptor
    /// in the heap, so retirement exists purely to keep an in-flight slot from being reused.
    Sampler {
        id: SamplerId,
    },
}

impl VulkanQueue {
    pub(crate) fn acquire_command_buffer(&self) -> RhiResult<vk::CommandBuffer> {
        if let Some(command_buffer) = self.available_commands.borrow_mut().pop() {
            unsafe {
                self.device
                    .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                    .map_err(|e| {
                        RhiError::CommandBuffer(format!("Reset command buffer: {e}").into())
                    })?;
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
                .map_err(|e| RhiError::CommandBuffer(e.into()))
        }
    }

    pub(crate) fn recycle_command_buffer(&self, command_buffer: vk::CommandBuffer) {
        self.available_commands.borrow_mut().push(command_buffer);
    }

    /// Hold a destroyed resource until every submission issued so far has retired, then free it:
    /// destroy the native handles, hand any bindless ID back to the free list, and return the
    /// buffer range to the pool. With nothing submitted yet, it is freed immediately.
    pub(crate) fn release_resource(&self, resource: VulkanRetiredResource) {
        self.retired_resources
            .release(self.next_completion_value.get(), resource, |resource| {
                self.free_resource(resource)
            });
    }

    fn collect_retired(&self, completed: u64) {
        self.retired_resources
            .collect(completed, |resource| self.free_resource(resource));
    }

    fn free_resource(&self, resource: VulkanRetiredResource) {
        match &resource {
            // The block owns the memory and its mapping; the buffer only returns its range.
            VulkanRetiredResource::Buffer(buffer) => {
                self.buffer_pool.borrow_mut().release(
                    buffer.block_index,
                    buffer.block_offset,
                    buffer.block_range,
                );
            }
            VulkanRetiredResource::Texture { texture, .. } => {
                // SAFETY: reached only once `completed` passed this entry's timeline value, so
                // no submission still references the image or its view.
                unsafe {
                    self.device.destroy_image_view(texture.image_view, None);
                    if !texture.is_view {
                        self.device.destroy_image(texture.image, None);
                    }
                }
            }
            // Descriptor-only; the slot is recycled below.
            VulkanRetiredResource::Sampler { .. } => {}
            // SAFETY: as above — every submission that could be tracing this has retired.
            VulkanRetiredResource::Accel(accel) => unsafe { accel.destroy() },
        }
        match resource {
            VulkanRetiredResource::Texture { id, .. } => {
                self.textures.recycle(id.0);
            }
            VulkanRetiredResource::Sampler { id, .. } => {
                self.samplers.recycle(id.0);
            }
            VulkanRetiredResource::Buffer(_) | VulkanRetiredResource::Accel(_) => {}
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
        let next = self
            .next_completion_value
            .get()
            .checked_add(1)
            .ok_or_else(|| RhiError::QueueSubmit("Vulkan completion timeline exhausted".into()))?;
        self.next_completion_value.set(next);
        Ok(next)
    }

    /// Block until the completion timeline reaches `value`.
    fn wait_for_completion(&self, value: u64) -> Result<(), vk::Result> {
        let semaphores = [self.completion_semaphore];
        let values = [value];
        let wait = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        // SAFETY: the semaphore is this device's, and `value` was handed to a submission.
        unsafe { self.device.wait_semaphores(&wait, u64::MAX) }
    }

    /// Free a command buffer that failed before it reached the queue.
    fn discard(&self, command_buffer: vk::CommandBuffer) {
        // SAFETY: it never reached the queue, so nothing references it.
        unsafe {
            self.device
                .free_command_buffers(self.command_pool, &[command_buffer])
        };
    }

    pub fn submit_with_desc(
        &self,
        mut cmd: VulkanCommandBuffer,
        desc: &SubmitDesc<'_>,
    ) -> RhiResult<()> {
        self.reclaim_completed_commands();
        if let Err(error) = cmd.finish() {
            self.discard(cmd.command_buffer);
            return Err(error);
        }
        let waits = timeline_entries(desc.wait_semaphores);
        let mut signals = timeline_entries(desc.signal_semaphores);
        let completion_value = self.next_completion_value()?;
        signals.push(semaphore_submit(
            self.completion_semaphore,
            completion_value,
        ));
        if let Err(err) =
            self.submit_timeline(cmd.command_buffer, &waits, &signals, vk::Fence::null())
        {
            self.discard(cmd.command_buffer);
            return Err(err);
        }
        self.pending_commands
            .borrow_mut()
            .push_back(PendingCommand {
                command_buffer: cmd.command_buffer,
                completion_value,
                retained_pipelines: std::mem::take(&mut cmd.retained_pipelines),
            });
        Ok(())
    }

    /// Reclaim completed generic command buffers without waiting. Submission stays asynchronous;
    /// the next submission pays one timeline-counter query and then reclaims a prefix of work.
    fn reclaim_completed_commands(&self) {
        let idle = self.pending_commands.borrow().is_empty() && self.retired_resources.is_empty();
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
            let PendingCommand {
                command_buffer,
                retained_pipelines,
                ..
            } = pending
                .pop_front()
                .expect("pending command queue front disappeared");
            // The submission has retired, so this is where its hold on the pipelines ends.
            drop(retained_pipelines);
            self.recycle_command_buffer(command_buffer);
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
                .map_err(|e| RhiError::QueueSubmit(e.into()))?;
        }
        Ok(())
    }

    pub fn acquire_image(&self, sc: &VulkanSwapchain, frame_index: usize) -> RhiResult<u32> {
        let armed = std::mem::take(&mut self.frames.borrow_mut()[frame_index].armed);
        let fence = sc.frames.borrow()[frame_index].fence;
        if armed {
            // SAFETY: `fence` belongs to this device and is signalled by the submission that
            // armed it.
            unsafe { self.device.wait_for_fences(&[fence], true, u64::MAX) }
                .map_err(|e| RhiError::SyncError(e.into()))?;
        }
        self.reclaim_completed_commands();

        let mut frames = sc.frames.borrow_mut();
        let slot = &mut frames[frame_index];
        let previous = std::mem::replace(&mut slot.command_buffer, vk::CommandBuffer::null());
        if previous != vk::CommandBuffer::null() {
            self.recycle_command_buffer(previous);
        }

        let semaphore = slot.present_complete;
        let image_index = match slot.acquired_image {
            Some(index) => index,
            None => {
                // SAFETY: the swapchain and semaphore are this device's, and the frame slot's
                // previous acquisition has completed (the fence wait above).
                let (index, _) = unsafe {
                    self.swapchain_loader.acquire_next_image(
                        sc.swapchain,
                        u64::MAX,
                        semaphore,
                        vk::Fence::null(),
                    )
                }
                .map_err(|e| match e {
                    vk::Result::ERROR_OUT_OF_DATE_KHR => RhiError::SwapchainOutOfDate,
                    _ => RhiError::SwapchainCreation(e.into()),
                })?;
                slot.acquired_image = Some(index);
                index
            }
        };

        Ok(image_index)
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
                    _ => RhiError::PresentFailed(e.into()),
                })?;
        }
        Ok(())
    }

    pub fn submit_frame(
        &self,
        mut cmd: VulkanCommandBuffer,
        sc: &VulkanSwapchain,
        frame_index: usize,
        image_index: u32,
    ) -> RhiResult<()> {
        // Acquire waits gate color writes; timeline waits cover all commands.
        let image_index = image_index as usize;
        if sc.frames.borrow()[frame_index].acquired_image != Some(image_index as u32) {
            return Err(RhiError::QueueSubmit(
                "frame does not own this acquired image".into(),
            ));
        }
        if let Err(error) = cmd.finish() {
            self.discard(cmd.command_buffer);
            return Err(error);
        }
        let (present_complete, fence) = {
            let slot = sc.frames.borrow()[frame_index];
            (slot.present_complete, slot.fence)
        };
        let waits: SmallVec<[vk::SemaphoreSubmitInfo<'_>; 4]> = SmallVec::from_elem(
            semaphore_submit(present_complete, 0)
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
        let raw_cmd = cmd.command_buffer;
        // Reset late: a frame that never submits would otherwise wedge this slot's next acquire.
        unsafe {
            self.device
                .reset_fences(&[fence])
                .map_err(|e| RhiError::SyncError(e.into()))?;
        }
        self.submit_timeline(raw_cmd, &waits, &signals, fence)?;
        self.frames.borrow_mut()[frame_index] = FrameSubmission {
            value: completion_value,
            armed: true,
        };
        {
            let slot = &mut sc.frames.borrow_mut()[frame_index];
            slot.acquired_image = None;
            slot.command_buffer = raw_cmd;
        }

        // `submit_frame` owns presentation on both backends. A stale swapchain is rebuilt on the
        // next acquire.
        match self.present(sc, image_index as u32, frame_index) {
            Ok(()) | Err(RhiError::SwapchainOutOfDate) => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// Submit the device's setup command buffer and block until it has completed.
    ///
    /// Goes through the shared completion timeline so the wait covers this submission alone.
    pub(crate) fn submit_setup_and_wait(&self, cmd: vk::CommandBuffer) -> RhiResult<()> {
        let completion_value = self.next_completion_value()?;
        let signals = [semaphore_submit(
            self.completion_semaphore,
            completion_value,
        )];
        self.submit_timeline(cmd, &[], &signals, vk::Fence::null())?;
        self.wait_for_completion(completion_value)
            .map_err(|e| RhiError::SyncError(e.into()))
    }

    pub fn wait_for_frame(&self, frame_index: usize) {
        let value = self.frames.borrow()[frame_index].value;
        if value != 0 {
            self.wait_for_completion(value)
                .expect("Failed to wait for Vulkan frame completion");
        }
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

fn timeline_entries<'a>(
    pairs: &[(&TimelineSemaphore, u64)],
) -> SmallVec<[vk::SemaphoreSubmitInfo<'a>; 4]> {
    pairs
        .iter()
        .map(|(sem, value)| semaphore_submit(sem.inner.semaphore, *value))
        .collect()
}
