//! Vulkan queue: submission, frame acquisition, presentation, and resource retirement.

use std::cell::RefCell;
use std::collections::VecDeque;

use ash::khr::swapchain;
use ash::{Device, vk};
use smallvec::SmallVec;

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
    pub(crate) pending_commands: RefCell<VecDeque<(vk::CommandBuffer, u64)>>,
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

pub(crate) enum VulkanRetiredResource {
    Buffer(VulkanBuffer),
    Texture {
        id: TextureId,
        texture: VulkanTexture,
    },
    Sampler {
        id: SamplerId,
        sampler: vk::Sampler,
    },
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
                VulkanRetiredResource::Buffer(buffer) => {
                    self.device.destroy_buffer(buffer.buffer, None);
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
                VulkanRetiredResource::Sampler { sampler, .. } => {
                    self.device.destroy_sampler(*sampler, None);
                }
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
        unsafe {
            self.device
                .get_semaphore_counter_value(self.completion_semaphore)
                .unwrap_or(0)
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
        let waits: SmallVec<[(vk::Semaphore, u64); 4]> = timeline_pairs(desc.wait_semaphores);
        let mut signals: SmallVec<[(vk::Semaphore, u64); 4]> =
            timeline_pairs(desc.signal_semaphores);
        let wait_stages: SmallVec<[vk::PipelineStageFlags; 4]> =
            SmallVec::from_elem(vk::PipelineStageFlags::ALL_COMMANDS, waits.len());
        let completion_value = self.next_completion_value()?;
        signals.push((self.completion_semaphore, completion_value));
        if let Err(err) = self.submit_timeline(
            cmd.command_buffer,
            &waits,
            &wait_stages,
            &signals,
            vk::Fence::null(),
        ) {
            unsafe {
                self.device
                    .free_command_buffers(self.command_pool, &[cmd.command_buffer]);
            }
            return Err(err);
        }
        self.pending_commands
            .borrow_mut()
            .push_back((cmd.command_buffer, completion_value));
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
            .is_some_and(|(_, value)| *value <= completed)
        {
            let (command_buffer, _) = pending
                .pop_front()
                .expect("pending command queue front disappeared");
            self.recycle_command_buffer(command_buffer);
        }
    }

    /// Encode a `vkQueueSubmit` with timeline-semaphore wait/signal pairs.
    ///
    /// `wait_stages` must have the same length as `waits`. Pass `vk::Fence::null()` when
    /// no completion fence is needed.
    fn submit_timeline(
        &self,
        cmd: vk::CommandBuffer,
        waits: &[(vk::Semaphore, u64)],
        wait_stages: &[vk::PipelineStageFlags],
        signals: &[(vk::Semaphore, u64)],
        fence: vk::Fence,
    ) -> RhiResult<()> {
        let command_buffers = [cmd];
        let mut wait_semaphores = SmallVec::<[vk::Semaphore; 4]>::with_capacity(waits.len());
        let mut wait_values = SmallVec::<[u64; 4]>::with_capacity(waits.len());
        for &(semaphore, value) in waits {
            wait_semaphores.push(semaphore);
            wait_values.push(value);
        }
        let mut signal_semaphores = SmallVec::<[vk::Semaphore; 4]>::with_capacity(signals.len());
        let mut signal_values = SmallVec::<[u64; 4]>::with_capacity(signals.len());
        for &(semaphore, value) in signals {
            signal_semaphores.push(semaphore);
            signal_values.push(value);
        }
        let mut submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);
        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(&signal_values);
        if !wait_values.is_empty() || !signal_values.is_empty() {
            submit_info = submit_info.push_next(&mut timeline_info);
        }
        unsafe {
            self.device
                .queue_submit(self.queue, &[submit_info], fence)
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
        let mut waits: SmallVec<[(vk::Semaphore, u64); 4]> = SmallVec::new();
        waits.push((sc.present_complete_semaphores[frame_index], 0));
        let mut wait_stages: SmallVec<[vk::PipelineStageFlags; 4]> = SmallVec::new();
        wait_stages.push(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT);
        let mut signals: SmallVec<[(vk::Semaphore, u64); 4]> = SmallVec::new();
        signals.push((sc.rendering_complete_semaphores[image_index], 0));
        let completion_value = self.next_completion_value()?;
        signals.push((self.completion_semaphore, completion_value));
        let fence = sc.in_flight_fences[frame_index];
        let raw_cmd = cmd.command_buffer;
        // Reset late: a frame that never submits would otherwise wedge this slot's next acquire.
        unsafe {
            self.device
                .reset_fences(&[fence])
                .map_err(|e| RhiError::SyncError(e.to_string()))?;
        }
        self.submit_timeline(raw_cmd, &waits, &wait_stages, &signals, fence)?;
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

/// Semaphore/value pairs for one submission.
type ValuePairs = SmallVec<[(vk::Semaphore, u64); 4]>;

/// Unwrap a slice of `(TimelineSemaphore, u64)` into `(vk::Semaphore, u64)` pairs.
fn timeline_pairs(pairs: &[(TimelineSemaphore, u64)]) -> ValuePairs {
    pairs
        .iter()
        .map(|(sem, value)| {
            (
                backend_expect!(&sem.inner, TimelineSemaphoreInner::Vulkan).semaphore,
                *value,
            )
        })
        .collect()
}
