//! Metal queue: submission, frame pacing, presentation, and resource retirement.
use super::as_allocation;
use super::command::MetalCommandBuffer;
use super::device::{
    FrameFenceValues, InFlightFrameCommands, MetalRetiredResource, MetalShared, PendingSubmissions,
    shared_event_as_event,
};
use super::swapchain::MetalSwapchain;
use crate::backend::retire::RetirementQueue;
use crate::error::{RhiError, RhiResult};
use crate::queue::SubmitDesc;
use crate::types::MAX_FRAMES_IN_FLIGHT;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTL4CommandBuffer, MTL4CommandQueue, MTLDrawable, MTLResidencySet, MTLSharedEvent,
};
use std::cell::Cell;
use std::ptr::NonNull;
use std::rc::Rc;

pub struct MetalQueue {
    pub(crate) queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    pub(crate) shared: Rc<MetalShared>,
    pub(crate) frame_fence_values: FrameFenceValues,
    pub(crate) frame_fence_next: Cell<u64>,
    pub(crate) frame_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    pub(crate) in_flight_frame_commands: InFlightFrameCommands,
    pub(crate) pending_submissions: PendingSubmissions,
    /// Resources destroyed by the app, each tagged with the fence value that must retire before
    /// its storage and its bindless slot can be reused.
    pub(crate) retired_resources: RetirementQueue<MetalRetiredResource>,
}

impl MetalQueue {
    /// Hold a resource until every submission issued so far has retired.
    pub(crate) fn release_resource(&self, resource: MetalRetiredResource) {
        self.retired_resources
            .release(self.frame_fence_next.get(), resource, |resource| {
                self.free_resource(resource)
            });
    }

    fn collect_retired(&self, completed: u64) {
        self.retired_resources
            .collect(completed, |resource| self.free_resource(resource));
    }

    fn free_resource(&self, resource: MetalRetiredResource) {
        match &resource {
            // Covered by its heap's residency entry; nothing to remove.
            MetalRetiredResource::Buffer(_) => {}
            MetalRetiredResource::Texture {
                texture,
                is_view: true,
                ..
            } => {
                self.shared
                    .residency_set
                    .removeAllocation(as_allocation(texture));
                self.shared.residency_dirty.set(true);
            }
            // Placed textures are covered by their heap's residency entry.
            MetalRetiredResource::Texture { .. } => {}
            // Samplers aren't MTLAllocations, so they never entered the residency set.
            MetalRetiredResource::Sampler { .. } => {}
            MetalRetiredResource::Accel(accel) => accel.release_residency(),
        }
        match resource {
            MetalRetiredResource::Texture { id, .. } => self.shared.textures.recycle(id.0),
            MetalRetiredResource::Sampler { id, sampler } => {
                // The submissions that could read this slot have retired, so the deferral ends.
                drop(sampler);
                self.shared.samplers.recycle(id.0);
            }
            MetalRetiredResource::Buffer(buffer) => buffer.release_to_pool(),
            // Dropping the last `Retained` handles frees the structure and its scratch.
            MetalRetiredResource::Accel(_) => {}
        }
    }

    pub fn submit_with_desc(
        &self,
        cmd: MetalCommandBuffer,
        desc: &SubmitDesc<'_>,
    ) -> RhiResult<()> {
        if self.shared.residency_dirty.replace(false) {
            self.shared.residency_set.commit();
        }
        self.reclaim_completed_submissions();

        for (semaphore, value) in desc.wait_semaphores {
            let event = &semaphore.inner;
            self.queue
                .waitForEvent_value(shared_event_as_event(&event.event), *value);
        }

        let mut cmd = cmd;
        cmd.finish();
        self.commit_single(&cmd.command_buffer);

        for (semaphore, value) in desc.signal_semaphores {
            let event = &semaphore.inner;
            self.queue
                .signalEvent_value(shared_event_as_event(&event.event), *value);
        }

        let value = self.next_fence_value();
        self.queue
            .signalEvent_value(shared_event_as_event(&self.frame_event), value);
        self.pending_submissions
            .borrow_mut()
            .push_back((value, cmd));
        Ok(())
    }

    pub fn submit_frame(
        &self,
        cmd: MetalCommandBuffer,
        sc: &MetalSwapchain,
        frame_index: usize,
        _image_index: u32,
    ) -> RhiResult<()> {
        if frame_index >= MAX_FRAMES_IN_FLIGHT {
            return Err(RhiError::Backend(
                "invalid frame index for Metal queue submission".into(),
            ));
        }
        if self.shared.residency_dirty.replace(false) {
            self.shared.residency_set.commit();
        }
        self.reclaim_completed_submissions();

        // Taken during recording, so the queue-side wait is enqueued here — still before the
        // commit that renders into it. A frame that never touched the swapchain skips this.
        let drawable = sc.drawable.current();
        if let Some(drawable) = drawable.as_ref() {
            self.queue.waitForDrawable(drawable);
        }

        let mut cmd = cmd;
        cmd.finish();
        self.commit_single(&cmd.command_buffer);

        let value = self.next_fence_value();
        self.frame_fence_values.borrow_mut()[frame_index] = value;
        self.queue
            .signalEvent_value(shared_event_as_event(&self.frame_event), value);

        if let Some(drawable) = drawable {
            self.queue.signalDrawable(&drawable);
            drawable.present();
        }
        sc.drawable.release();

        self.in_flight_frame_commands.borrow_mut()[frame_index] = Some(cmd);

        Ok(())
    }

    pub fn acquire_image(&self, _sc: &MetalSwapchain, frame_index: usize) -> RhiResult<u32> {
        self.reclaim_completed_submissions();

        let value = self.frame_fence_values.borrow()[frame_index];
        if value != 0 {
            assert!(
                self.frame_event
                    .waitUntilSignaledValue_timeoutMS(value, u64::MAX),
                "Failed to wait for Metal frame completion"
            );
        }
        self.in_flight_frame_commands.borrow_mut()[frame_index] = None;

        // No `nextDrawable` here: the drawable stays in the pool until the frame encodes into
        // it. See `MetalDrawableSlot`. Metal has one current drawable, so the index is always 0.
        Ok(0)
    }

    pub fn wait_idle(&self) {
        let value = self.next_fence_value();
        self.queue
            .signalEvent_value(shared_event_as_event(&self.frame_event), value);
        assert!(
            self.frame_event
                .waitUntilSignaledValue_timeoutMS(value, u64::MAX),
            "Failed to wait for Metal queue idle"
        );
        self.pending_submissions.borrow_mut().clear();
        for slot in self.in_flight_frame_commands.borrow_mut().iter_mut() {
            *slot = None;
        }
        self.collect_retired(value);
    }

    fn next_fence_value(&self) -> u64 {
        let value = self.frame_fence_next.get().wrapping_add(1);
        self.frame_fence_next.set(value);
        value
    }

    pub(crate) fn reclaim_completed_submissions(&self) {
        if self.pending_submissions.borrow().is_empty() && self.retired_resources.is_empty() {
            return;
        }
        let completed = self.frame_event.signaledValue();
        {
            let mut pending = self.pending_submissions.borrow_mut();
            while pending
                .front()
                .is_some_and(|(value, _)| *value <= completed)
            {
                pending.pop_front();
            }
        }
        self.collect_retired(completed);
    }

    /// Commit one command buffer. `commit_count` is the batching entry point, but the RHI hands
    /// buffers to the queue one at a time, so there is never more than one to pass.
    fn commit_single(&self, cmd: &Retained<ProtocolObject<dyn MTL4CommandBuffer>>) {
        let mut buffers: [NonNull<ProtocolObject<dyn MTL4CommandBuffer>>; 1] =
            [NonNull::from(cmd.as_ref())];
        // SAFETY: `buffers` is a live array of exactly one non-null command-buffer pointer, and
        // its length is what is passed alongside it.
        unsafe {
            self.queue
                .commit_count(NonNull::from(&mut buffers).cast(), buffers.len());
        }
    }
}
