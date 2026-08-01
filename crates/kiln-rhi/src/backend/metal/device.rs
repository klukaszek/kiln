use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ptr::NonNull;
use std::rc::Rc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_core_foundation::CGSize;
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4CommandAllocator, MTL4CommandBuffer, MTL4CommandQueue, MTL4Compiler,
    MTL4CompilerDescriptor, MTL4ComputePipelineDescriptor, MTL4CounterHeap,
    MTL4CounterHeapDescriptor, MTL4CounterHeapType, MTL4LibraryFunctionDescriptor,
    MTL4PipelineDescriptor, MTL4PipelineOptions, MTL4ShaderReflection, MTLAllocation, MTLBinding,
    MTLBindingType, MTLBuffer, MTLCompileOptions, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLCullMode, MTLDevice, MTLDrawable, MTLEvent, MTLHeap,
    MTLLanguageVersion, MTLLibrary, MTLPixelFormat, MTLRenderPipelineState, MTLResidencySet,
    MTLResidencySetDescriptor, MTLSamplerDescriptor, MTLSamplerState, MTLSharedEvent,
    MTLStorageMode, MTLTexture, MTLTextureDescriptor, MTLTextureType,
    MTLTextureUsage as MtlTextureUsage, MTLWinding,
};
use objc2_quartz_core::CAMetalLayer;
use raw_window_handle::RawWindowHandle;
use smallvec::SmallVec;

use crate::accel::{AccelInner, AccelerationStructure};
use crate::command::{CommandBuffer, SignalOp, SignalValueDesc, WaitOp, WaitValueDesc};
use crate::device::{BindlessMode, DeviceDesc};
use crate::error::{RhiError, RhiResult};
use crate::memory::{BufferDesc, GpuBuffer, GpuBufferInner};
use crate::pipeline::*;
use crate::query::{QueryPool, QueryPoolInner};
use crate::queue::{Queue, QueueInner, SubmitDesc};
use crate::sampler::{Sampler, SamplerDesc};
use crate::shader::{ShaderModule, ShaderModuleDesc};
use crate::surface::{Surface, SurfaceDesc, SurfaceInner};
use crate::swapchain::{AcquiredImage, Swapchain, SwapchainDesc, SwapchainInner};
use crate::sync::{TimelineSemaphore, TimelineSemaphoreInner};
use crate::texture::{Texture, TextureDesc, TextureSizeAlign, TextureUsage};
use crate::types::*;

use super::command::{
    MetalCommandBuffer, MetalFrameTableSlot, MetalTableState, SharedFrameTableSlots,
    SharedTableSlotPool,
};
use super::memory::{MetalBuffer, MetalBufferPool, SharedMetalBufferPool};
use super::pipeline::{MetalComputePso, MetalGraphicsPso};
use super::query::MetalQueryPool;
use super::shader::MetalShaderModule;
use super::surface::MetalSurface;
use super::swapchain::{MetalDrawableSlot, MetalSwapchain};
use super::sync::MetalTimelineSemaphore;
use super::texture::{format_to_mtl, mtl_to_format};

type FrameFenceValues = Rc<RefCell<[u64; MAX_FRAMES_IN_FLIGHT]>>;
type InFlightFrameCommands = Rc<RefCell<Vec<Option<MetalCommandBuffer>>>>;
type PendingSubmissions = Rc<RefCell<VecDeque<(u64, MetalCommandBuffer)>>>;
pub(crate) type SharedTextures = Rc<RefCell<Vec<Option<Retained<ProtocolObject<dyn MTLTexture>>>>>>;
pub(crate) type SharedSamplers =
    Rc<RefCell<Vec<Option<Retained<ProtocolObject<dyn MTLSamplerState>>>>>>;
type SharedTextureFreeIds = Rc<RefCell<Vec<TextureId>>>;
type SharedSamplerFreeIds = Rc<RefCell<Vec<SamplerId>>>;
/// Buffer allocations keyed by GPU base address, enabling O(log n) address->buffer
/// resolution for blit copies and indirect draws instead of a linear scan.
pub(crate) type SharedAllocations = Rc<RefCell<BTreeMap<u64, BufferAllocation>>>;
/// Reverse index for CPU-mapped allocations, used by the public pointer bridge.
type SharedMappedAllocations = Rc<RefCell<BTreeMap<usize, MappedAllocation>>>;
pub(crate) type SharedBindlessGeneration = Rc<Cell<u64>>;
type ValueSyncMap = RefCell<HashMap<u64, MetalValueSyncState>>;
type MetalEventWaits = SmallVec<[(Retained<ProtocolObject<dyn MTLSharedEvent>>, u64); 4]>;
/// Events being merged, tagged with the value pointer they sync on. Linear merge: a submission
/// carries one or two entries, so a `HashMap` would allocate and hash for nothing.
type MergedEvents = SmallVec<[(u64, Retained<ProtocolObject<dyn MTLSharedEvent>>, u64); 4]>;

/// Record `value` under `key`, folding into any existing entry with `combine`.
fn merge_event(
    merged: &mut MergedEvents,
    key: u64,
    event: &Retained<ProtocolObject<dyn MTLSharedEvent>>,
    value: u64,
    combine: fn(u64, u64) -> u64,
) {
    if let Some((_, _, existing)) = merged.iter_mut().find(|(candidate, ..)| *candidate == key) {
        *existing = combine(*existing, value);
    } else {
        merged.push((key, event.clone(), value));
    }
}

fn finish_merged_events(merged: MergedEvents) -> MetalEventWaits {
    merged
        .into_iter()
        .map(|(_, event, value)| (event, value))
        .collect()
}
type RetiredResources = Rc<RefCell<VecDeque<(u64, MetalRetiredResource)>>>;
type PendingRetiredResources = Rc<RefCell<Vec<MetalRetiredResource>>>;
type SharedPendingRetirementWork = Rc<Cell<bool>>;
pub(crate) type SharedActiveRecordings = Rc<Cell<usize>>;

pub(crate) enum MetalRetiredResource {
    Buffer(MetalBuffer),
    Texture {
        id: TextureId,
        texture: Retained<ProtocolObject<dyn MTLTexture>>,
    },
    Sampler {
        id: SamplerId,
        sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
    },
}

fn shared_event_as_event(
    event: &Retained<ProtocolObject<dyn MTLSharedEvent>>,
) -> &ProtocolObject<dyn MTLEvent> {
    unsafe {
        &*(event.as_ref() as *const ProtocolObject<dyn MTLSharedEvent>
            as *const ProtocolObject<dyn MTLEvent>)
    }
}

// Metal 4 argument tables carry root and bindless-heap buffer addresses.

#[derive(Clone)]
pub(crate) struct BufferAllocation {
    pub base: GpuAddress,
    pub size: u64,
    pub buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub heap: Retained<ProtocolObject<dyn MTLHeap>>,
}

struct MappedAllocation {
    gpu_base: GpuAddress,
    size: u64,
}

fn resolve_mapped_pointer(
    allocations: &BTreeMap<usize, MappedAllocation>,
    ptr: usize,
) -> Option<GpuAddress> {
    let (&base, allocation) = allocations.range(..=ptr).next_back()?;
    let offset = (ptr - base) as u64;
    (offset < allocation.size).then(|| allocation.gpu_base.offset(offset))
}

pub struct MetalDevice {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    buffer_pool: SharedMetalBufferPool,
    rhi_queue: Queue,
    residency_set: Retained<ProtocolObject<dyn MTLResidencySet>>,
    residency_dirty: Rc<Cell<bool>>,
    textures: SharedTextures,
    texture_view_flags: RefCell<Vec<bool>>,
    free_texture_ids: SharedTextureFreeIds,
    samplers: SharedSamplers,
    free_sampler_ids: SharedSamplerFreeIds,
    texture_generation: SharedBindlessGeneration,
    sampler_generation: SharedBindlessGeneration,
    allocations: SharedAllocations,
    mapped_allocations: SharedMappedAllocations,
    active_recordings: SharedActiveRecordings,
    /// Per-frame fence values for swapchain acquisition.
    frame_fence_values: FrameFenceValues,
    /// Shared event for per-frame synchronization.
    frame_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    frame_table_slots: SharedFrameTableSlots,
    /// Free list of table slots for non-swapchain command buffers.
    table_slot_pool: SharedTableSlotPool,
    bindless_mode: BindlessMode,
    /// Monotonic counter for AccelerationStructureId assignment.
    accel_counter: RefCell<u32>,
    mdi_icb_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

pub struct MetalQueue {
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    #[allow(dead_code)]
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    residency_set: Retained<ProtocolObject<dyn MTLResidencySet>>,
    residency_dirty: Rc<Cell<bool>>,
    frame_fence_values: FrameFenceValues,
    frame_fence_next: Rc<Cell<u64>>,
    frame_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    in_flight_frame_commands: InFlightFrameCommands,
    pending_submissions: PendingSubmissions,
    retired_resources: RetiredResources,
    pending_retired_resources: PendingRetiredResources,
    pending_retirement_work: SharedPendingRetirementWork,
    active_recordings: SharedActiveRecordings,
    last_submission_value: Cell<Option<u64>>,
    free_texture_ids: SharedTextureFreeIds,
    free_sampler_ids: SharedSamplerFreeIds,
    value_sync: ValueSyncMap,
}

#[derive(Clone)]
struct MetalValueSyncState {
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    scheduled_value: u64,
}

impl MetalQueue {
    pub fn submit(&self, cmd: MetalCommandBuffer) -> RhiResult<()> {
        self.submit_with_desc(cmd, &SubmitDesc::default())
    }

    pub(crate) fn retire_resource(&self, resource: MetalRetiredResource) {
        self.pending_retired_resources.borrow_mut().push(resource);
        self.pending_retirement_work.set(true);
    }

    fn schedule_pending_retirements(&self) {
        if !self.pending_retirement_work.get() || self.active_recordings.get() != 0 {
            return;
        }
        let pending = std::mem::take(&mut *self.pending_retired_resources.borrow_mut());
        self.pending_retirement_work.set(false);
        if let Some(retire_value) = self.last_submission_value.get() {
            self.retired_resources
                .borrow_mut()
                .extend(pending.into_iter().map(|resource| (retire_value, resource)));
        } else {
            for resource in pending {
                self.release_retired_resource(resource);
            }
        }
    }

    fn release_retired_resource(&self, resource: MetalRetiredResource) {
        match &resource {
            MetalRetiredResource::Buffer(buffer) => {
                let allocation = unsafe {
                    &*(buffer.buffer.as_ref() as *const ProtocolObject<dyn MTLBuffer>
                        as *const ProtocolObject<dyn MTLAllocation>)
                };
                self.residency_set.removeAllocation(allocation);
                self.residency_dirty.set(true);
            }
            MetalRetiredResource::Texture { texture, .. } => {
                let allocation = unsafe {
                    &*(texture.as_ref() as *const ProtocolObject<dyn MTLTexture>
                        as *const ProtocolObject<dyn MTLAllocation>)
                };
                self.residency_set.removeAllocation(allocation);
                self.residency_dirty.set(true);
            }
            // Sampler states are not MTLResource/MTLAllocation objects and do not belong in the
            // residency set. Retain them until the fence completes, then ARC releases them.
            MetalRetiredResource::Sampler { sampler, .. } => {
                let _ = sampler;
            }
        }
        match resource {
            MetalRetiredResource::Texture { id, .. } => {
                self.free_texture_ids.borrow_mut().push(id);
            }
            MetalRetiredResource::Sampler { id, .. } => {
                self.free_sampler_ids.borrow_mut().push(id);
            }
            MetalRetiredResource::Buffer(buffer) => buffer.release_to_pool(),
        }
    }

    fn reclaim_retired_resources(&self) {
        let completed = self.frame_event.signaledValue();
        let mut ready = SmallVec::<[MetalRetiredResource; 4]>::new();
        {
            let mut retired = self.retired_resources.borrow_mut();
            while retired
                .front()
                .is_some_and(|(value, _)| *value <= completed)
            {
                ready.push(
                    retired
                        .pop_front()
                        .expect("retired resource queue front disappeared")
                        .1,
                );
            }
        }
        for resource in ready {
            self.release_retired_resource(resource);
        }
        if self
            .last_submission_value
            .get()
            .is_some_and(|value| value <= completed)
        {
            self.last_submission_value.set(None);
        }
    }

    fn ensure_value_sync_state<'a>(
        &self,
        value_ptr: GpuAddress,
        map: &'a mut HashMap<u64, MetalValueSyncState>,
    ) -> RhiResult<&'a mut MetalValueSyncState> {
        let key = value_ptr.0;
        match map.entry(key) {
            std::collections::hash_map::Entry::Occupied(entry) => Ok(entry.into_mut()),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let event = self.device.newSharedEvent().ok_or_else(|| {
                    RhiError::SyncError(format!(
                        "Failed to create Metal value event for {:#x}",
                        key
                    ))
                })?;
                event.setSignaledValue(0);
                Ok(entry.insert(MetalValueSyncState {
                    event,
                    scheduled_value: 0,
                }))
            }
        }
    }

    fn collect_value_waits(&self, waits: &[WaitValueDesc]) -> RhiResult<MetalEventWaits> {
        if waits.is_empty() {
            return Ok(SmallVec::new());
        }
        let mut map = self.value_sync.borrow_mut();
        let mut merged = MergedEvents::new();
        for wait in waits {
            let state = self.ensure_value_sync_state(wait.value_ptr, &mut map)?;
            let required = match wait.wait_op {
                WaitOp::GreaterOrEqual => wait.value,
                WaitOp::Equal => {
                    if wait.value < state.scheduled_value {
                        return Err(RhiError::SyncError(format!(
                            "Metal wait_before_value(Equal) on {:#x} requested {}, but value already advanced to {}",
                            wait.value_ptr.0, wait.value, state.scheduled_value
                        )));
                    }
                    wait.value
                }
                WaitOp::MaskedEqual => {
                    if wait.mask == u64::MAX {
                        if wait.value < state.scheduled_value {
                            return Err(RhiError::SyncError(format!(
                                "Metal wait_before_value(MaskedEqual full-mask) on {:#x} requested {}, but value already advanced to {}",
                                wait.value_ptr.0, wait.value, state.scheduled_value
                            )));
                        }
                        wait.value
                    } else {
                        let masked_current = state.scheduled_value & wait.mask;
                        let masked_target = wait.value & wait.mask;
                        if masked_current != masked_target {
                            return Err(RhiError::SyncError(format!(
                                "Metal wait_before_value(MaskedEqual) for {:#x} cannot be represented with queue event wait: current masked value {:#x} != target {:#x}",
                                wait.value_ptr.0, masked_current, masked_target
                            )));
                        }
                        state.scheduled_value
                    }
                }
            };
            state.scheduled_value = state.scheduled_value.max(required);
            merge_event(
                &mut merged,
                wait.value_ptr.0,
                &state.event,
                required,
                u64::max,
            );
        }
        Ok(finish_merged_events(merged))
    }

    fn collect_value_signals(&self, signals: &[SignalValueDesc]) -> RhiResult<MetalEventWaits> {
        if signals.is_empty() {
            return Ok(SmallVec::new());
        }
        let mut map = self.value_sync.borrow_mut();
        let mut merged = MergedEvents::new();
        for signal in signals {
            let state = self.ensure_value_sync_state(signal.value_ptr, &mut map)?;
            let target = match signal.signal_op {
                SignalOp::AtomicSet => signal.value,
                SignalOp::AtomicMax => signal.value.max(state.scheduled_value),
                SignalOp::AtomicOr => state.scheduled_value | signal.value,
            };
            if target < state.scheduled_value {
                return Err(RhiError::SyncError(format!(
                    "Metal signal_after_value would decrease {:#x} from {} to {}",
                    signal.value_ptr.0, state.scheduled_value, target
                )));
            }
            state.scheduled_value = target;
            merge_event(
                &mut merged,
                signal.value_ptr.0,
                &state.event,
                target,
                |_, latest| latest,
            );
        }
        Ok(finish_merged_events(merged))
    }

    pub fn submit_with_desc(
        &self,
        cmd: MetalCommandBuffer,
        desc: &SubmitDesc<'_>,
    ) -> RhiResult<()> {
        for (semaphore, _) in desc.wait_semaphores.iter().chain(desc.signal_semaphores) {
            if !matches!(&semaphore.inner, TimelineSemaphoreInner::Metal(_)) {
                return Err(RhiError::SyncError(
                    "Timeline semaphore backend mismatch on Metal queue submit".into(),
                ));
            }
        }
        let value_waits = self.collect_value_waits(&cmd.pending_value_waits)?;
        let value_signals = self.collect_value_signals(&cmd.pending_value_signals)?;
        if self.residency_dirty.replace(false) {
            self.residency_set.commit();
        }
        self.reclaim_completed_submissions();

        for (semaphore, value) in desc.wait_semaphores {
            match &semaphore.inner {
                TimelineSemaphoreInner::Metal(mtl_semaphore) => {
                    self.queue
                        .waitForEvent_value(shared_event_as_event(&mtl_semaphore.event), *value);
                }
                #[allow(unreachable_patterns)]
                _ => {
                    return Err(RhiError::SyncError(
                        "Timeline wait semaphore backend mismatch on Metal queue submit".into(),
                    ));
                }
            }
        }
        for (event, value) in value_waits {
            self.queue
                .waitForEvent_value(shared_event_as_event(&event), value);
        }

        let mut cmd = cmd;
        cmd.finish();
        self.commit_single(&cmd.command_buffer);

        for (semaphore, value) in desc.signal_semaphores {
            match &semaphore.inner {
                TimelineSemaphoreInner::Metal(mtl_semaphore) => {
                    self.queue
                        .signalEvent_value(shared_event_as_event(&mtl_semaphore.event), *value);
                }
                #[allow(unreachable_patterns)]
                _ => {
                    return Err(RhiError::SyncError(
                        "Timeline signal semaphore backend mismatch on Metal queue submit".into(),
                    ));
                }
            }
        }
        for (event, value) in value_signals {
            self.queue
                .signalEvent_value(shared_event_as_event(&event), value);
        }

        let value = self.next_fence_value();
        self.queue
            .signalEvent_value(shared_event_as_event(&self.frame_event), value);
        cmd.resolve_recording();
        self.pending_submissions
            .borrow_mut()
            .push_back((value, cmd));
        self.last_submission_value.set(Some(value));
        self.schedule_pending_retirements();
        Ok(())
    }

    pub fn submit_frame(
        &self,
        cmd: MetalCommandBuffer,
        sc: &MetalSwapchain,
        frame_index: usize,
        _image_index: u32,
    ) -> RhiResult<()> {
        if frame_index >= self.frame_fence_values.borrow().len()
            || frame_index >= self.in_flight_frame_commands.borrow().len()
        {
            return Err(RhiError::Backend(
                "invalid frame index for Metal queue submission".into(),
            ));
        }
        let value_waits = self.collect_value_waits(&cmd.pending_value_waits)?;
        let value_signals = self.collect_value_signals(&cmd.pending_value_signals)?;
        if self.residency_dirty.replace(false) {
            self.residency_set.commit();
        }
        self.reclaim_completed_submissions();

        for (event, value) in value_waits {
            self.queue
                .waitForEvent_value(shared_event_as_event(&event), value);
        }

        // Taken during recording, so the queue-side wait is enqueued here — still before the
        // commit that renders into it. A frame that never touched the swapchain skips this.
        let drawable = sc.drawable.current();
        if let Some(drawable) = drawable.as_ref() {
            self.queue.waitForDrawable(drawable);
        }

        let mut cmd = cmd;
        cmd.finish();
        self.commit_single(&cmd.command_buffer);

        for (event, value) in value_signals {
            self.queue
                .signalEvent_value(shared_event_as_event(&event), value);
        }

        let value = self.next_fence_value();
        self.frame_fence_values.borrow_mut()[frame_index] = value;
        self.queue
            .signalEvent_value(shared_event_as_event(&self.frame_event), value);
        cmd.resolve_recording();

        if let Some(drawable) = drawable {
            self.queue.signalDrawable(&drawable);
            drawable.present();
        }
        sc.drawable.release();

        let mut frame_cmds = self.in_flight_frame_commands.borrow_mut();
        if frame_cmds[frame_index].is_some() {
            log::warn!("Overwriting in-flight Metal frame command before completion");
        }
        frame_cmds[frame_index] = Some(cmd);
        self.last_submission_value.set(Some(value));
        self.schedule_pending_retirements();

        Ok(())
    }

    pub fn acquire_image(
        &self,
        sc: &MetalSwapchain,
        frame_index: usize,
    ) -> RhiResult<AcquiredImage> {
        if frame_index >= self.frame_fence_values.borrow().len()
            || frame_index >= self.in_flight_frame_commands.borrow().len()
        {
            return Err(RhiError::Backend("invalid Metal frame index".into()));
        }
        self.reclaim_completed_submissions();

        let value = self.frame_fence_values.borrow()[frame_index];
        if value != 0 {
            let _ = self
                .frame_event
                .waitUntilSignaledValue_timeoutMS(value, u64::MAX);
        }
        if frame_index < self.in_flight_frame_commands.borrow().len() {
            self.in_flight_frame_commands.borrow_mut()[frame_index] = None;
        }

        // No `nextDrawable` here: everything below comes from the layer's configuration, so the
        // drawable stays in the pool until the frame encodes into it. See `MetalDrawableSlot`.
        Ok(AcquiredImage {
            index: 0, // Metal only has one "current" drawable
            format: sc.format,
            width: sc.extent[0],
            height: sc.extent[1],
        })
    }

    pub fn wait_idle(&self) {
        let value = self.next_fence_value();
        self.queue
            .signalEvent_value(shared_event_as_event(&self.frame_event), value);
        self.last_submission_value.set(Some(value));
        self.schedule_pending_retirements();
        let _ = self
            .frame_event
            .waitUntilSignaledValue_timeoutMS(value, u64::MAX);
        self.pending_submissions.borrow_mut().clear();
        for slot in self.in_flight_frame_commands.borrow_mut().iter_mut() {
            *slot = None;
        }
        self.reclaim_retired_resources();
    }

    fn next_fence_value(&self) -> u64 {
        let value = self.frame_fence_next.get().wrapping_add(1);
        self.frame_fence_next.set(value);
        value
    }

    fn reclaim_completed_submissions(&self) {
        let completed = self.frame_event.signaledValue();
        let mut pending = self.pending_submissions.borrow_mut();
        while pending
            .front()
            .is_some_and(|(value, _)| *value <= completed)
        {
            pending
                .pop_front()
                .expect("pending submission queue front disappeared");
        }
        self.reclaim_retired_resources();
    }

    fn commit_single(&self, cmd: &Retained<ProtocolObject<dyn MTL4CommandBuffer>>) {
        let cmd_ptr = NonNull::from(cmd.as_ref());
        let mut bufs = [cmd_ptr];
        unsafe {
            let ptr =
                NonNull::new(bufs.as_mut_ptr()).expect("command buffer array pointer is null");
            self.queue.commit_count(ptr, 1);
        }
    }
}

const METAL_MDI_ICB_SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_command_buffer>
using namespace metal;

struct RhiDrawIndirectMultiArgs {
    uint vertex_count;
    uint instance_count;
    uint first_vertex;
    uint first_instance;
};

struct RhiIcbRange {
    uint location;
    uint length;
};

// MSL exposes `command_buffer` through an argument buffer; slot 3 carries its resource ID.
struct RhiIcbContainer {
    command_buffer cmd [[id(0)]];
};

static inline primitive_type rhi_primitive_type(uint id) {
    switch (id) {
        case 0: return primitive_type::point;
        case 1: return primitive_type::line;
        case 2: return primitive_type::line_strip;
        case 4: return primitive_type::triangle_strip;
        default: return primitive_type::triangle;
    }
}

kernel void rhi_encode_mdi_icb(
    device const RhiDrawIndirectMultiArgs* draws [[buffer(0)]],
    device atomic_uint* drawCount [[buffer(1)]],
    device RhiIcbRange* range [[buffer(2)]],
    device const RhiIcbContainer& icb_container [[buffer(3)]],
    constant uint& maxDrawCount [[buffer(4)]],
    constant uint& primitiveType [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    uint count = min(atomic_load_explicit(drawCount, memory_order_relaxed), maxDrawCount);
    if (tid == 0) {
        range->location = 0;
        range->length = count;
    }
    if (tid >= count) {
        return;
    }

    RhiDrawIndirectMultiArgs draw = draws[tid];

    render_command cmd(icb_container.cmd, tid);
    cmd.draw_primitives(
        rhi_primitive_type(primitiveType),
        draw.first_vertex,
        draw.vertex_count,
        draw.instance_count,
        draw.first_instance);
}
"#;

/// Translate the unified `Cull` value into Metal's `(cull_mode, front-face winding)` pair.
/// All variants imply CCW as the front-face convention. `Cull::All` is approximated as
/// Back + CW since Metal has no FRONT_AND_BACK cull mode.
fn cull_to_mtl(cull: Cull) -> (MTLCullMode, MTLWinding) {
    match cull {
        Cull::None => (MTLCullMode::None, MTLWinding::CounterClockwise),
        Cull::Cw => (MTLCullMode::Back, MTLWinding::CounterClockwise),
        Cull::Ccw => (MTLCullMode::Front, MTLWinding::CounterClockwise),
        Cull::All => (MTLCullMode::Back, MTLWinding::Clockwise),
    }
}

fn create_mdi_icb_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
) -> RhiResult<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version4_0);
    let source = NSString::from_str(METAL_MDI_ICB_SOURCE);
    let library = device
        .newLibraryWithSource_options_error(&source, Some(&options))
        .map_err(|e| {
            RhiError::PipelineCreation(format!(
                "Metal MDI ICB encoder library compilation failed: {e}"
            ))
        })?;
    let function_name = NSString::from_str("rhi_encode_mdi_icb");
    let function = library.newFunctionWithName(&function_name).ok_or_else(|| {
        RhiError::PipelineCreation("Metal MDI ICB encoder function was not found".into())
    })?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| {
            RhiError::PipelineCreation(format!(
                "Metal MDI ICB encoder pipeline creation failed: {e}"
            ))
        })
}

impl MetalDevice {
    pub fn new(desc: &DeviceDesc) -> RhiResult<Self> {
        let device = MTLCreateSystemDefaultDevice().ok_or(RhiError::NoSuitableGpu)?;

        let residency_desc = MTLResidencySetDescriptor::new();
        unsafe {
            residency_desc.setInitialCapacity(1024);
        }
        let residency_set = device
            .newResidencySetWithDescriptor_error(&residency_desc)
            .map_err(|e| {
                RhiError::DeviceCreation(format!("Failed to create Metal residency set: {e}"))
            })?;

        let queue = device.newMTL4CommandQueue().ok_or_else(|| {
            RhiError::DeviceCreation("Failed to create Metal 4 command queue".into())
        })?;

        queue.addResidencySet(&residency_set);

        let frame_event = device
            .newSharedEvent()
            .ok_or_else(|| RhiError::DeviceCreation("Failed to create MTLSharedEvent".into()))?;
        let buffer_pool: SharedMetalBufferPool =
            Rc::new(RefCell::new(MetalBufferPool::new(device.clone())));
        let frame_fence_values: FrameFenceValues =
            Rc::new(RefCell::new([0u64; MAX_FRAMES_IN_FLIGHT]));
        let frame_fence_next = Rc::new(Cell::new(0u64));
        let in_flight_frame_commands: InFlightFrameCommands = Rc::new(RefCell::new(
            std::iter::repeat_with(|| None)
                .take(MAX_FRAMES_IN_FLIGHT)
                .collect(),
        ));
        let pending_submissions: PendingSubmissions = Rc::new(RefCell::new(VecDeque::new()));
        let free_texture_ids = Rc::new(RefCell::new(Vec::new()));
        let free_sampler_ids = Rc::new(RefCell::new(Vec::new()));
        let active_recordings = Rc::new(Cell::new(0));

        let residency_dirty = Rc::new(Cell::new(false));
        let frame_table_slots = Rc::new(RefCell::new(vec![None; MAX_FRAMES_IN_FLIGHT]));
        let metal_queue = MetalQueue {
            queue: queue.clone(),
            device: device.clone(),
            residency_set: residency_set.clone(),
            residency_dirty: residency_dirty.clone(),
            frame_fence_values: frame_fence_values.clone(),
            frame_fence_next,
            frame_event: frame_event.clone(),
            in_flight_frame_commands,
            pending_submissions,
            retired_resources: Rc::new(RefCell::new(VecDeque::new())),
            pending_retired_resources: Rc::new(RefCell::new(Vec::new())),
            pending_retirement_work: Rc::new(Cell::new(false)),
            active_recordings: active_recordings.clone(),
            last_submission_value: Cell::new(None),
            free_texture_ids: free_texture_ids.clone(),
            free_sampler_ids: free_sampler_ids.clone(),
            value_sync: RefCell::new(HashMap::new()),
        };

        let rhi_queue = Queue {
            inner: QueueInner::Metal(Box::new(metal_queue)),
        };

        log::info!("Metal device created: {}", device.name());

        if desc.bindless_mode == Some(BindlessMode::DescriptorBuffer) {
            return Err(RhiError::Unsupported(
                "Metal requires argument-table bindless mode".into(),
            ));
        }
        let bindless_mode = BindlessMode::ArgumentTable;
        let mdi_icb = create_mdi_icb_pipeline(device.as_ref())?;

        let device = Self {
            device,
            buffer_pool,
            rhi_queue,
            residency_set,
            residency_dirty,
            textures: Rc::new(RefCell::new(Vec::new())),
            texture_view_flags: RefCell::new(Vec::new()),
            free_texture_ids,
            samplers: Rc::new(RefCell::new(Vec::new())),
            free_sampler_ids,
            texture_generation: Rc::new(Cell::new(1)),
            sampler_generation: Rc::new(Cell::new(1)),
            allocations: Rc::new(RefCell::new(BTreeMap::new())),
            mapped_allocations: Rc::new(RefCell::new(BTreeMap::new())),
            active_recordings,
            frame_fence_values,
            frame_event,
            frame_table_slots,
            table_slot_pool: Rc::new(RefCell::new(Vec::new())),
            bindless_mode,
            accel_counter: RefCell::new(0),
            mdi_icb_pipeline: mdi_icb,
        };

        Ok(device)
    }

    pub fn queue(&self) -> &Queue {
        &self.rhi_queue
    }

    pub fn bindless_mode(&self) -> BindlessMode {
        self.bindless_mode
    }

    pub fn wait_idle(&self) {
        match &self.rhi_queue.inner {
            QueueInner::Metal(q) => q.wait_idle(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
        // Heaps are only handed back here, never on the retire path, so this cannot land
        // mid-frame. The queue is idle already.
        self.buffer_pool.borrow_mut().trim();
    }

    pub fn wait_for_frame(&self, frame_index: usize) {
        let Some(value) = self.frame_fence_values.borrow().get(frame_index).copied() else {
            log::warn!("Ignoring invalid Metal frame index {frame_index}");
            return;
        };
        if value != 0 {
            let _ = self
                .frame_event
                .waitUntilSignaledValue_timeoutMS(value, u64::MAX);
        }
        match &self.rhi_queue.inner {
            #[cfg(feature = "metal")]
            QueueInner::Metal(q) => {
                if frame_index < q.in_flight_frame_commands.borrow().len() {
                    q.in_flight_frame_commands.borrow_mut()[frame_index] = None;
                }
                q.reclaim_completed_submissions();
            }
            #[cfg(feature = "vulkan")]
            QueueInner::Vulkan(_) => {}
        }
    }

    pub fn create_surface(&self, desc: &SurfaceDesc) -> RhiResult<Surface> {
        let layer = match desc.window_handle {
            RawWindowHandle::AppKit(handle) => unsafe {
                use objc2::msg_send;
                use objc2::runtime::{AnyObject, Bool};

                let ns_view = handle.ns_view.as_ptr() as *mut AnyObject;

                let layer = CAMetalLayer::new();
                layer.setDevice(Some(&self.device));
                layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm_sRGB);
                layer.setFramebufferOnly(true);

                let _: () = msg_send![ns_view, setWantsLayer: Bool::YES];
                let layer_ptr: *mut AnyObject = objc2::rc::Retained::as_ptr(&layer) as *mut _;
                let _: () = msg_send![ns_view, setLayer: layer_ptr];

                layer
            },
            _ => {
                return Err(RhiError::SurfaceCreation(
                    "Only AppKit windows are supported for Metal".into(),
                ));
            }
        };

        Ok(Surface {
            inner: SurfaceInner::Metal(MetalSurface { layer }),
        })
    }

    pub fn create_swapchain(
        &self,
        surface: &Surface,
        desc: &SwapchainDesc,
    ) -> RhiResult<Swapchain> {
        let layer = match &surface.inner {
            SurfaceInner::Metal(s) => &s.layer,
            #[allow(unreachable_patterns)]
            _ => unreachable!("wrong backend"),
        };

        let mtl_format = format_to_mtl(desc.format);
        layer.setPixelFormat(mtl_format);
        layer.setDisplaySyncEnabled(desc.vsync);
        layer.setMaximumDrawableCount(desc.image_count.clamp(2, 3) as usize);
        layer.setDrawableSize(CGSize {
            width: desc.width as f64,
            height: desc.height as f64,
        });

        let format = mtl_to_format(layer.pixelFormat());

        Ok(Swapchain {
            inner: SwapchainInner::Metal(Box::new(MetalSwapchain {
                drawable: Rc::new(MetalDrawableSlot::new(layer.clone())),
                format,
                extent: [desc.width, desc.height],
            })),
        })
    }

    pub fn recreate_swapchain(
        &self,
        swapchain: &mut Swapchain,
        desc: &SwapchainDesc,
    ) -> RhiResult<()> {
        match &mut swapchain.inner {
            SwapchainInner::Metal(sc) => {
                self.wait_idle();
                sc.drawable.release();
                let layer = &sc.drawable.layer;
                layer.setPixelFormat(format_to_mtl(desc.format));
                layer.setDisplaySyncEnabled(desc.vsync);
                layer.setMaximumDrawableCount(desc.image_count.clamp(2, 3) as usize);
                layer.setDrawableSize(CGSize {
                    width: desc.width as f64,
                    height: desc.height as f64,
                });
                sc.format = mtl_to_format(layer.pixelFormat());
                sc.extent = [desc.width, desc.height];
                Ok(())
            }
            #[allow(unreachable_patterns)]
            _ => unreachable!("wrong backend"),
        }
    }

    pub fn create_buffer(&self, desc: &BufferDesc) -> RhiResult<GpuBuffer> {
        let length = usize::try_from(desc.size).map_err(|_| {
            RhiError::BufferCreation("Metal buffer size exceeds the host address space".into())
        })?;
        let metal_buffer =
            MetalBufferPool::allocate_shared(&self.buffer_pool, length, desc.memory)?;

        // Track the texture for Metal 4 residency.
        let allocation = unsafe {
            &*(metal_buffer.buffer.as_ref() as *const ProtocolObject<dyn MTLBuffer>
                as *const ProtocolObject<dyn MTLAllocation>)
        };
        self.residency_set.addAllocation(allocation);
        self.residency_dirty.set(true);

        if let Some(label) = &desc.label {
            use objc2_metal::MTLResource;
            let ns_label = NSString::from_str(label);
            metal_buffer.buffer.setLabel(Some(&ns_label));
        }

        {
            let mut allocations = self.allocations.borrow_mut();
            allocations.insert(
                metal_buffer.gpu_address().0,
                BufferAllocation {
                    base: metal_buffer.gpu_address(),
                    size: metal_buffer.size,
                    buffer: metal_buffer.buffer.clone(),
                    heap: metal_buffer.heap.clone(),
                },
            );
        }
        if let Some(mapped_ptr) = metal_buffer.mapped_ptr() {
            self.mapped_allocations.borrow_mut().insert(
                mapped_ptr as usize,
                MappedAllocation {
                    gpu_base: metal_buffer.gpu_address(),
                    size: metal_buffer.size,
                },
            );
        }

        Ok(GpuBuffer {
            inner: GpuBufferInner::Metal(metal_buffer),
        })
    }

    pub fn host_to_device_pointer(&self, cpu_ptr: *const u8) -> Option<GpuAddress> {
        if cpu_ptr.is_null() {
            return None;
        }
        let ptr = cpu_ptr as usize;
        let allocations = self.mapped_allocations.borrow();
        resolve_mapped_pointer(&allocations, ptr)
    }

    /// Build the native `MTLTextureDescriptor` for a `TextureDesc`. Shared by
    /// `texture_size_align` and `create_texture` so the mapping lives in one place.
    fn build_texture_descriptor(&self, desc: &TextureDesc) -> Retained<MTLTextureDescriptor> {
        let mtl_desc = MTLTextureDescriptor::new();

        let texture_type = match desc.dimension {
            TextureDimension::D1 => MTLTextureType::Type1D,
            TextureDimension::D2 => MTLTextureType::Type2D,
            TextureDimension::D2Array => MTLTextureType::Type2DArray,
            TextureDimension::D3 => MTLTextureType::Type3D,
            TextureDimension::Cube => MTLTextureType::TypeCube,
            TextureDimension::CubeArray => MTLTextureType::TypeCubeArray,
        };

        let sample_count = match desc.sample_count {
            SampleCount::S1 => 1usize,
            SampleCount::S2 => 2,
            SampleCount::S4 => 4,
            SampleCount::S8 => 8,
            SampleCount::S16 => 16,
        };

        let mut usage = MtlTextureUsage::empty();
        if desc.usage.contains(TextureUsage::SAMPLED) {
            usage |= MtlTextureUsage::ShaderRead;
        }
        if desc.usage.contains(TextureUsage::STORAGE) {
            usage |= MtlTextureUsage::ShaderRead | MtlTextureUsage::ShaderWrite;
        }
        if desc.usage.contains(TextureUsage::COLOR_ATTACHMENT) {
            usage |= MtlTextureUsage::RenderTarget;
        }
        if desc.usage.contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT) {
            usage |= MtlTextureUsage::RenderTarget;
        }

        unsafe {
            mtl_desc.setPixelFormat(format_to_mtl(desc.format));
            mtl_desc.setWidth(desc.width as usize);
            mtl_desc.setHeight(desc.height as usize);
            mtl_desc.setDepth(desc.depth as usize);
            mtl_desc.setMipmapLevelCount(desc.mip_levels as usize);
            mtl_desc.setArrayLength(desc.array_layers as usize);
            mtl_desc.setTextureType(texture_type);
            mtl_desc.setSampleCount(sample_count);
            mtl_desc.setUsage(usage);
            mtl_desc.setStorageMode(MTLStorageMode::Private);
        }

        mtl_desc
    }

    pub fn texture_size_align(&self, desc: &TextureDesc) -> RhiResult<TextureSizeAlign> {
        let mtl_desc = self.build_texture_descriptor(desc);
        let size_align = self.device.heapTextureSizeAndAlignWithDescriptor(&mtl_desc);
        Ok(TextureSizeAlign {
            size: size_align.size as u64,
            align: size_align.align as u64,
        })
    }

    fn allocate_texture_id(&self) -> RhiResult<TextureId> {
        if let Some(id) = self.free_texture_ids.borrow_mut().pop() {
            return Ok(id);
        }
        let next = self.textures.borrow().len();
        if next > u32::MAX as usize {
            return Err(RhiError::TextureCreation(
                "Metal texture ID space exhausted".into(),
            ));
        }
        Ok(TextureId(next as u32))
    }

    fn allocate_sampler_id(&self) -> RhiResult<SamplerId> {
        if let Some(id) = self.free_sampler_ids.borrow_mut().pop() {
            return Ok(id);
        }
        let next = self.samplers.borrow().len();
        if next > u32::MAX as usize {
            return Err(RhiError::Backend("Metal sampler ID space exhausted".into()));
        }
        Ok(SamplerId(next as u32))
    }

    pub fn create_texture(
        &self,
        desc: &TextureDesc,
        texture_gpu: GpuAddress,
    ) -> RhiResult<Texture> {
        if texture_gpu.is_null() {
            return Err(RhiError::TextureCreation(
                "create_texture requires a non-null texture allocation address".into(),
            ));
        }

        let mtl_desc = self.build_texture_descriptor(desc);
        let size_align = self.device.heapTextureSizeAndAlignWithDescriptor(&mtl_desc);
        let (heap, heap_offset) = {
            let allocations = self.allocations.borrow();
            let alloc = allocations
                .range(..=texture_gpu.0)
                .next_back()
                .map(|(_, alloc)| alloc)
                .filter(|alloc| texture_gpu.0 - alloc.base.0 < alloc.size)
                .ok_or_else(|| {
                    RhiError::TextureCreation(format!(
                        "texture allocation address 0x{:x} was not returned by gpuMalloc",
                        texture_gpu.0
                    ))
                })?;

            let offset = texture_gpu.0 - alloc.base.0;
            if !offset.is_multiple_of(size_align.align as u64) {
                return Err(RhiError::TextureCreation(format!(
                    "texture allocation address 0x{:x} has heap offset {offset}, expected alignment {}",
                    texture_gpu.0, size_align.align
                )));
            }
            if size_align.size as u64 > alloc.size - offset {
                return Err(RhiError::TextureCreation(format!(
                    "texture allocation address 0x{:x} has {} bytes available, needs {}",
                    texture_gpu.0,
                    alloc.size - offset,
                    size_align.size
                )));
            }
            (alloc.heap.clone(), offset)
        };

        let texture =
            unsafe { heap.newTextureWithDescriptor_offset(&mtl_desc, heap_offset as usize) }
                .ok_or_else(|| {
                    RhiError::TextureCreation("Metal placed texture allocation failed".into())
                })?;

        // Track the buffer for Metal 4 residency.
        let allocation = unsafe {
            &*(texture.as_ref() as *const ProtocolObject<dyn MTLTexture>
                as *const ProtocolObject<dyn MTLAllocation>)
        };
        self.residency_set.addAllocation(allocation);
        self.residency_dirty.set(true);

        if let Some(label) = &desc.label {
            use objc2_metal::MTLResource;
            let ns_label = NSString::from_str(label);
            texture.setLabel(Some(&ns_label));
        }

        let id = self.allocate_texture_id()?;
        let idx = id.0 as usize;
        let mut textures = self.textures.borrow_mut();
        if textures.len() <= idx {
            textures.resize_with(idx + 1, || None);
        }
        textures[idx] = Some(texture.clone());
        drop(textures);
        let mut view_flags = self.texture_view_flags.borrow_mut();
        if view_flags.len() <= idx {
            view_flags.resize(idx + 1, false);
        }
        view_flags[idx] = false;
        self.texture_generation
            .set(self.texture_generation.get().wrapping_add(1));

        Ok(Texture {
            id,
            gpu_address: texture_gpu,
            desc: desc.clone(),
        })
    }

    pub fn create_sampler(&self, desc: &SamplerDesc) -> RhiResult<Sampler> {
        let mtl_desc = MTLSamplerDescriptor::new();

        mtl_desc.setMinFilter(filter_to_mtl(desc.min_filter));
        mtl_desc.setMagFilter(filter_to_mtl(desc.mag_filter));
        mtl_desc.setMipFilter(mip_filter_to_mtl(desc.mip_filter));
        mtl_desc.setSAddressMode(address_to_mtl(desc.address_u));
        mtl_desc.setTAddressMode(address_to_mtl(desc.address_v));
        mtl_desc.setRAddressMode(address_to_mtl(desc.address_w));
        mtl_desc.setLodMinClamp(desc.min_lod);
        mtl_desc.setLodMaxClamp(desc.max_lod);

        if let Some(aniso) = desc.max_anisotropy {
            mtl_desc.setMaxAnisotropy(aniso as usize);
        }

        if let Some(cmp) = desc.compare {
            mtl_desc.setCompareFunction(compare_op_to_mtl(cmp));
        }
        mtl_desc.setSupportArgumentBuffers(true);

        let sampler = self
            .device
            .newSamplerStateWithDescriptor(&mtl_desc)
            .ok_or_else(|| RhiError::Backend("Failed to create Metal sampler".into()))?;

        let id = self.allocate_sampler_id()?;
        let idx = id.0 as usize;
        let mut samplers = self.samplers.borrow_mut();
        if samplers.len() <= idx {
            samplers.resize_with(idx + 1, || None);
        }
        samplers[idx] = Some(sampler.clone());
        drop(samplers);
        self.sampler_generation
            .set(self.sampler_generation.get().wrapping_add(1));

        Ok(Sampler { id })
    }

    pub fn create_shader_module(&self, desc: &ShaderModuleDesc) -> RhiResult<ShaderModule> {
        // For Metal, `code` should be a compiled .metallib binary.
        let ptr = std::ptr::NonNull::new(desc.code.as_ptr() as *mut std::ffi::c_void)
            .expect("shader code pointer is null");
        let dispatch_data = unsafe {
            dispatch2::DispatchData::new(ptr, desc.code.len(), None, std::ptr::null_mut())
        };

        let library = self
            .device
            .newLibraryWithData_error(&dispatch_data)
            .map_err(|e| {
                RhiError::ShaderCompilation(format!("Metal library creation failed: {e}"))
            })?;

        Ok(ShaderModule {
            inner: crate::shader::ShaderModuleInner::Metal(Box::new(MetalShaderModule {
                library,
                entry_point: desc.entry_point.to_string(),
            })),
            stage: desc.stage,
        })
    }

    pub fn create_graphics_pso(
        &self,
        desc: &GraphicsPsoDesc,
        vert_module: &MetalShaderModule,
        frag_module: &MetalShaderModule,
    ) -> RhiResult<GraphicsPso> {
        let compiler_desc = MTL4CompilerDescriptor::new();
        let compiler = self
            .device
            .newCompilerWithDescriptor_error(&compiler_desc)
            .map_err(|e| {
                RhiError::PipelineCreation(format!("Metal MTL4 compiler creation failed: {e}"))
            })?;

        let mut color_formats = Vec::with_capacity(desc.color_targets.len());
        let mut color_write_masks = Vec::with_capacity(desc.color_targets.len());
        for target in &desc.color_targets {
            color_formats.push(format_to_mtl(target.format));
            color_write_masks.push(target.write_mask);
        }

        let sample_count = match desc.sample_count {
            SampleCount::S1 => 1,
            SampleCount::S2 => 2,
            SampleCount::S4 => 4,
            SampleCount::S8 => 8,
            SampleCount::S16 => 16,
        };

        let initial_blend = desc.blendstate.as_ref().cloned().unwrap_or_default();
        let pipeline_state = MetalGraphicsPso::compile_pipeline_state(
            compiler.as_ref(),
            vert_module.library.as_ref(),
            &vert_module.entry_point,
            frag_module.library.as_ref(),
            &frag_module.entry_point,
            &color_formats,
            &color_write_masks,
            sample_count,
            desc.alpha_to_coverage,
            &initial_blend,
        )?;

        let graphics_argument_buffer_slots = pipeline_state
            .reflection()
            .map(|r| {
                let mut slots = Vec::new();
                let collect_slots =
                    |bindings: &objc2_foundation::NSArray<ProtocolObject<dyn MTLBinding>>,
                     slots: &mut Vec<usize>| {
                        for i in 0..bindings.count() {
                            let binding = bindings.objectAtIndexedSubscript(i);
                            if binding.r#type() == MTLBindingType::Buffer && binding.isArgument() {
                                slots.push(binding.index());
                            }
                        }
                    };
                collect_slots(r.vertexBindings().as_ref(), &mut slots);
                collect_slots(r.fragmentBindings().as_ref(), &mut slots);
                slots.sort_unstable();
                slots.dedup();
                slots
            })
            .unwrap_or_default();

        let (cull_mode, winding) = cull_to_mtl(desc.cull);

        let topology = match desc.topology {
            Topology::TriangleList => objc2_metal::MTLPrimitiveType::Triangle,
            Topology::TriangleStrip => objc2_metal::MTLPrimitiveType::TriangleStrip,
            // Metal has no native TriangleFan; use TriangleList instead.
            Topology::TriangleFan => panic!(
                "TriangleFan is not supported on Metal. \
                 Rewrite fan indices to TriangleList before creating this PSO."
            ),
        };

        let mut blend_pipelines = HashMap::new();
        blend_pipelines.insert(initial_blend, pipeline_state.clone());

        Ok(GraphicsPso {
            inner: GraphicsPsoInner::Metal(Box::new(MetalGraphicsPso {
                cull_mode,
                winding,
                topology,
                compiler,
                vertex_library: vert_module.library.clone(),
                vertex_entry_point: vert_module.entry_point.clone(),
                fragment_library: frag_module.library.clone(),
                fragment_entry_point: frag_module.entry_point.clone(),
                color_formats,
                color_write_masks,
                sample_count,
                alpha_to_coverage: desc.alpha_to_coverage,
                graphics_argument_buffer_slots,
                blend_pipelines: RefCell::new(blend_pipelines),
            })),
        })
    }

    pub fn create_compute_pso(
        &self,
        desc: &ComputePsoDesc,
        compute_module: &MetalShaderModule,
    ) -> RhiResult<ComputePso> {
        let fn_name = NSString::from_str(&compute_module.entry_point);
        let compiler_desc = MTL4CompilerDescriptor::new();
        let compiler = self
            .device
            .newCompilerWithDescriptor_error(&compiler_desc)
            .map_err(|e| {
                RhiError::PipelineCreation(format!("Metal MTL4 compiler creation failed: {e}"))
            })?;

        let func_desc = MTL4LibraryFunctionDescriptor::new();
        func_desc.setName(Some(&fn_name));
        func_desc.setLibrary(Some(&compute_module.library));

        let pipeline_desc = MTL4ComputePipelineDescriptor::new();
        pipeline_desc.setComputeFunctionDescriptor(Some(&func_desc));
        let pipeline_options = MTL4PipelineOptions::new();
        pipeline_options.setShaderReflection(
            MTL4ShaderReflection::BindingInfo | MTL4ShaderReflection::BufferTypeInfo,
        );
        pipeline_desc.setOptions(Some(&pipeline_options));
        if desc.threads_per_threadgroup.contains(&0) {
            return Err(RhiError::PipelineCreation(
                "Metal compute PSO requires non-zero threads_per_threadgroup".into(),
            ));
        }
        let tg = objc2_metal::MTLSize {
            width: desc.threads_per_threadgroup[0] as usize,
            height: desc.threads_per_threadgroup[1] as usize,
            depth: desc.threads_per_threadgroup[2] as usize,
        };
        pipeline_desc.setRequiredThreadsPerThreadgroup(tg);

        let pipeline_state = compiler
            .newComputePipelineStateWithDescriptor_compilerTaskOptions_error(&pipeline_desc, None)
            .map_err(|e| {
                RhiError::PipelineCreation(format!("Metal compute PSO creation failed: {e}"))
            })?;
        let compute_argument_buffer_slots = pipeline_state
            .reflection()
            .map(|r| {
                let bindings = r.bindings();
                let mut slots = Vec::new();
                for i in 0..bindings.count() {
                    let binding = bindings.objectAtIndexedSubscript(i);
                    if binding.r#type() == MTLBindingType::Buffer && binding.isArgument() {
                        slots.push(binding.index());
                    }
                }
                slots.sort_unstable();
                slots.dedup();
                slots
            })
            .unwrap_or_default();

        Ok(ComputePso {
            inner: ComputePsoInner::Metal(Box::new(MetalComputePso {
                pipeline: pipeline_state,
                threads_per_threadgroup: desc.threads_per_threadgroup,
                compute_argument_buffer_slots,
            })),
        })
    }

    pub fn create_meshlet_pso(
        &self,
        desc: &MeshletPsoDesc,
        mesh_module: &MetalShaderModule,
        frag_module: &MetalShaderModule,
    ) -> RhiResult<MeshletPso> {
        use super::pipeline::MetalMeshletPso;
        use objc2_metal::MTL4MeshRenderPipelineDescriptor;

        let compiler_desc = MTL4CompilerDescriptor::new();
        let compiler = self
            .device
            .newCompilerWithDescriptor_error(&compiler_desc)
            .map_err(|e| RhiError::PipelineCreation(format!("MTL4 compiler: {e}")))?;

        let mesh_fn_name = NSString::from_str(&mesh_module.entry_point);
        let mesh_func_desc = MTL4LibraryFunctionDescriptor::new();
        mesh_func_desc.setName(Some(&mesh_fn_name));
        mesh_func_desc.setLibrary(Some(&mesh_module.library));

        let frag_fn_name = NSString::from_str(&frag_module.entry_point);
        let frag_func_desc = MTL4LibraryFunctionDescriptor::new();
        frag_func_desc.setName(Some(&frag_fn_name));
        frag_func_desc.setLibrary(Some(&frag_module.library));

        let pipeline_desc = MTL4MeshRenderPipelineDescriptor::new();
        pipeline_desc.setMeshFunctionDescriptor(Some(&mesh_func_desc));
        pipeline_desc.setFragmentFunctionDescriptor(Some(&frag_func_desc));

        let sample_count = match desc.sample_count {
            SampleCount::S1 => 1,
            SampleCount::S2 => 2,
            SampleCount::S4 => 4,
            SampleCount::S8 => 8,
            SampleCount::S16 => 16,
        };
        unsafe {
            pipeline_desc.setRasterSampleCount(sample_count);
            if desc.alpha_to_coverage {
                pipeline_desc
                    .setAlphaToCoverageState(objc2_metal::MTL4AlphaToCoverageState::Enabled);
            }
        }

        let mut color_formats = Vec::with_capacity(desc.color_targets.len());
        for target in &desc.color_targets {
            color_formats.push(super::texture::format_to_mtl(target.format));
        }

        for (i, &fmt) in color_formats.iter().enumerate() {
            let att = unsafe { pipeline_desc.colorAttachments().objectAtIndexedSubscript(i) };
            att.setPixelFormat(fmt);
        }

        let pipeline_options = MTL4PipelineOptions::new();
        pipeline_options.setShaderReflection(
            MTL4ShaderReflection::BindingInfo | MTL4ShaderReflection::BufferTypeInfo,
        );
        pipeline_desc.setOptions(Some(&pipeline_options));

        // Reflection supplies the bindless argument-buffer slots.
        let base_desc: &MTL4PipelineDescriptor = pipeline_desc.as_ref();
        let default_pipeline = compiler
            .newRenderPipelineStateWithDescriptor_compilerTaskOptions_error(base_desc, None)
            .map_err(|e| RhiError::PipelineCreation(format!("Mesh PSO: {e}")))?;

        let argument_buffer_slots = default_pipeline
            .reflection()
            .map(|r| {
                let mut slots = Vec::new();
                let collect_slots =
                    |bindings: &objc2_foundation::NSArray<ProtocolObject<dyn MTLBinding>>,
                     slots: &mut Vec<usize>| {
                        for i in 0..bindings.count() {
                            let binding = bindings.objectAtIndexedSubscript(i);
                            if binding.r#type() == MTLBindingType::Buffer && binding.isArgument() {
                                slots.push(binding.index());
                            }
                        }
                    };
                collect_slots(&r.meshBindings(), &mut slots);
                collect_slots(&r.fragmentBindings(), &mut slots);
                slots.sort_unstable();
                slots.dedup();
                slots
            })
            .unwrap_or_default();

        let (cull_mode, winding) = cull_to_mtl(desc.cull);

        Ok(MeshletPso {
            inner: crate::pipeline::MeshletPsoInner::Metal(Box::new(MetalMeshletPso {
                cull_mode,
                winding,
                argument_buffer_slots,
                default_pipeline,
            })),
        })
    }

    pub fn create_blas(&self, desc: &BlasDesc) -> RhiResult<AccelerationStructure> {
        use super::accel::make_blas_geometry_descriptors;
        use objc2_metal::MTL4PrimitiveAccelerationStructureDescriptor;

        let geometries = make_blas_geometry_descriptors(desc);
        let primitive_desc = MTL4PrimitiveAccelerationStructureDescriptor::new();
        primitive_desc.setGeometryDescriptors(Some(&geometries.array));
        let primitive_base = unsafe {
            &*(primitive_desc.as_ref() as *const MTL4PrimitiveAccelerationStructureDescriptor
                as *const objc2_metal::MTLAccelerationStructureDescriptor)
        };
        super::accel::set_accel_usage(primitive_base, desc.flags);

        let sizes = self
            .device
            .accelerationStructureSizesWithDescriptor(primitive_base);
        self.finalize_accel_structure(sizes, "BLAS")
    }

    pub fn create_tlas(&self, desc: &TlasDesc) -> RhiResult<AccelerationStructure> {
        use objc2_metal::MTL4InstanceAccelerationStructureDescriptor;

        let instance_desc = MTL4InstanceAccelerationStructureDescriptor::new();
        unsafe {
            instance_desc.setInstanceDescriptorBuffer(objc2_metal::MTL4BufferRange {
                bufferAddress: desc.instance_buffer.0,
                // The descriptor uses Metal's indirect instance layout; callers write it with
                // `Device::write_tlas_instance`.
                length: (desc.instance_count as u64) * self.tlas_instance_stride() as u64,
            });
            instance_desc.setInstanceCount(desc.instance_count as usize);
        }
        let instance_base = unsafe {
            &*(instance_desc.as_ref() as *const MTL4InstanceAccelerationStructureDescriptor
                as *const objc2_metal::MTLAccelerationStructureDescriptor)
        };
        super::accel::set_accel_usage(instance_base, desc.flags);

        let sizes = self
            .device
            .accelerationStructureSizesWithDescriptor(instance_base);
        self.finalize_accel_structure(sizes, "TLAS")
    }

    /// Native size of one TLAS instance descriptor. Metal's instance acceleration structure
    /// uses the *indirect* descriptor layout (references the BLAS by `gpuResourceID`).
    pub fn tlas_instance_stride(&self) -> usize {
        std::mem::size_of::<objc2_metal::MTLIndirectAccelerationStructureInstanceDescriptor>()
    }

    /// Encode `inst` into `dst` in Metal's native indirect instance-descriptor layout.
    /// `dst` must have room for `tlas_instance_stride()` bytes.
    ///
    /// `inst.acceleration_structure_reference` must be the BLAS `gpuResourceID` (`blas.gpu()`).
    pub fn write_tlas_instance(&self, dst: *mut u8, inst: &crate::types::TlasInstance) {
        use objc2_metal::{
            MTLAccelerationStructureInstanceOptions,
            MTLIndirectAccelerationStructureInstanceDescriptor, MTLPackedFloat3, MTLPackedFloat4x3,
            MTLResourceID,
        };

        // TlasInstance.transform is row-major 3x4 (transform[row][col]); Metal's packed 4x3
        // is column-major (columns[c] = (m[0][c], m[1][c], m[2][c])).
        let t = &inst.transform;
        let col = |c: usize| MTLPackedFloat3 {
            x: t[0][c],
            y: t[1][c],
            z: t[2][c],
        };
        let resource_id: MTLResourceID =
            unsafe { std::mem::transmute(inst.acceleration_structure_reference.0) };

        let instance_flags =
            InstanceFlags::from_bits_retain((inst.instance_sbt_offset_and_flags >> 24) as u8);
        let mut options = MTLAccelerationStructureInstanceOptions::empty();
        if instance_flags.contains(InstanceFlags::TRIANGLE_FACING_CULL_DISABLE) {
            options |= MTLAccelerationStructureInstanceOptions::DisableTriangleCulling;
        }
        if instance_flags.contains(InstanceFlags::TRIANGLE_FLIP_FACING) {
            options |=
                MTLAccelerationStructureInstanceOptions::TriangleFrontFacingWindingCounterClockwise;
        }
        if instance_flags.contains(InstanceFlags::FORCE_OPAQUE) {
            options |= MTLAccelerationStructureInstanceOptions::Opaque;
        }
        if instance_flags.contains(InstanceFlags::FORCE_NO_OPAQUE) {
            options |= MTLAccelerationStructureInstanceOptions::NonOpaque;
        }

        let desc = MTLIndirectAccelerationStructureInstanceDescriptor {
            transformationMatrix: MTLPackedFloat4x3 {
                columns: [col(0), col(1), col(2), col(3)],
            },
            options,
            mask: (inst.instance_custom_index_and_mask >> 24) & 0xFF,
            intersectionFunctionTableOffset: inst.instance_sbt_offset_and_flags & 0x00FF_FFFF,
            userID: inst.instance_custom_index_and_mask & 0x00FF_FFFF,
            accelerationStructureID: resource_id,
        };
        unsafe {
            std::ptr::write_unaligned(
                dst as *mut MTLIndirectAccelerationStructureInstanceDescriptor,
                desc,
            );
        }
    }

    /// Allocate the acceleration structure + scratch buffer for `sizes`, register both
    /// with the residency set, query the GPU resource ID, mint an `AccelerationStructureId`,
    /// and wrap into the public handle. Shared by `create_blas` / `create_tlas`.
    fn finalize_accel_structure(
        &self,
        sizes: objc2_metal::MTLAccelerationStructureSizes,
        label: &'static str,
    ) -> RhiResult<AccelerationStructure> {
        use super::accel::MetalAccelerationStructure;
        use objc2_metal::{MTLAccelerationStructure as _, MTLDevice, MTLResourceOptions};

        let accel = self
            .device
            .newAccelerationStructureWithSize(sizes.accelerationStructureSize)
            .ok_or_else(|| {
                RhiError::AllocationFailed(format!("Failed to allocate Metal {label}"))
            })?;
        let scratch = self
            .device
            .newBufferWithLength_options(
                sizes.buildScratchBufferSize,
                MTLResourceOptions::StorageModePrivate,
            )
            .ok_or_else(|| {
                RhiError::AllocationFailed(format!("Failed to allocate {label} scratch buffer"))
            })?;

        unsafe {
            let accel_alloc = &*(accel.as_ref()
                as *const ProtocolObject<dyn objc2_metal::MTLAccelerationStructure>
                as *const ProtocolObject<dyn MTLAllocation>);
            self.residency_set.addAllocation(accel_alloc);
            let scratch_alloc = &*(scratch.as_ref() as *const ProtocolObject<dyn MTLBuffer>
                as *const ProtocolObject<dyn MTLAllocation>);
            self.residency_set.addAllocation(scratch_alloc);
        }
        self.residency_dirty.set(true);

        let gpu_resource_id = accel.gpuResourceID().to_raw();

        let id = {
            let mut counter = self.accel_counter.borrow_mut();
            let next = *counter;
            *counter += 1;
            AccelerationStructureId(next)
        };

        Ok(AccelerationStructure {
            id,
            inner: AccelInner::Metal(Box::new(MetalAccelerationStructure {
                acceleration_structure: accel,
                gpu_resource_id,
                scratch_buffer: scratch,
                residency_set: self.residency_set.clone(),
                residency_dirty: self.residency_dirty.clone(),
            })),
        })
    }

    /// Build a fresh table slot. `pool` is the free list it returns to on drop; swapchain slots
    /// pass `None`.
    fn new_table_slot(
        &self,
        pool: Option<SharedTableSlotPool>,
        what: &str,
    ) -> RhiResult<Rc<MetalFrameTableSlot>> {
        let state = MetalTableState::new(
            self.device.as_ref(),
            self.residency_set.as_ref(),
            &self.residency_dirty,
        )?;
        let command_allocator = self.device.newCommandAllocator().ok_or_else(|| {
            RhiError::CommandBuffer(format!("Failed to create {what} MTL4CommandAllocator"))
        })?;
        let command_buffer = self.device.newCommandBuffer().ok_or_else(|| {
            RhiError::CommandBuffer(format!("Failed to create {what} MTL4CommandBuffer"))
        })?;
        Ok(Rc::new(MetalFrameTableSlot {
            state: Rc::new(RefCell::new(state)),
            command_allocator,
            command_buffer,
            argument_table: RefCell::new(None),
            in_use: Cell::new(false),
            pool,
        }))
    }

    fn acquire_frame_table_slot(&self, frame_index: usize) -> RhiResult<Rc<MetalFrameTableSlot>> {
        let mut slots = self.frame_table_slots.borrow_mut();
        let slot_entry = slots
            .get_mut(frame_index)
            .ok_or_else(|| RhiError::CommandBuffer("invalid Metal frame index".into()))?;
        let slot = if let Some(slot) = slot_entry {
            slot.clone()
        } else {
            let slot = self.new_table_slot(None, "frame")?;
            *slot_entry = Some(slot.clone());
            slot
        };
        if slot.in_use.replace(true) {
            return Err(RhiError::CommandBuffer(
                "Metal frame table slot is still in use".into(),
            ));
        }
        slot.command_allocator.reset();
        Ok(slot)
    }

    /// Take a generic table slot from the free list, or build one when empty. Without this,
    /// `create_command_buffer` rebuilt an allocator, command buffer, argument table and 1 MiB root
    /// ring on every call — Vulkan pools all of its command buffers, so this keeps parity.
    fn acquire_pooled_table_slot(&self) -> RhiResult<Rc<MetalFrameTableSlot>> {
        let pooled = self.table_slot_pool.borrow_mut().pop();
        let slot = match pooled {
            Some(slot) => slot,
            None => self.new_table_slot(Some(self.table_slot_pool.clone()), "pooled")?,
        };
        debug_assert!(!slot.in_use.get(), "pooled slot handed out while in use");
        slot.in_use.set(true);
        slot.command_allocator.reset();
        Ok(slot)
    }

    /// Wrap an acquired slot in a recording command buffer, releasing it again on failure so a
    /// failed create never strands it as in-use.
    fn create_command_buffer_in_slot(
        &self,
        slot: Rc<MetalFrameTableSlot>,
    ) -> RhiResult<CommandBuffer> {
        let mtl_cmd = MetalCommandBuffer::new(
            slot.command_buffer.clone(),
            slot.command_allocator.clone(),
            self.device.clone(),
            self.residency_set.clone(),
            self.residency_dirty.clone(),
            self.textures.clone(),
            self.samplers.clone(),
            self.texture_generation.clone(),
            self.sampler_generation.clone(),
            self.allocations.clone(),
            self.active_recordings.clone(),
            Some(slot.clone()),
            self.mdi_icb_pipeline.clone(),
        );
        let mtl_cmd = match mtl_cmd {
            Ok(cmd) => cmd,
            Err(error) => {
                slot.in_use.set(false);
                if let Some(pool) = slot.pool.clone() {
                    pool.borrow_mut().push(slot);
                }
                return Err(error);
            }
        };

        Ok(CommandBuffer {
            inner: crate::command::CommandBufferInner::Metal(Box::new(mtl_cmd)),
        })
    }

    pub fn create_command_buffer(&self) -> RhiResult<CommandBuffer> {
        let slot = self.acquire_pooled_table_slot()?;
        self.create_command_buffer_in_slot(slot)
    }

    pub fn create_command_buffer_for_swapchain(
        &self,
        swapchain: &Swapchain,
        frame_index: usize,
    ) -> RhiResult<CommandBuffer> {
        let frame_table_slot = self.acquire_frame_table_slot(frame_index)?;
        let mut cmd_buf = self.create_command_buffer_in_slot(frame_table_slot)?;

        match (&mut cmd_buf.inner, &swapchain.inner) {
            (crate::command::CommandBufferInner::Metal(mtl_cmd), SwapchainInner::Metal(sc)) => {
                mtl_cmd.drawable_slot = Some(sc.drawable.clone());
            }
            #[allow(unreachable_patterns)]
            _ => unreachable!("wrong backend"),
        }

        Ok(cmd_buf)
    }

    pub fn create_timeline_semaphore(&self, initial_value: u64) -> RhiResult<TimelineSemaphore> {
        let event = self
            .device
            .newSharedEvent()
            .ok_or_else(|| RhiError::SyncError("Failed to create MTLSharedEvent".into()))?;
        event.setSignaledValue(initial_value);

        Ok(TimelineSemaphore {
            inner: crate::sync::TimelineSemaphoreInner::Metal(Box::new(MetalTimelineSemaphore {
                event,
            })),
        })
    }

    #[allow(irrefutable_let_patterns)]
    pub fn destroy_buffer(&self, buffer: GpuBuffer) {
        match buffer.inner {
            #[cfg(feature = "metal")]
            GpuBufferInner::Metal(mtl) => {
                {
                    let mut allocations = self.allocations.borrow_mut();
                    allocations.remove(&mtl.gpu_address().0);
                }
                if let Some(mapped_ptr) = mtl.mapped_ptr() {
                    self.mapped_allocations
                        .borrow_mut()
                        .remove(&(mapped_ptr as usize));
                }
                if let QueueInner::Metal(queue) = &self.rhi_queue.inner {
                    queue.retire_resource(MetalRetiredResource::Buffer(mtl));
                }
            }
            #[cfg(feature = "vulkan")]
            GpuBufferInner::Vulkan(_) => {}
        }
    }

    #[allow(irrefutable_let_patterns)]
    pub fn destroy_texture(&self, texture: Texture) {
        let retired = {
            let mut textures = self.textures.borrow_mut();
            let idx = texture.id.0 as usize;
            if idx < textures.len() {
                textures[idx].take()
            } else {
                None
            }
        };
        if let Some(tex) = retired {
            if let Some(is_view) = self
                .texture_view_flags
                .borrow_mut()
                .get_mut(texture.id.0 as usize)
            {
                *is_view = false;
            }
            self.texture_generation
                .set(self.texture_generation.get().wrapping_add(1));
            if let QueueInner::Metal(queue) = &self.rhi_queue.inner {
                queue.retire_resource(MetalRetiredResource::Texture {
                    id: texture.id,
                    texture: tex,
                });
            }
        }
    }

    #[allow(irrefutable_let_patterns)]
    pub fn destroy_texture_view(&self, id: TextureId) {
        // Claim the view in one borrow, mirroring the Vulkan path.
        let was_view = self
            .texture_view_flags
            .borrow_mut()
            .get_mut(id.0 as usize)
            .is_some_and(|flag| std::mem::replace(flag, false));
        if !was_view {
            return;
        }
        let retired = self
            .textures
            .borrow_mut()
            .get_mut(id.0 as usize)
            .and_then(Option::take);
        if let Some(texture) = retired {
            self.texture_generation
                .set(self.texture_generation.get().wrapping_add(1));
            if let QueueInner::Metal(queue) = &self.rhi_queue.inner {
                queue.retire_resource(MetalRetiredResource::Texture { id, texture });
            }
        }
    }

    #[allow(irrefutable_let_patterns)]
    pub fn destroy_sampler(&self, sampler: Sampler) {
        let sampler_id = sampler.id();
        let retired = {
            let mut samplers = self.samplers.borrow_mut();
            let idx = sampler_id.0 as usize;
            if idx < samplers.len() {
                samplers[idx].take()
            } else {
                None
            }
        };
        if let Some(sampler) = retired {
            self.sampler_generation
                .set(self.sampler_generation.get().wrapping_add(1));
            if let QueueInner::Metal(queue) = &self.rhi_queue.inner {
                queue.retire_resource(MetalRetiredResource::Sampler {
                    id: sampler_id,
                    sampler,
                });
            }
        }
    }

    pub fn create_sampled_view(
        &self,
        source: &Texture,
        view: &crate::texture::TextureViewDesc,
    ) -> RhiResult<TextureId> {
        self.create_view_internal(source, view)
    }

    pub fn create_storage_view(
        &self,
        source: &Texture,
        view: &crate::texture::TextureViewDesc,
    ) -> RhiResult<TextureId> {
        self.create_view_internal(source, view)
    }

    /// Value to store in a [`TextureHandle`](crate::TextureHandle) root field for sampled view
    /// `id`. On Metal a `DescriptorHandle<Texture2D>` is the texture's `gpuResourceID`, which the
    /// shader uses directly.
    pub fn bindless_texture_handle(&self, id: TextureId) -> GpuAddress {
        let textures = self.textures.borrow();
        let raw = textures
            .get(id.0 as usize)
            .and_then(|t| t.as_ref())
            .map(|t| t.gpuResourceID().to_raw())
            .unwrap_or(0);
        GpuAddress(raw)
    }

    /// Value to store in a [`SamplerHandle`](crate::SamplerHandle) root field for sampler `id`.
    pub fn bindless_sampler_handle(&self, id: crate::types::SamplerId) -> GpuAddress {
        let samplers = self.samplers.borrow();
        let raw = samplers
            .get(id.0 as usize)
            .and_then(|s| s.as_ref())
            .map(|s| s.gpuResourceID().to_raw())
            .unwrap_or(0);
        GpuAddress(raw)
    }

    pub fn create_query_pool(&self, count: u32) -> RhiResult<QueryPool> {
        use objc2_foundation::NSRange;

        let desc = MTL4CounterHeapDescriptor::new();
        desc.setType(MTL4CounterHeapType::Timestamp);
        // SAFETY: count is the heap entry count; not bounds-checked by the API.
        unsafe { desc.setCount(count as usize) };
        let heap = self
            .device
            .newCounterHeapWithDescriptor_error(&desc)
            .map_err(|e| RhiError::Backend(format!("newCounterHeapWithDescriptor: {e}")))?;
        // Invalidate the new heap so unwritten slots resolve to zero.
        unsafe {
            heap.invalidateCounterRange(NSRange {
                location: 0,
                length: count as usize,
            })
        };
        Ok(QueryPool {
            inner: QueryPoolInner::Metal(MetalQueryPool { heap }),
            count,
        })
    }

    pub fn destroy_query_pool(&self, _pool: QueryPool) {}

    pub fn timestamp_period_ns(&self) -> f64 {
        let freq = self.device.queryTimestampFrequency();
        if freq == 0 { 0.0 } else { 1.0e9 / freq as f64 }
    }

    pub fn read_timestamps(&self, pool: &QueryPool) -> RhiResult<Vec<u64>> {
        use objc2::rc::autoreleasepool;
        use objc2_foundation::NSRange;

        // MTL4CounterHeap::resolveCounterRange returns a newly allocated autoreleased NSData.
        // The window event loop does not guarantee a pool around every render callback, so keep
        // the resolve and byte copy inside an explicit pool; otherwise the HUD's App Memory grows
        // once per frame even though Metal residency remains flat.
        autoreleasepool(|_| {
            let heap = match &pool.inner {
                QueryPoolInner::Metal(p) => &p.heap,
                #[allow(unreachable_patterns)]
                _ => unreachable!("query pool backend does not match device backend"),
            };
            // The writing frame has completed, so resolve the timestamp heap directly.
            let range = NSRange {
                location: 0,
                length: pool.count as usize,
            };
            let data = unsafe { heap.resolveCounterRange(range) }
                .ok_or_else(|| RhiError::Backend("resolveCounterRange returned nil".into()))?;
            let mut out = vec![0u64; pool.count as usize];
            // `NSData::as_bytes_unchecked` is safe here: `data` remains alive and immutable for
            // the whole copy, and Metal returns exactly one packed u64 per counter slot.
            for (slot, chunk) in out
                .iter_mut()
                .zip(unsafe { data.as_bytes_unchecked() }.chunks_exact(8))
            {
                *slot =
                    u64::from_ne_bytes(chunk.try_into().expect("chunks_exact(8) yields 8 bytes"));
            }
            Ok(out)
        })
    }

    fn create_view_internal(
        &self,
        source: &Texture,
        view: &crate::texture::TextureViewDesc,
    ) -> RhiResult<TextureId> {
        use crate::texture::{ALL_LAYERS, ALL_MIPS};
        use objc2_foundation::NSRange;

        let textures_borrow = self.textures.borrow();
        let src_texture = textures_borrow
            .get(source.id.0 as usize)
            .and_then(|t| t.as_ref())
            .ok_or_else(|| {
                RhiError::TextureCreation("create texture view: invalid source TextureId".into())
            })?
            .clone();
        drop(textures_borrow);

        let src_format = super::texture::format_to_mtl(view.format.unwrap_or(source.desc().format));
        let src_type = match source.desc().dimension {
            TextureDimension::D1 => objc2_metal::MTLTextureType::Type1D,
            TextureDimension::D2 => {
                if source.desc().array_layers > 1 {
                    objc2_metal::MTLTextureType::Type2DArray
                } else {
                    objc2_metal::MTLTextureType::Type2D
                }
            }
            TextureDimension::D2Array => objc2_metal::MTLTextureType::Type2DArray,
            TextureDimension::D3 => objc2_metal::MTLTextureType::Type3D,
            TextureDimension::Cube => objc2_metal::MTLTextureType::TypeCube,
            TextureDimension::CubeArray => objc2_metal::MTLTextureType::TypeCubeArray,
        };

        let src_mips = source.desc().mip_levels;
        let src_layers = source.desc().array_layers;

        let mip_start = view.base_mip as usize;
        let mip_count = if view.mip_count == ALL_MIPS {
            (src_mips as usize).saturating_sub(mip_start)
        } else {
            view.mip_count as usize
        };
        let layer_start = view.base_layer as usize;
        let layer_count = if view.layer_count == ALL_LAYERS {
            (src_layers as usize).saturating_sub(layer_start)
        } else {
            view.layer_count as usize
        };

        let level_range = NSRange::new(mip_start, mip_count);
        let slice_range = NSRange::new(layer_start, layer_count);

        let view_texture = unsafe {
            src_texture
                .newTextureViewWithPixelFormat_textureType_levels_slices(
                    src_format,
                    src_type,
                    level_range,
                    slice_range,
                )
                .ok_or_else(|| {
                    RhiError::TextureCreation("Metal texture view creation failed".into())
                })?
        };

        // Views share the source allocation but still need residency tracking.
        let allocation = unsafe {
            &*(view_texture.as_ref() as *const ProtocolObject<dyn MTLTexture>
                as *const ProtocolObject<dyn MTLAllocation>)
        };
        self.residency_set.addAllocation(allocation);
        self.residency_dirty.set(true);

        let id = match self.allocate_texture_id() {
            Ok(id) => id,
            Err(err) => {
                self.residency_set.removeAllocation(allocation);
                self.residency_dirty.set(true);
                return Err(err);
            }
        };
        let idx = id.0 as usize;
        let mut textures = self.textures.borrow_mut();
        if textures.len() <= idx {
            textures.resize_with(idx + 1, || None);
        }
        textures[idx] = Some(view_texture);
        drop(textures);
        let mut view_flags = self.texture_view_flags.borrow_mut();
        if view_flags.len() <= idx {
            view_flags.resize(idx + 1, false);
        }
        view_flags[idx] = true;
        self.texture_generation
            .set(self.texture_generation.get().wrapping_add(1));

        Ok(id)
    }
}

fn filter_to_mtl(f: crate::types::FilterMode) -> objc2_metal::MTLSamplerMinMagFilter {
    match f {
        FilterMode::Nearest => objc2_metal::MTLSamplerMinMagFilter::Nearest,
        FilterMode::Linear => objc2_metal::MTLSamplerMinMagFilter::Linear,
    }
}

fn mip_filter_to_mtl(f: crate::types::FilterMode) -> objc2_metal::MTLSamplerMipFilter {
    match f {
        FilterMode::Nearest => objc2_metal::MTLSamplerMipFilter::Nearest,
        FilterMode::Linear => objc2_metal::MTLSamplerMipFilter::Linear,
    }
}

fn address_to_mtl(a: crate::types::AddressMode) -> objc2_metal::MTLSamplerAddressMode {
    match a {
        AddressMode::Repeat => objc2_metal::MTLSamplerAddressMode::Repeat,
        AddressMode::MirroredRepeat => objc2_metal::MTLSamplerAddressMode::MirrorRepeat,
        AddressMode::ClampToEdge => objc2_metal::MTLSamplerAddressMode::ClampToEdge,
        AddressMode::ClampToBorder => objc2_metal::MTLSamplerAddressMode::ClampToBorderColor,
    }
}

fn compare_op_to_mtl(op: CompareOp) -> objc2_metal::MTLCompareFunction {
    match op {
        CompareOp::Never => objc2_metal::MTLCompareFunction::Never,
        CompareOp::Less => objc2_metal::MTLCompareFunction::Less,
        CompareOp::Equal => objc2_metal::MTLCompareFunction::Equal,
        CompareOp::LessOrEqual => objc2_metal::MTLCompareFunction::LessEqual,
        CompareOp::Greater => objc2_metal::MTLCompareFunction::Greater,
        CompareOp::NotEqual => objc2_metal::MTLCompareFunction::NotEqual,
        CompareOp::GreaterOrEqual => objc2_metal::MTLCompareFunction::GreaterEqual,
        CompareOp::Always => objc2_metal::MTLCompareFunction::Always,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapped_pointer_lookup_uses_predecessor_and_respects_end() {
        let mut allocations = BTreeMap::new();
        allocations.insert(
            0x1000,
            MappedAllocation {
                gpu_base: GpuAddress(0x8000),
                size: 0x20,
            },
        );
        allocations.insert(
            0x2000,
            MappedAllocation {
                gpu_base: GpuAddress(0x9000),
                size: 0x10,
            },
        );

        assert_eq!(
            resolve_mapped_pointer(&allocations, 0x100f),
            Some(GpuAddress(0x800f))
        );
        assert_eq!(
            resolve_mapped_pointer(&allocations, 0x200f),
            Some(GpuAddress(0x900f))
        );
        assert_eq!(resolve_mapped_pointer(&allocations, 0x1020), None);
        assert_eq!(resolve_mapped_pointer(&allocations, 0x1fff), None);
    }
}
