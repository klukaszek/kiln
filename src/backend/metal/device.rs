use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap};
use std::ptr::NonNull;
use std::rc::Rc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_core_foundation::CGSize;
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4CommandBuffer, MTL4CommandQueue, MTL4Compiler, MTL4CompilerDescriptor,
    MTL4ComputePipelineDescriptor, MTL4LibraryFunctionDescriptor, MTL4PipelineDescriptor,
    MTL4PipelineOptions, MTL4ShaderReflection, MTLAllocation, MTLBinding, MTLBindingType,
    MTLBuffer, MTLCompileOptions, MTLComputePipelineState, MTLCreateSystemDefaultDevice,
    MTLCullMode, MTLDevice, MTLDrawable, MTLEvent, MTLHeap, MTLHeapDescriptor,
    MTLHeapType, MTLLanguageVersion, MTLLibrary, MTLPixelFormat, MTLRenderPipelineState,
    MTLResidencySet, MTLResidencySetDescriptor, MTLResourceOptions, MTLSamplerDescriptor,
    MTLSamplerState, MTLSharedEvent, MTLStorageMode, MTLTexture, MTLTextureDescriptor,
    MTLTextureType, MTLTextureUsage as MtlTextureUsage, MTLWinding,
};
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};
use raw_window_handle::RawWindowHandle;

use crate::accel::{AccelInner, AccelerationStructure};
use crate::command::{CommandBuffer, SignalOp, SignalValueDesc, WaitOp, WaitValueDesc};
use crate::device::{BindlessMode, DeviceDesc};
use crate::error::{RhiError, RhiResult};
use crate::memory::{BufferDesc, MemoryType, GpuBuffer, GpuBufferInner};
use crate::pipeline::*;
use crate::queue::{Queue, QueueInner, SubmitDesc};
use crate::sampler::{Sampler, SamplerDesc};
use crate::shader::{ShaderModule, ShaderModuleDesc};
use crate::surface::{Surface, SurfaceDesc, SurfaceInner};
use crate::swapchain::{AcquiredImage, Swapchain, SwapchainDesc, SwapchainInner};
use crate::sync::{TimelineSemaphore, TimelineSemaphoreInner};
use crate::texture::{Texture, TextureDesc, TextureSizeAlign, TextureUsage};
use crate::types::*;

use super::command::MetalCommandBuffer;
use super::memory::MetalBuffer;
use super::pipeline::{MetalComputePso, MetalGraphicsPso};
use super::shader::MetalShaderModule;
use super::surface::MetalSurface;
use super::swapchain::MetalSwapchain;
use super::sync::MetalTimelineSemaphore;
use super::texture::{format_to_mtl, mtl_to_format};

type FrameFenceValues = Rc<RefCell<[u64; MAX_FRAMES_IN_FLIGHT]>>;
type InFlightFrameCommands = Rc<RefCell<Vec<Option<MetalCommandBuffer>>>>;
type PendingSubmissions = Rc<RefCell<Vec<(u64, MetalCommandBuffer)>>>;
pub(crate) type SharedTextures = Rc<RefCell<Vec<Option<Retained<ProtocolObject<dyn MTLTexture>>>>>>;
pub(crate) type SharedSamplers = Rc<RefCell<Vec<Retained<ProtocolObject<dyn MTLSamplerState>>>>>;
/// Buffer allocations keyed by GPU base address, enabling O(log n) address->buffer
/// resolution for blit copies and indirect draws instead of a linear scan.
pub(crate) type SharedAllocations = Rc<RefCell<BTreeMap<u64, BufferAllocation>>>;
type ValueSyncMap = RefCell<HashMap<u64, MetalValueSyncState>>;
type MetalEventWaits = Vec<(Retained<ProtocolObject<dyn MTLSharedEvent>>, u64)>;

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
    pub heap: Option<Retained<ProtocolObject<dyn MTLHeap>>>,
    pub mapped_ptr: Option<*mut u8>,
}

pub struct MetalDevice {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    rhi_queue: Queue,
    residency_set: Retained<ProtocolObject<dyn MTLResidencySet>>,
    residency_dirty: Rc<Cell<bool>>,
    textures: SharedTextures,
    samplers: SharedSamplers,
    allocations: SharedAllocations,
    /// Per-frame fence values for swapchain acquisition.
    frame_fence_values: FrameFenceValues,
    /// Shared event for per-frame synchronization.
    frame_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
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

    fn ensure_value_sync_state<'a>(
        &self,
        value_ptr: GpuAddress,
        map: &'a mut HashMap<u64, MetalValueSyncState>,
    ) -> RhiResult<&'a mut MetalValueSyncState> {
        let key = value_ptr.0;
        if let std::collections::hash_map::Entry::Vacant(entry) = map.entry(key) {
            let event = self.device.newSharedEvent().ok_or_else(|| {
                RhiError::SyncError(format!("Failed to create Metal value event for {:#x}", key))
            })?;
            event.setSignaledValue(0);
            entry.insert(MetalValueSyncState {
                event,
                scheduled_value: 0,
            });
        }
        Ok(map
            .get_mut(&key)
            .expect("value sync state should exist after insertion"))
    }

    fn collect_value_waits(&self, waits: &[WaitValueDesc]) -> RhiResult<MetalEventWaits> {
        let mut map = self.value_sync.borrow_mut();
        let mut merged: HashMap<u64, (Retained<ProtocolObject<dyn MTLSharedEvent>>, u64)> =
            HashMap::new();
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
            merged
                .entry(wait.value_ptr.0)
                .and_modify(|(_, value)| *value = (*value).max(required))
                .or_insert((state.event.clone(), required));
        }
        Ok(merged.into_values().collect())
    }

    fn collect_value_signals(&self, signals: &[SignalValueDesc]) -> RhiResult<MetalEventWaits> {
        let mut map = self.value_sync.borrow_mut();
        let mut merged: HashMap<u64, (Retained<ProtocolObject<dyn MTLSharedEvent>>, u64)> =
            HashMap::new();
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
            merged.insert(signal.value_ptr.0, (state.event.clone(), target));
        }
        Ok(merged.into_values().collect())
    }

    pub fn submit_with_desc(
        &self,
        cmd: MetalCommandBuffer,
        desc: &SubmitDesc<'_>,
    ) -> RhiResult<()> {
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
        self.pending_submissions.borrow_mut().push((value, cmd));
        Ok(())
    }

    pub fn submit_frame(
        &self,
        cmd: MetalCommandBuffer,
        sc: &MetalSwapchain,
        frame_index: usize,
        _image_index: u32,
    ) -> RhiResult<()> {
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

        let mut cmd = cmd;
        cmd.finish();
        self.commit_single(&cmd.command_buffer);

        for (event, value) in value_signals {
            self.queue
                .signalEvent_value(shared_event_as_event(&event), value);
        }

        // Signal a per-frame fence value after GPU work completes.
        let value = self.next_fence_value();
        self.frame_fence_values.borrow_mut()[frame_index] = value;
        self.queue
            .signalEvent_value(shared_event_as_event(&self.frame_event), value);

        // Present the drawable after committing.
        if let Some(drawable) = sc.current_drawable.borrow().as_ref() {
            self.queue.signalDrawable(drawable);
            drawable.present();
        }

        // Clear drawable state
        let _ = sc.current_drawable.borrow_mut().take();
        let _ = sc.current_drawable_texture.borrow_mut().take();

        // Keep the command buffer and associated resources alive until this frame slot completes.
        let mut frame_cmds = self.in_flight_frame_commands.borrow_mut();
        if frame_index >= frame_cmds.len() {
            return Err(RhiError::Backend(
                "invalid frame index for Metal queue submission".into(),
            ));
        }
        if frame_cmds[frame_index].is_some() {
            log::warn!("Overwriting in-flight Metal frame command before completion");
        }
        frame_cmds[frame_index] = Some(cmd);

        Ok(())
    }

    pub fn acquire_image(
        &self,
        sc: &MetalSwapchain,
        frame_index: usize,
    ) -> RhiResult<AcquiredImage> {
        self.reclaim_completed_submissions();

        // Wait for previous GPU work on this frame slot to finish.
        let value = self.frame_fence_values.borrow()[frame_index];
        if value != 0 {
            let _ = self
                .frame_event
                .waitUntilSignaledValue_timeoutMS(value, u64::MAX);
        }
        if frame_index < self.in_flight_frame_commands.borrow().len() {
            self.in_flight_frame_commands.borrow_mut()[frame_index] = None;
        }

        let drawable = sc
            .layer
            .nextDrawable()
            .ok_or_else(|| RhiError::SwapchainCreation("nextDrawable returned nil".into()))?;

        // Store the drawable's texture for rendering
        let texture = drawable.texture();
        *sc.current_drawable_texture.borrow_mut() = Some(texture);

        // Store the drawable for presentation
        // CAMetalDrawable conforms to MTLDrawable, so we need to upcast
        let drawable_proto: Retained<ProtocolObject<dyn MTLDrawable>> =
            ProtocolObject::from_retained(drawable);
        *sc.current_drawable.borrow_mut() = Some(drawable_proto);

        Ok(AcquiredImage {
            index: 0, // Metal only has one "current" drawable
            format: sc.format,
            width: sc.extent[0],
            height: sc.extent[1],
        })
    }

    pub fn present(
        &self,
        sc: &MetalSwapchain,
        _image_index: u32,
        _frame_index: usize,
    ) -> RhiResult<()> {
        // Present is handled in submit_frame via presentDrawable on the command buffer.
        // If called separately, we just drop the drawable (it was already presented).
        let _ = sc.current_drawable.borrow_mut().take();
        let _ = sc.current_drawable_texture.borrow_mut().take();
        Ok(())
    }

    pub fn wait_idle(&self) {
        let value = self.next_fence_value();
        self.queue
            .signalEvent_value(shared_event_as_event(&self.frame_event), value);
        let _ = self
            .frame_event
            .waitUntilSignaledValue_timeoutMS(value, u64::MAX);
        self.pending_submissions.borrow_mut().clear();
        for slot in self.in_flight_frame_commands.borrow_mut().iter_mut() {
            *slot = None;
        }
    }

    fn next_fence_value(&self) -> u64 {
        let value = self.frame_fence_next.get().wrapping_add(1);
        self.frame_fence_next.set(value);
        value
    }

    fn reclaim_completed_submissions(&self) {
        let completed = self.frame_event.signaledValue();
        self.pending_submissions
            .borrow_mut()
            .retain(|(value, _)| *value > completed);
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

// MSL forbids `command_buffer` as a direct `[[buffer(N)]]` kernel parameter; it must be
// referenced through an argument-buffer struct (a `command_buffer` field at `[[id(0)]]`).
// The CPU binds an 8-byte argument buffer holding the ICB's gpuResourceID at slot 3.
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

// Encodes non-indexed multi-draw indirect into an MTLIndirectCommandBuffer.
// Per NoGraphicsApi.md (lines 904-907, 1124, 59, 1152) the architecture's MDI
// is fully GPU-driven and non-indexed at the hardware level: vertex shaders
// perform programmable index fetch from a pointer inside per-draw root data,
// selected by [[draw_id]] + stride.
//
// The ICB is bound directly via Metal 4's setResource_atBufferIndex (no argument buffer).
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
    let function = library
        .newFunctionWithName(&function_name)
        .ok_or_else(|| {
            RhiError::PipelineCreation(
                "Metal MDI ICB encoder function was not found".into(),
            )
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

        // Keep the residency set active on the command queue.
        queue.addResidencySet(&residency_set);

        let frame_event = device
            .newSharedEvent()
            .ok_or_else(|| RhiError::DeviceCreation("Failed to create MTLSharedEvent".into()))?;
        let frame_fence_values: FrameFenceValues =
            Rc::new(RefCell::new([0u64; MAX_FRAMES_IN_FLIGHT]));
        let frame_fence_next = Rc::new(Cell::new(0u64));
        let in_flight_frame_commands: InFlightFrameCommands = Rc::new(RefCell::new(
            std::iter::repeat_with(|| None)
                .take(MAX_FRAMES_IN_FLIGHT)
                .collect(),
        ));
        let pending_submissions: PendingSubmissions = Rc::new(RefCell::new(Vec::new()));

        let residency_dirty = Rc::new(Cell::new(false));
        let metal_queue = MetalQueue {
            queue: queue.clone(),
            device: device.clone(),
            residency_set: residency_set.clone(),
            residency_dirty: residency_dirty.clone(),
            frame_fence_values: frame_fence_values.clone(),
            frame_fence_next: frame_fence_next.clone(),
            frame_event: frame_event.clone(),
            in_flight_frame_commands,
            pending_submissions,
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
            rhi_queue,
            residency_set,
            residency_dirty,
            textures: Rc::new(RefCell::new(Vec::new())),
            samplers: Rc::new(RefCell::new(Vec::new())),
            allocations: Rc::new(RefCell::new(BTreeMap::new())),
            frame_fence_values,
            frame_event,
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
    }

    pub fn wait_for_frame(&self, frame_index: usize) {
        let value = self.frame_fence_values.borrow()[frame_index];
        if value != 0 {
            let _ = self
                .frame_event
                .waitUntilSignaledValue_timeoutMS(value, u64::MAX);
        }
        // `match` (not `if let`) so single-backend builds, where `QueueInner` has only the
        // Metal variant, don't see an irrefutable pattern.
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

                // Create CAMetalLayer
                let layer = CAMetalLayer::new();
                layer.setDevice(Some(&self.device));
                layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm_sRGB);
                layer.setFramebufferOnly(true);

                // Set the layer on the view
                // [view setWantsLayer:YES]
                let _: () = msg_send![ns_view, setWantsLayer: Bool::YES];
                // [view setLayer:layer]
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

        // Configure the layer
        let mtl_format = format_to_mtl(desc.format);
        layer.setPixelFormat(mtl_format);
        layer.setDrawableSize(CGSize {
            width: desc.width as f64,
            height: desc.height as f64,
        });

        let format = mtl_to_format(layer.pixelFormat());

        // Create depth texture
        let depth_texture = self.create_depth_texture(desc.width, desc.height)?;

        Ok(Swapchain {
            inner: SwapchainInner::Metal(MetalSwapchain {
                layer: layer.clone(),
                format,
                extent: [desc.width, desc.height],
                depth_texture,
                current_drawable: RefCell::new(None),
                current_drawable_texture: RefCell::new(None),
            }),
        })
    }

    pub fn recreate_swapchain(
        &self,
        swapchain: &mut Swapchain,
        desc: &SwapchainDesc,
    ) -> RhiResult<()> {
        match &mut swapchain.inner {
            SwapchainInner::Metal(sc) => {
                sc.layer.setDrawableSize(CGSize {
                    width: desc.width as f64,
                    height: desc.height as f64,
                });
                sc.extent = [desc.width, desc.height];
                sc.depth_texture = self.create_depth_texture(desc.width, desc.height)?;
                Ok(())
            }
            #[allow(unreachable_patterns)]
            _ => unreachable!("wrong backend"),
        }
    }

    pub fn create_buffer(&self, desc: &BufferDesc) -> RhiResult<GpuBuffer> {
        let options = match desc.memory {
            MemoryType::Default | MemoryType::Readback => MTLResourceOptions::StorageModeShared,
            MemoryType::GpuOnly => MTLResourceOptions::StorageModePrivate,
        };

        let buffer_size = self
            .device
            .heapBufferSizeAndAlignWithLength_options(desc.size as usize, options);
        let heap_desc = MTLHeapDescriptor::new();
        heap_desc.setType(MTLHeapType::Placement);
        heap_desc.setSize(buffer_size.size);
        heap_desc.setResourceOptions(options);

        let heap = self
            .device
            .newHeapWithDescriptor(&heap_desc)
            .ok_or_else(|| RhiError::BufferCreation("Metal heap allocation failed".into()))?;

        let buffer =
            unsafe { heap.newBufferWithLength_options_offset(desc.size as usize, options, 0) }
                .ok_or_else(|| {
                    RhiError::BufferCreation("Metal placed buffer allocation failed".into())
                })?;

        // Register for residency (Metal 4 pointer model).
        let allocation = unsafe {
            &*(buffer.as_ref() as *const ProtocolObject<dyn MTLBuffer>
                as *const ProtocolObject<dyn MTLAllocation>)
        };
        self.residency_set.addAllocation(allocation);
        self.residency_dirty.set(true);

        if let Some(label) = &desc.label {
            use objc2_metal::MTLResource;
            let ns_label = NSString::from_str(label);
            buffer.setLabel(Some(&ns_label));
        }

        let is_shared = matches!(desc.memory, MemoryType::Default | MemoryType::Readback);

        let metal_buffer = MetalBuffer {
            buffer,
            heap: Some(heap),
            size: desc.size,
            is_shared,
        };

        {
            let mut allocations = self.allocations.borrow_mut();
            allocations.insert(
                metal_buffer.gpu_address().0,
                BufferAllocation {
                    base: metal_buffer.gpu_address(),
                    size: metal_buffer.size,
                    buffer: metal_buffer.buffer.clone(),
                    heap: metal_buffer.heap.clone(),
                    mapped_ptr: metal_buffer.mapped_ptr(),
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
        let allocations = self.allocations.borrow();
        for alloc in allocations.values() {
            if let Some(mapped) = alloc.mapped_ptr {
                let base = mapped as usize;
                let end = base + alloc.size as usize;
                if ptr >= base && ptr < end {
                    let offset = (ptr - base) as u64;
                    return Some(GpuAddress(alloc.base.0 + offset));
                }
            }
        }
        None
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
                .filter(|alloc| texture_gpu.0 < alloc.base.0 + alloc.size)
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
            if offset + size_align.size as u64 > alloc.size {
                return Err(RhiError::TextureCreation(format!(
                    "texture allocation address 0x{:x} has {} bytes available, needs {}",
                    texture_gpu.0,
                    alloc.size - offset,
                    size_align.size
                )));
            }
            let heap = alloc.heap.clone().ok_or_else(|| {
                RhiError::TextureCreation(
                    "texture allocation is not backed by a Metal placement heap".into(),
                )
            })?;

            (heap, offset)
        };

        let texture =
            unsafe { heap.newTextureWithDescriptor_offset(&mtl_desc, heap_offset as usize) }
                .ok_or_else(|| {
                    RhiError::TextureCreation("Metal placed texture allocation failed".into())
                })?;

        // Register for residency (Metal 4 pointer model).
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

        // Add to texture tracking
        let mut textures = self.textures.borrow_mut();
        let id = TextureId(textures.len() as u32);
        textures.push(Some(texture.clone()));
        drop(textures);

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

        let mut samplers = self.samplers.borrow_mut();
        let id = SamplerId(samplers.len() as u32);
        samplers.push(sampler.clone());
        drop(samplers);

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
            inner: crate::shader::ShaderModuleInner::Metal(MetalShaderModule {
                library,
                entry_point: desc.entry_point.to_string(),
            }),
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
        for target in &desc.color_targets {
            color_formats.push(format_to_mtl(target.format));
        }

        let sample_count = match desc.sample_count {
            SampleCount::S1 => 1,
            SampleCount::S2 => 2,
            SampleCount::S4 => 4,
            SampleCount::S8 => 8,
            SampleCount::S16 => 16,
        };

        // Depth/stencil formats stored on the PSO for render-pass construction;
        // MTL4RenderPipelineDescriptor does not accept them at compile time.
        let depth_format_mtl = desc
            .depth_format
            .map(format_to_mtl)
            .unwrap_or(MTLPixelFormat::Invalid);
        let stencil_format_mtl = desc
            .stencil_format
            .map(format_to_mtl)
            .unwrap_or(MTLPixelFormat::Invalid);

        let pipeline_state = MetalGraphicsPso::compile_pipeline_state(
            compiler.as_ref(),
            vert_module.library.as_ref(),
            &vert_module.entry_point,
            frag_module.library.as_ref(),
            &frag_module.entry_point,
            &color_formats,
            sample_count,
            desc.alpha_to_coverage,
            &BlendState::default(),
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
            // Metal has no native TriangleFan. The caller must rewrite indices to TriangleList
            // before submission. We panic here to surface the mistake early.
            Topology::TriangleFan => panic!(
                "TriangleFan is not supported on Metal. \
                 Rewrite fan indices to TriangleList before creating this PSO."
            ),
        };

        // Pre-bake the embedded blend state if provided, otherwise bake the default.
        let initial_blend = desc.blendstate.as_ref().cloned().unwrap_or_default();
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
                depth_format: depth_format_mtl,
                stencil_format: stencil_format_mtl,
                sample_count,
                alpha_to_coverage: desc.alpha_to_coverage,
                root_constant_size: desc.root_constant_size,
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
            inner: ComputePsoInner::Metal(MetalComputePso {
                pipeline: pipeline_state,
                threads_per_threadgroup: desc.threads_per_threadgroup,
                root_constant_size: desc.root_constant_size,
                compute_argument_buffer_slots,
            }),
        })
    }

    // -- Meshlet (mesh shader) pipeline --

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

        // Mesh function descriptor
        let mesh_fn_name = NSString::from_str(&mesh_module.entry_point);
        let mesh_func_desc = MTL4LibraryFunctionDescriptor::new();
        mesh_func_desc.setName(Some(&mesh_fn_name));
        mesh_func_desc.setLibrary(Some(&mesh_module.library));

        // Fragment function descriptor
        let frag_fn_name = NSString::from_str(&frag_module.entry_point);
        let frag_func_desc = MTL4LibraryFunctionDescriptor::new();
        frag_func_desc.setName(Some(&frag_fn_name));
        frag_func_desc.setLibrary(Some(&frag_module.library));

        let pipeline_desc = MTL4MeshRenderPipelineDescriptor::new();
        pipeline_desc.setMeshFunctionDescriptor(Some(&mesh_func_desc));
        pipeline_desc.setFragmentFunctionDescriptor(Some(&frag_func_desc));

        // Sample count and alpha-to-coverage
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

        // Color attachment formats
        for (i, &fmt) in color_formats.iter().enumerate() {
            let att = unsafe { pipeline_desc.colorAttachments().objectAtIndexedSubscript(i) };
            att.setPixelFormat(fmt);
        }

        let pipeline_options = MTL4PipelineOptions::new();
        pipeline_options.setShaderReflection(
            MTL4ShaderReflection::BindingInfo | MTL4ShaderReflection::BufferTypeInfo,
        );
        pipeline_desc.setOptions(Some(&pipeline_options));

        // Fragment function — needed by the argument encoder to build bindless heap layouts.
        // MTL4MeshRenderPipelineDescriptor inherits from MTL4PipelineDescriptor.
        let base_desc: &MTL4PipelineDescriptor = pipeline_desc.as_ref();
        let default_pipeline = compiler
            .newRenderPipelineStateWithDescriptor_compilerTaskOptions_error(base_desc, None)
            .map_err(|e| RhiError::PipelineCreation(format!("Mesh PSO: {e}")))?;

        // Extract argument buffer slot indices from mesh + fragment shader reflection.
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
                // Mesh pipelines expose mesh + fragment bindings in reflection.
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
                sample_count,
                alpha_to_coverage: desc.alpha_to_coverage,
                color_formats,
                depth_format: desc
                    .depth_format
                    .map(super::texture::format_to_mtl)
                    .unwrap_or(MTLPixelFormat::Invalid),
                stencil_format: desc
                    .stencil_format
                    .map(super::texture::format_to_mtl)
                    .unwrap_or(MTLPixelFormat::Invalid),
                root_constant_size: desc.root_constant_size,
                argument_buffer_slots,
                blend_pipelines: std::cell::RefCell::new(std::collections::HashMap::new()),
                default_pipeline,
            })),
        })
    }

    // -- Acceleration structures --

    pub fn create_blas(&self, desc: &BlasDesc) -> RhiResult<AccelerationStructure> {
        use super::accel::make_blas_geometry_descriptors;
        use objc2_metal::MTL4PrimitiveAccelerationStructureDescriptor;

        let geometries = make_blas_geometry_descriptors(desc)?;
        let primitive_desc = MTL4PrimitiveAccelerationStructureDescriptor::new();
        primitive_desc.setGeometryDescriptors(Some(&geometries.array));

        let sizes = unsafe {
            self.device.accelerationStructureSizesWithDescriptor(
                &*(primitive_desc.as_ref() as *const MTL4PrimitiveAccelerationStructureDescriptor
                    as *const objc2_metal::MTLAccelerationStructureDescriptor),
            )
        };
        self.finalize_accel_structure(sizes, "BLAS")
    }

    pub fn create_tlas(&self, desc: &TlasDesc) -> RhiResult<AccelerationStructure> {
        use objc2_metal::MTL4InstanceAccelerationStructureDescriptor;

        let instance_desc = MTL4InstanceAccelerationStructureDescriptor::new();
        unsafe {
            instance_desc.setInstanceDescriptorBuffer(objc2_metal::MTL4BufferRange {
                bufferAddress: desc.instance_buffer.0,
                // MTL4InstanceAccelerationStructureDescriptor defaults to the *indirect*
                // instance-descriptor type, whose native layout is
                // MTLIndirectAccelerationStructureInstanceDescriptor (NOT the Vulkan-shaped
                // TlasInstance). Callers fill the buffer via device.write_tlas_instance.
                length: (desc.instance_count as u64) * self.tlas_instance_stride() as u64,
            });
            instance_desc.setInstanceCount(desc.instance_count as usize);
        }

        let sizes = unsafe {
            self.device.accelerationStructureSizesWithDescriptor(
                &*(instance_desc.as_ref() as *const MTL4InstanceAccelerationStructureDescriptor
                    as *const objc2_metal::MTLAccelerationStructureDescriptor),
            )
        };
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
    /// `inst.acceleration_structure_reference` must be the BLAS `gpuResourceID` (i.e. the
    /// value returned by `device.accel_gpu_address(blas)` on Metal).
    pub fn write_tlas_instance(&self, dst: *mut u8, inst: &crate::types::TlasInstance) {
        use objc2_metal::{
            MTLAccelerationStructureInstanceOptions, MTLIndirectAccelerationStructureInstanceDescriptor,
            MTLPackedFloat3, MTLPackedFloat4x3, MTLResourceID,
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
            unsafe { std::mem::transmute(inst.acceleration_structure_reference) };

        let desc = MTLIndirectAccelerationStructureInstanceDescriptor {
            transformationMatrix: MTLPackedFloat4x3 {
                columns: [col(0), col(1), col(2), col(3)],
            },
            options: MTLAccelerationStructureInstanceOptions::empty(),
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
                scratch_buffer: Some(scratch),
            })),
        })
    }

    pub fn create_command_buffer(&self) -> RhiResult<CommandBuffer> {
        let allocator = self.device.newCommandAllocator().ok_or_else(|| {
            RhiError::CommandBuffer("Failed to create MTL4CommandAllocator".into())
        })?;

        let cmd = self
            .device
            .newCommandBuffer()
            .ok_or_else(|| RhiError::CommandBuffer("Failed to create MTL4CommandBuffer".into()))?;

        let mtl_cmd = MetalCommandBuffer::new(
            cmd,
            allocator,
            self.device.clone(),
            self.residency_set.clone(),
            self.residency_dirty.clone(),
            self.textures.clone(),
            self.samplers.clone(),
            self.allocations.clone(),
            self.mdi_icb_pipeline.clone(),
        )?;

        Ok(CommandBuffer {
            inner: crate::command::CommandBufferInner::Metal(Box::new(mtl_cmd)),
        })
    }

    pub fn create_command_buffer_for_swapchain(
        &self,
        swapchain: &Swapchain,
    ) -> RhiResult<CommandBuffer> {
        let mut cmd_buf = self.create_command_buffer()?;

        // Set drawable and depth textures on the Metal command buffer
        match (&mut cmd_buf.inner, &swapchain.inner) {
            (crate::command::CommandBufferInner::Metal(mtl_cmd), SwapchainInner::Metal(sc)) => {
                mtl_cmd.drawable_texture = sc.current_drawable_texture.borrow().clone();
                mtl_cmd.depth_texture = Some(sc.depth_texture.clone());
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

    pub fn destroy_buffer(&self, buffer: GpuBuffer) {
        match buffer.inner {
            #[cfg(feature = "metal")]
            GpuBufferInner::Metal(mtl) => {
                {
                    let mut allocations = self.allocations.borrow_mut();
                    allocations.remove(&mtl.gpu_address().0);
                }
                let allocation = unsafe {
                    &*(mtl.buffer.as_ref() as *const ProtocolObject<dyn MTLBuffer>
                        as *const ProtocolObject<dyn MTLAllocation>)
                };
                self.residency_set.removeAllocation(allocation);
                self.residency_dirty.set(true);
            }
            #[cfg(feature = "vulkan")]
            GpuBufferInner::Vulkan(_) => {}
        }
    }

    pub fn destroy_texture(&self, texture: Texture) {
        let mut textures = self.textures.borrow_mut();
        let idx = texture.id.0 as usize;
        if idx < textures.len()
            && let Some(tex) = textures[idx].take()
        {
            let allocation = unsafe {
                &*(tex.as_ref() as *const ProtocolObject<dyn MTLTexture>
                    as *const ProtocolObject<dyn MTLAllocation>)
            };
            self.residency_set.removeAllocation(allocation);
            self.residency_dirty.set(true);
        }
    }

    pub fn texture_view_descriptor(
        &self,
        source: &Texture,
        view: &crate::texture::GpuViewDesc,
    ) -> RhiResult<TextureId> {
        self.create_view_internal(source, view)
    }

    pub fn rw_texture_view_descriptor(
        &self,
        source: &Texture,
        view: &crate::texture::GpuViewDesc,
    ) -> RhiResult<TextureId> {
        // Metal does not separate sampled vs storage view creation — the same MTLTextureView
        // is used for both read and read/write access. Usage is controlled by shader binding type.
        self.create_view_internal(source, view)
    }

    fn create_view_internal(
        &self,
        source: &Texture,
        view: &crate::texture::GpuViewDesc,
    ) -> RhiResult<TextureId> {
        use crate::texture::{ALL_LAYERS, ALL_MIPS};
        use objc2_foundation::NSRange;

        let textures_borrow = self.textures.borrow();
        let src_texture = textures_borrow
            .get(source.id.0 as usize)
            .and_then(|t| t.as_ref())
            .ok_or_else(|| {
                RhiError::TextureCreation(
                    "texture_view_descriptor: invalid source TextureId".into(),
                )
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

        // Register view for residency (it shares memory with the source).
        let allocation = unsafe {
            &*(view_texture.as_ref() as *const ProtocolObject<dyn MTLTexture>
                as *const ProtocolObject<dyn MTLAllocation>)
        };
        self.residency_set.addAllocation(allocation);
        self.residency_dirty.set(true);

        let mut textures = self.textures.borrow_mut();
        let id = TextureId(textures.len() as u32);
        textures.push(Some(view_texture));
        drop(textures);

        Ok(id)
    }

    fn create_depth_texture(
        &self,
        width: u32,
        height: u32,
    ) -> RhiResult<Retained<ProtocolObject<dyn MTLTexture>>> {
        let desc = MTLTextureDescriptor::new();
        unsafe {
            desc.setPixelFormat(MTLPixelFormat::Depth32Float);
            desc.setWidth(width as usize);
            desc.setHeight(height as usize);
            desc.setStorageMode(MTLStorageMode::Private);
            desc.setUsage(MtlTextureUsage::RenderTarget);
        }

        let texture = self
            .device
            .newTextureWithDescriptor(&desc)
            .ok_or_else(|| RhiError::TextureCreation("Depth texture creation failed".into()))?;

        let allocation = unsafe {
            &*(texture.as_ref() as *const ProtocolObject<dyn MTLTexture>
                as *const ProtocolObject<dyn MTLAllocation>)
        };
        self.residency_set.addAllocation(allocation);
        self.residency_dirty.set(true);

        Ok(texture)
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
