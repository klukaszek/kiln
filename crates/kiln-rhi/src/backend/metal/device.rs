use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, VecDeque};
use std::ptr::NonNull;
use std::rc::Rc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_core_foundation::CGSize;
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4CommandAllocator, MTL4CommandBuffer, MTL4CommandQueue, MTL4Compiler,
    MTL4CompilerDescriptor, MTL4ComputePipelineDescriptor, MTL4LibraryFunctionDescriptor,
    MTL4PipelineDescriptor, MTLBuffer, MTLCreateSystemDefaultDevice, MTLCullMode, MTLDevice,
    MTLDrawable, MTLEvent, MTLHeap, MTLPixelFormat, MTLResidencySet, MTLResidencySetDescriptor,
    MTLSamplerState, MTLSharedEvent, MTLTexture, MTLWinding,
};
use objc2_quartz_core::CAMetalLayer;
use raw_window_handle::RawWindowHandle;

use crate::accel::{AccelInner, AccelerationStructure};
use crate::command::CommandBuffer;
use crate::device::DeviceDesc;
use crate::error::{RhiError, RhiResult};
use crate::memory::{Allocation, AllocationDesc, AllocationInner};
use crate::pipeline::{
    ComputePso, ComputePsoDesc, ComputePsoInner, GraphicsPso, GraphicsPsoDesc, GraphicsPsoInner,
    MeshletPso, MeshletPsoDesc,
};
use crate::queue::{Queue, QueueInner, SubmitDesc};
use crate::shader::{ShaderModule, ShaderModuleDesc};
use crate::surface::{Surface, SurfaceDesc, SurfaceInner};
use crate::swapchain::{AcquiredImage, Swapchain, SwapchainDesc, SwapchainInner};
use crate::sync::{TimelineSemaphore, TimelineSemaphoreInner};
use crate::types::{
    BlasDesc, Cull, GpuPtr, InstanceFlags, MAX_BINDLESS_SAMPLERS, MAX_BINDLESS_TEXTURES,
    MAX_FRAMES_IN_FLIGHT,
    SampleCount, SamplerId, TextureId, TlasDesc, Topology,
};

use super::as_allocation;
use super::command::{
    MetalCommandBuffer, MetalFrameTableSlot, SharedFrameTableSlots, SharedTableSlotPool,
    create_root_table_buffer,
};
use super::memory::{MetalBuffer, MetalBufferPool, SharedMetalBufferPool};
use super::pipeline::{MetalComputePso, MetalGraphicsPso};
use super::shader::MetalShaderModule;
use super::surface::MetalSurface;
use super::swapchain::{MetalDrawableSlot, MetalSwapchain};
use super::sync::MetalTimelineSemaphore;
use super::texture::{format_to_mtl, mtl_to_format};

type FrameFenceValues = Rc<RefCell<[u64; MAX_FRAMES_IN_FLIGHT]>>;
type InFlightFrameCommands = Rc<RefCell<Vec<Option<MetalCommandBuffer>>>>;
type PendingSubmissions = Rc<RefCell<VecDeque<(u64, MetalCommandBuffer)>>>;

/// Fixed capacities of the bindless descriptor heaps; one `gpuResourceID` per entry.
pub(crate) const METAL_BINDLESS_TEXTURE_CAPACITY: usize = MAX_BINDLESS_TEXTURES as usize;
pub(crate) const METAL_BINDLESS_SAMPLER_CAPACITY: usize = MAX_BINDLESS_SAMPLERS as usize;
/// Bindless slot tables, indexed by `TextureId`/`SamplerId`. A slot is `None` once its resource
/// is destroyed and before the ID is handed out again.
type TextureSlots = RefCell<Vec<Option<super::texture::MetalTexture>>>;
type SamplerSlots = RefCell<Vec<Option<Retained<ProtocolObject<dyn MTLSamplerState>>>>>;

/// Reverse index for CPU-mapped allocations, used by the public pointer bridge.
type SharedMappedAllocations = Rc<RefCell<BTreeMap<usize, MappedAllocation>>>;

/// Device state the queue and every command buffer also need. Held behind one `Rc` so creating a
/// command buffer threads a single handle instead of a dozen.
pub(crate) struct MetalShared {
    pub(crate) device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub(crate) residency_set: Retained<ProtocolObject<dyn MTLResidencySet>>,
    /// Set when the residency set gains or loses an allocation; committed at the next submit.
    /// Shared with the buffer pool, which tracks residency once per heap.
    pub(crate) residency_dirty: Rc<Cell<bool>>,
    /// Texture views live here alongside textures; both consume `TextureId`s.
    pub(crate) textures: TextureSlots,
    pub(crate) samplers: SamplerSlots,
    pub(crate) free_texture_ids: RefCell<Vec<TextureId>>,
    pub(crate) free_sampler_ids: RefCell<Vec<SamplerId>>,
    /// Buffer allocations keyed by GPU base address, so blit copies and indirect draws resolve an
    /// address to its `MTLBuffer` in O(log n) rather than by scanning.
    pub(crate) allocations: RefCell<BTreeMap<u64, BufferAllocation>>,
    /// `gpuResourceID`s indexed by TextureId/SamplerId, bound at argument-table slots 1 and 2.
    pub(crate) texture_heap: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) sampler_heap: Retained<ProtocolObject<dyn MTLBuffer>>,
}

pub(crate) enum MetalRetiredResource {
    Buffer(MetalBuffer),
    Texture {
        id: TextureId,
        texture: Retained<ProtocolObject<dyn MTLTexture>>,
        /// Views are the only textures held in the residency set individually: a placed texture
        /// is covered by its heap, but a view is created from another texture, not from a heap.
        is_view: bool,
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
    pub base: GpuPtr<u8>,
    pub size: u64,
    pub buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub heap: Retained<ProtocolObject<dyn MTLHeap>>,
    /// Where `base` sits inside `heap`. Placed textures are positioned against the heap, so an
    /// allocation-relative offset would alias whatever else occupies that heap offset.
    pub heap_offset: u64,
}

pub(crate) struct MappedAllocation {
    gpu_base: GpuPtr<u8>,
    size: u64,
}

fn resolve_mapped_pointer(
    allocations: &BTreeMap<usize, MappedAllocation>,
    ptr: usize,
) -> Option<GpuPtr<u8>> {
    let (&base, allocation) = allocations.range(..=ptr).next_back()?;
    let offset = (ptr - base) as u64;
    (offset < allocation.size).then(|| allocation.gpu_base.offset(offset))
}

pub struct MetalDevice {
    pub(crate) shared: Rc<MetalShared>,
    /// `MTL4Compiler` owns a compilation context, so it is built once rather than per PSO.
    pub(crate) compiler: Retained<ProtocolObject<dyn MTL4Compiler>>,
    pub(crate) buffer_pool: SharedMetalBufferPool,
    pub(crate) rhi_queue: Queue,
    pub(crate) mapped_allocations: SharedMappedAllocations,
    /// Per-frame fence values for swapchain acquisition.
    pub(crate) frame_fence_values: FrameFenceValues,
    /// Shared event for per-frame synchronization.
    pub(crate) frame_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    pub(crate) frame_table_slots: SharedFrameTableSlots,
    /// Free list of table slots for non-swapchain command buffers.
    pub(crate) table_slot_pool: SharedTableSlotPool,
}

pub struct MetalQueue {
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    shared: Rc<MetalShared>,
    frame_fence_values: FrameFenceValues,
    frame_fence_next: Rc<Cell<u64>>,
    frame_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    in_flight_frame_commands: InFlightFrameCommands,
    pending_submissions: PendingSubmissions,
    /// Resources destroyed by the app, each tagged with the fence value that must retire before
    /// its storage and its bindless slot can be reused. See `MetalQueue::release_resource`.
    retired_resources: RefCell<VecDeque<(u64, MetalRetiredResource)>>,
}

impl MetalQueue {
    /// Hold a resource until every submission issued so far has retired.
    pub(crate) fn release_resource(&self, resource: MetalRetiredResource) {
        let pending_until = self.frame_fence_next.get();
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
            MetalRetiredResource::Sampler { sampler, .. } => {
                let _ = sampler;
            }
        }
        match resource {
            MetalRetiredResource::Texture { id, .. } => {
                self.shared.free_texture_ids.borrow_mut().push(id);
            }
            MetalRetiredResource::Sampler { id, .. } => {
                self.shared.free_sampler_ids.borrow_mut().push(id);
            }
            MetalRetiredResource::Buffer(buffer) => buffer.release_to_pool(),
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
            let event = backend_expect!(&semaphore.inner, TimelineSemaphoreInner::Metal);
            self.queue
                .waitForEvent_value(shared_event_as_event(&event.event), *value);
        }

        let mut cmd = cmd;
        cmd.finish();
        self.commit_single(&cmd.command_buffer);

        for (semaphore, value) in desc.signal_semaphores {
            let event = backend_expect!(&semaphore.inner, TimelineSemaphoreInner::Metal);
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
        if frame_index >= self.frame_fence_values.borrow().len()
            || frame_index >= self.in_flight_frame_commands.borrow().len()
        {
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

    pub fn acquire_image(
        &self,
        sc: &MetalSwapchain,
        frame_index: usize,
    ) -> RhiResult<AcquiredImage> {
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

    fn reclaim_completed_submissions(&self) {
        if self.pending_submissions.borrow().is_empty()
            && self.retired_resources.borrow().is_empty()
        {
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

    fn commit_single(&self, cmd: &Retained<ProtocolObject<dyn MTL4CommandBuffer>>) {
        let cmd_ptr = NonNull::from(cmd.as_ref());
        let mut bufs = [cmd_ptr];
        unsafe {
            let ptr = NonNull::from(&mut bufs[0]);
            self.queue.commit_count(ptr, 1);
        }
    }
}

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

impl MetalDevice {
    pub fn new(desc: &DeviceDesc) -> RhiResult<Self> {
        let device = MTLCreateSystemDefaultDevice().ok_or(RhiError::NoSuitableGpu)?;

        let residency_desc = MTLResidencySetDescriptor::new();
        if let Some(label) = &desc.label {
            residency_desc.setLabel(Some(&NSString::from_str(label)));
        }
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

        // Shared with the buffer pool so heap-level residency changes reach the same commit.
        let residency_dirty = Rc::new(Cell::new(true));

        let frame_event = device
            .newSharedEvent()
            .ok_or_else(|| RhiError::DeviceCreation("Failed to create MTLSharedEvent".into()))?;
        let buffer_pool: SharedMetalBufferPool =
            Rc::new(RefCell::new(MetalBufferPool::new(
                device.clone(),
                residency_set.clone(),
                residency_dirty.clone(),
            )));
        let frame_fence_values: FrameFenceValues =
            Rc::new(RefCell::new([0u64; MAX_FRAMES_IN_FLIGHT]));
        let frame_fence_next = Rc::new(Cell::new(0u64));
        let in_flight_frame_commands: InFlightFrameCommands = Rc::new(RefCell::new(
            std::iter::repeat_with(|| None)
                .take(MAX_FRAMES_IN_FLIGHT)
                .collect(),
        ));
        let pending_submissions: PendingSubmissions = Rc::new(RefCell::new(VecDeque::new()));
        let frame_table_slots = Rc::new(RefCell::new(vec![None; MAX_FRAMES_IN_FLIGHT]));

        log::info!("Metal device created: {}", device.name());


        let create_heap = |len: usize, label: &str| {
            let heap = device
                .newBufferWithLength_options(
                    len * std::mem::size_of::<u64>(),
                    objc2_metal::MTLResourceOptions::StorageModeShared,
                )
                .ok_or_else(|| {
                    RhiError::DeviceCreation(format!("Failed to allocate Metal {label} heap"))
                })?;
            {
                use objc2_metal::MTLResource;
                heap.setLabel(Some(&NSString::from_str(label)));
            }
            residency_set.addAllocation(as_allocation(&heap));
            Ok::<_, RhiError>(heap)
        };
        let texture_heap = create_heap(METAL_BINDLESS_TEXTURE_CAPACITY, "bindless-texture-heap")?;
        let sampler_heap = create_heap(METAL_BINDLESS_SAMPLER_CAPACITY, "bindless-sampler-heap")?;

        let shared = Rc::new(MetalShared {
            device: device.clone(),
            residency_set,
            residency_dirty,
            textures: RefCell::new(Vec::new()),
            samplers: RefCell::new(Vec::new()),
            free_texture_ids: RefCell::new(Vec::new()),
            free_sampler_ids: RefCell::new(Vec::new()),
            allocations: RefCell::new(BTreeMap::new()),
            texture_heap,
            sampler_heap,
        });

        let rhi_queue = Queue {
            inner: QueueInner::Metal(Box::new(MetalQueue {
                queue: queue.clone(),
                shared: shared.clone(),
                frame_fence_values: frame_fence_values.clone(),
                frame_fence_next,
                frame_event: frame_event.clone(),
                in_flight_frame_commands,
                pending_submissions,
                retired_resources: RefCell::new(VecDeque::new()),
            })),
        };

        let compiler_desc = MTL4CompilerDescriptor::new();
        let compiler = device
            .newCompilerWithDescriptor_error(&compiler_desc)
            .map_err(|e| {
                RhiError::DeviceCreation(format!("Metal MTL4 compiler creation failed: {e}"))
            })?;

        let device = Self {
            shared,
            compiler,
            buffer_pool,
            rhi_queue,
            mapped_allocations: Rc::new(RefCell::new(BTreeMap::new())),
            frame_fence_values,
            frame_event,
            frame_table_slots,
            table_slot_pool: Rc::new(RefCell::new(Vec::new())),
        };

        Ok(device)
    }

    pub fn queue(&self) -> &Queue {
        &self.rhi_queue
    }

    pub fn wait_idle(&self) {
        backend_expect!(&self.rhi_queue.inner, QueueInner::Metal).wait_idle();
        // Heaps are only handed back here, never on the destroy path, so this cannot land
        // mid-frame. The queue is idle already.
        self.buffer_pool.borrow_mut().trim();
    }

    pub fn wait_for_frame(&self, frame_index: usize) {
        let value = self.frame_fence_values.borrow()[frame_index];
        if value != 0 {
            assert!(
                self.frame_event
                    .waitUntilSignaledValue_timeoutMS(value, u64::MAX),
                "Failed to wait for Metal frame completion"
            );
        }
        let q = backend_expect!(&self.rhi_queue.inner, QueueInner::Metal);
        q.in_flight_frame_commands.borrow_mut()[frame_index] = None;
        q.reclaim_completed_submissions();
    }

    pub fn create_surface(&self, desc: &SurfaceDesc) -> RhiResult<Surface> {
        let layer = match desc.window_handle {
            RawWindowHandle::AppKit(handle) => unsafe {
                use objc2::msg_send;
                use objc2::runtime::{AnyObject, Bool};

                let ns_view = handle.ns_view.as_ptr() as *mut AnyObject;

                let layer = CAMetalLayer::new();
                layer.setDevice(Some(&self.shared.device));
                layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm_sRGB);
                layer.setFramebufferOnly(true);
                layer.setOpaque(true);

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
            _owner: None,
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

        // Drawables live in the layer's own residency set; adding it to the queue is what makes
        // them resident. The layer outlives every swapchain built from it, so this happens once.
        backend_expect!(&self.rhi_queue.inner, QueueInner::Metal)
            .queue
            .addResidencySet(&layer.residencySet());

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
            _owner: None,
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

    pub fn create_allocation(&self, desc: &AllocationDesc) -> RhiResult<Allocation> {
        let length = usize::try_from(desc.size).map_err(|_| {
            RhiError::BufferCreation("Metal buffer size exceeds the host address space".into())
        })?;
        let metal_buffer =
            MetalBufferPool::allocate_shared(&self.buffer_pool, length, desc.memory)?;

        if let Some(label) = &desc.label {
            use objc2_metal::MTLResource;
            let ns_label = NSString::from_str(label);
            metal_buffer.buffer.setLabel(Some(&ns_label));
        }

        {
            let mut allocations = self.shared.allocations.borrow_mut();
            allocations.insert(
                metal_buffer.gpu_address().address,
                BufferAllocation {
                    base: metal_buffer.gpu_address(),
                    size: metal_buffer.size,
                    buffer: metal_buffer.buffer.clone(),
                    heap: metal_buffer.heap.clone(),
                    heap_offset: metal_buffer.heap_offset(),
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

        Ok(Allocation {
            inner: AllocationInner::Metal(metal_buffer),
            _owner: None,
            offset: 0,
            size: desc.size,
        })
    }

    pub fn host_to_device_pointer(&self, cpu_ptr: *const u8) -> Option<GpuPtr<u8>> {
        if cpu_ptr.is_null() {
            return None;
        }
        let ptr = cpu_ptr as usize;
        let allocations = self.mapped_allocations.borrow();
        resolve_mapped_pointer(&allocations, ptr)
    }

    /// Write `resource_id` into a bindless heap slot. Destroyed resources leave their slot stale
    /// rather than zeroed; an in-flight frame may still legitimately read it.
    pub(crate) fn write_heap_slot(
        heap: &ProtocolObject<dyn MTLBuffer>,
        index: usize,
        resource_id: u64,
    ) {
        unsafe {
            let base = heap.contents().as_ptr() as *mut u64;
            base.add(index).write(resource_id);
        }
    }

    pub fn create_shader_module(&self, desc: &ShaderModuleDesc) -> RhiResult<ShaderModule> {
        // For Metal, `code` should be a compiled .metallib binary.
        let ptr = std::ptr::NonNull::new(desc.code.as_ptr().cast::<std::ffi::c_void>().cast_mut())
            .expect("shader code pointer is null");
        let dispatch_data = unsafe {
            dispatch2::DispatchData::new(ptr, desc.code.len(), None, std::ptr::null_mut())
        };

        let library = self
            .shared
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
            _owner: None,
        })
    }

    pub fn create_graphics_pso(
        &self,
        desc: &GraphicsPsoDesc,
        vert_module: &MetalShaderModule,
        frag_module: &MetalShaderModule,
    ) -> RhiResult<GraphicsPso> {
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

        let blend = desc.blendstate.clone().unwrap_or_default();
        let pipeline_state = MetalGraphicsPso::compile_pipeline_state(
            self.compiler.as_ref(),
            vert_module.library.as_ref(),
            &vert_module.entry_point,
            frag_module.library.as_ref(),
            &frag_module.entry_point,
            &color_formats,
            &color_write_masks,
            sample_count,
            desc.alpha_to_coverage,
            &blend,
            desc.label.as_deref(),
        )?;

        let (cull_mode, winding) = cull_to_mtl(desc.cull);

        let topology = match desc.topology {
            Topology::TriangleList => objc2_metal::MTLPrimitiveType::Triangle,
            Topology::TriangleStrip => objc2_metal::MTLPrimitiveType::TriangleStrip,
        };

        Ok(GraphicsPso {
            inner: GraphicsPsoInner::Metal(Box::new(MetalGraphicsPso {
                pipeline: pipeline_state,
                depth_stencil: super::pipeline::make_depth_stencil_state(
                    self.shared.device.as_ref(),
                    desc.depth,
                ),
                depth_bias: (
                    desc.depth.bias,
                    desc.depth.bias_slope,
                    desc.depth.bias_clamp,
                ),
                cull_mode,
                winding,
                topology,
            })),
            _owner: None,
        })
    }

    pub fn create_compute_pso(
        &self,
        desc: &ComputePsoDesc,
        compute_module: &MetalShaderModule,
    ) -> RhiResult<ComputePso> {
        let fn_name = NSString::from_str(&compute_module.entry_point);
        let func_desc = MTL4LibraryFunctionDescriptor::new();
        func_desc.setName(Some(&fn_name));
        func_desc.setLibrary(Some(&compute_module.library));

        let pipeline_desc = MTL4ComputePipelineDescriptor::new();
        pipeline_desc.setComputeFunctionDescriptor(Some(&func_desc));
        if let Some(label) = desc.label.as_deref() {
            let base: &MTL4PipelineDescriptor = pipeline_desc.as_ref();
            base.setLabel(Some(&NSString::from_str(label)));
        }
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

        let pipeline_state = self
            .compiler
            .newComputePipelineStateWithDescriptor_compilerTaskOptions_error(&pipeline_desc, None)
            .map_err(|e| {
                RhiError::PipelineCreation(format!("Metal compute PSO creation failed: {e}"))
            })?;

        // A threadgroup wider than the pipeline's register budget is undefined in Metal: the
        // dispatch is dropped and the frame comes out black, with nothing raised anywhere. Vulkan
        // rejects the same mistake at validation, so report it here rather than let it render.
        let max_threads = {
            use objc2_metal::MTLComputePipelineState;
            pipeline_state.maxTotalThreadsPerThreadgroup()
        };
        let requested = tg.width * tg.height * tg.depth;
        if requested > max_threads {
            return Err(RhiError::PipelineCreation(format!(
                "Metal compute PSO {:?} requested {requested} threads per threadgroup, but the \
                 compiled shader's register use allows at most {max_threads}",
                desc.label.as_deref().unwrap_or("<unlabelled>"),
            )));
        }

        Ok(ComputePso {
            inner: ComputePsoInner::Metal(Box::new(MetalComputePso {
                pipeline: pipeline_state,
                threads_per_threadgroup: desc.threads_per_threadgroup,
                label: desc.label.clone(),
            })),
            _owner: None,
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
        if let Some(label) = desc.label.as_deref() {
            let base: &MTL4PipelineDescriptor = pipeline_desc.as_ref();
            base.setLabel(Some(&NSString::from_str(label)));
        }

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

        let blend = desc.blendstate.clone().unwrap_or_default();
        for (i, target) in desc.color_targets.iter().enumerate() {
            let att = unsafe { pipeline_desc.colorAttachments().objectAtIndexedSubscript(i) };
            att.setPixelFormat(super::texture::format_to_mtl(target.format));
            let blend_att = blend.attachments.get(i).copied().unwrap_or_default();
            super::pipeline::apply_blend_to_attachment(att.as_ref(), blend_att, target.write_mask);
        }

        let base_desc: &MTL4PipelineDescriptor = pipeline_desc.as_ref();
        let default_pipeline = self
            .compiler
            .newRenderPipelineStateWithDescriptor_compilerTaskOptions_error(base_desc, None)
            .map_err(|e| RhiError::PipelineCreation(format!("Mesh PSO: {e}")))?;

        let (cull_mode, winding) = cull_to_mtl(desc.cull);

        Ok(MeshletPso {
            inner: crate::pipeline::MeshletPsoInner::Metal(Box::new(MetalMeshletPso {
                cull_mode,
                winding,
                depth_stencil: super::pipeline::make_depth_stencil_state(
                    self.shared.device.as_ref(),
                    desc.depth,
                ),
                depth_bias: (
                    desc.depth.bias,
                    desc.depth.bias_slope,
                    desc.depth.bias_clamp,
                ),
                default_pipeline,
            })),
            _owner: None,
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
            .shared
            .device
            .accelerationStructureSizesWithDescriptor(primitive_base);
        self.finalize_accel_structure(sizes, "BLAS")
    }

    pub fn create_tlas(&self, desc: &TlasDesc) -> RhiResult<AccelerationStructure> {
        use objc2_metal::MTL4InstanceAccelerationStructureDescriptor;

        let instance_desc = MTL4InstanceAccelerationStructureDescriptor::new();
        unsafe {
            instance_desc.setInstanceDescriptorBuffer(objc2_metal::MTL4BufferRange {
                bufferAddress: desc.instance_buffer.address,
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
            .shared
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
    /// `inst.acceleration_structure_reference` must be the BLAS handle (`blas.gpu()`).
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
        // SAFETY: the handle came from `MTLAccelerationStructure::gpuResourceID().to_raw()`
        // in `create_blas` / `create_tlas`, so it is a live resource ID for this device.
        let resource_id =
            unsafe { MTLResourceID::from_raw(inst.acceleration_structure_reference.0) };

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
    /// with the residency set and query the GPU resource ID. Shared by `create_blas`/`create_tlas`.
    fn finalize_accel_structure(
        &self,
        sizes: objc2_metal::MTLAccelerationStructureSizes,
        label: &'static str,
    ) -> RhiResult<AccelerationStructure> {
        use super::accel::MetalAccelerationStructure;
        use objc2_metal::{MTLAccelerationStructure as _, MTLDevice, MTLResourceOptions};

        let accel = self
            .shared
            .device
            .newAccelerationStructureWithSize(sizes.accelerationStructureSize)
            .ok_or_else(|| {
                RhiError::AllocationFailed(format!("Failed to allocate Metal {label}"))
            })?;
        let scratch = self
            .shared
            .device
            .newBufferWithLength_options(
                sizes.buildScratchBufferSize,
                MTLResourceOptions::StorageModePrivate,
            )
            .ok_or_else(|| {
                RhiError::AllocationFailed(format!("Failed to allocate {label} scratch buffer"))
            })?;

        self.shared
            .residency_set
            .addAllocation(as_allocation(&accel));
        self.shared
            .residency_set
            .addAllocation(as_allocation(&scratch));
        self.shared.residency_dirty.set(true);

        let gpu_resource_id = accel.gpuResourceID().to_raw();

        Ok(AccelerationStructure {
            inner: AccelInner::Metal(Box::new(MetalAccelerationStructure {
                acceleration_structure: accel,
                gpu_resource_id,
                scratch_buffer: scratch,
                shared: self.shared.clone(),
            })),
            _owner: None,
        })
    }

    /// Build a fresh table slot. `pool` is the free list it returns to on drop; swapchain slots
    /// pass `None`.
    fn new_table_slot(
        &self,
        pool: Option<SharedTableSlotPool>,
        what: &str,
    ) -> RhiResult<Rc<MetalFrameTableSlot>> {
        let root_table_buffer = create_root_table_buffer(
            self.shared.device.as_ref(),
            self.shared.residency_set.as_ref(),
            &self.shared.residency_dirty,
        )?;
        let command_allocator = self.shared.device.newCommandAllocator().ok_or_else(|| {
            RhiError::CommandBuffer(format!("Failed to create {what} MTL4CommandAllocator"))
        })?;
        let command_buffer = self.shared.device.newCommandBuffer().ok_or_else(|| {
            RhiError::CommandBuffer(format!("Failed to create {what} MTL4CommandBuffer"))
        })?;
        Ok(Rc::new(MetalFrameTableSlot {
            root_table_buffer,
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
            self.shared.clone(),
            slot.clone(),
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
            _owner: None,
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
            .shared
            .device
            .newSharedEvent()
            .ok_or_else(|| RhiError::SyncError("Failed to create MTLSharedEvent".into()))?;
        event.setSignaledValue(initial_value);

        Ok(TimelineSemaphore {
            inner: crate::sync::TimelineSemaphoreInner::Metal(Box::new(MetalTimelineSemaphore {
                event,
            })),
            _owner: None,
        })
    }

    pub fn destroy_allocation(&self, buffer: Allocation) {
        match buffer.inner {
            #[cfg(feature = "metal")]
            AllocationInner::Metal(mtl) => {
                {
                    let mut allocations = self.shared.allocations.borrow_mut();
                    allocations.remove(&mtl.gpu_address().address);
                }
                if let Some(mapped_ptr) = mtl.mapped_ptr() {
                    self.mapped_allocations
                        .borrow_mut()
                        .remove(&(mapped_ptr as usize));
                }
                backend_expect!(&self.rhi_queue.inner, QueueInner::Metal)
                    .release_resource(MetalRetiredResource::Buffer(mtl));
            }
            #[cfg(feature = "vulkan")]
            AllocationInner::Vulkan(_) => {}
        }
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
                gpu_base: GpuPtr::from_addr(0x8000),
                size: 0x20,
            },
        );
        allocations.insert(
            0x2000,
            MappedAllocation {
                gpu_base: GpuPtr::from_addr(0x9000),
                size: 0x10,
            },
        );

        assert_eq!(
            resolve_mapped_pointer(&allocations, 0x100f),
            Some(GpuPtr::from_addr(0x800f))
        );
        assert_eq!(
            resolve_mapped_pointer(&allocations, 0x200f),
            Some(GpuPtr::from_addr(0x900f))
        );
        assert_eq!(resolve_mapped_pointer(&allocations, 0x1020), None);
        assert_eq!(resolve_mapped_pointer(&allocations, 0x1fff), None);
    }
}
