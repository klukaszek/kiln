use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, VecDeque};
use std::rc::Rc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4CommandQueue, MTL4Compiler, MTL4CompilerDescriptor, MTLBuffer,
    MTLCreateSystemDefaultDevice, MTLCullMode, MTLDevice, MTLEvent, MTLHeap, MTLResidencySet,
    MTLResidencySetDescriptor, MTLSamplerState, MTLSharedEvent, MTLTexture, MTLWinding,
};

use crate::device::DeviceDesc;
use crate::error::{RhiError, RhiResult};
use crate::queue::Queue;
use crate::types::{Cull, GpuPtr, MAX_FRAMES_IN_FLIGHT, SamplerId, TextureId};

use super::as_allocation;
use super::command::{MetalCommandBuffer, SharedFrameTableSlots, SharedTableSlotPool};
use super::heap::BindlessTable;
use super::memory::{MetalBuffer, MetalBufferPool, SharedMetalBufferPool};
use super::queue::MetalQueue;
use crate::backend::mapped::MappedAllocations;
use crate::backend::retire::RetirementQueue;

pub(crate) type FrameFenceValues = RefCell<[u64; MAX_FRAMES_IN_FLIGHT]>;
pub(crate) type InFlightFrameCommands = RefCell<[Option<MetalCommandBuffer>; MAX_FRAMES_IN_FLIGHT]>;
pub(crate) type PendingSubmissions = RefCell<VecDeque<(u64, MetalCommandBuffer)>>;

/// Reverse index for CPU-mapped allocations, used by the public pointer bridge.
type SharedMappedAllocations = Rc<RefCell<MappedAllocations>>;

/// Device state the queue and every command buffer also need. Held behind one `Rc` so creating a
/// command buffer threads a single handle instead of a dozen.
pub(crate) struct MetalShared {
    pub(crate) device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub(crate) residency_set: Retained<ProtocolObject<dyn MTLResidencySet>>,
    /// Set when the residency set gains or loses an allocation; committed at the next submit.
    /// Shared with the buffer pool, which tracks residency once per heap.
    pub(crate) residency_dirty: Rc<Cell<bool>>,
    /// Texture views live here alongside textures; both consume `TextureId`s.
    pub(crate) textures: BindlessTable<super::texture::MetalTexture>,
    pub(crate) samplers: BindlessTable<Retained<ProtocolObject<dyn MTLSamplerState>>>,
    /// Buffer allocations keyed by GPU base address, so blit copies and indirect draws resolve an
    /// address to its `MTLBuffer` in O(log n) rather than by scanning.
    pub(crate) allocations: RefCell<BTreeMap<u64, BufferAllocation>>,
}

pub(crate) enum MetalRetiredResource {
    Buffer(MetalBuffer),
    /// Holds the `Retained` handles as well as the residency entries: releasing either while a
    /// frame is still tracing against the structure is a use-after-free.
    Accel(Box<super::accel::MetalAccelerationStructure>),
    Texture {
        id: TextureId,
        texture: Retained<ProtocolObject<dyn MTLTexture>>,
        /// Views are the only textures held in the residency set individually: a placed texture
        /// is covered by its heap, but a view is created from another texture, not from a heap.
        is_view: bool,
    },
    Sampler {
        id: SamplerId,
        /// Released only once the submissions reading its `gpuResourceID` out of the bindless
        /// heap have retired; dropping it earlier is a use-after-free.
        sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
    },
}

/// `MTLSharedEvent` refines `MTLEvent`, which is what the queue's wait/signal calls take.
pub(crate) fn shared_event_as_event(
    event: &Retained<ProtocolObject<dyn MTLSharedEvent>>,
) -> &ProtocolObject<dyn MTLEvent> {
    // SAFETY: the MTLSharedEvent protocol inherits from MTLEvent, so every object conforming to
    // the former conforms to the latter.
    unsafe { super::cast_protocol(&**event) }
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

pub struct MetalDevice {
    pub(crate) shared: Rc<MetalShared>,
    /// `MTL4Compiler` owns a compilation context, so it is built once rather than per PSO.
    pub(crate) compiler: Retained<ProtocolObject<dyn MTL4Compiler>>,
    pub(crate) buffer_pool: SharedMetalBufferPool,
    pub(crate) queue: Rc<MetalQueue>,
    /// The same queue, wrapped for [`RhiDevice::queue`]. Both point at one `MetalQueue`.
    rhi_queue: Queue,
    pub(crate) mapped_allocations: SharedMappedAllocations,
    /// Shared event carrying the submission timeline, used for per-frame synchronization.
    pub(crate) frame_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    pub(crate) frame_table_slots: SharedFrameTableSlots,
    /// Free list of table slots for non-swapchain command buffers.
    pub(crate) table_slot_pool: SharedTableSlotPool,
}

/// Translate the unified `Cull` value into Metal's `(cull_mode, front-face winding)` pair.
/// Every variant implies CCW as the front-face convention.
pub(crate) fn cull_to_mtl(cull: Cull) -> (MTLCullMode, MTLWinding) {
    let winding = MTLWinding::CounterClockwise;
    match cull {
        Cull::None => (MTLCullMode::None, winding),
        Cull::Cw => (MTLCullMode::Back, winding),
        Cull::Ccw => (MTLCullMode::Front, winding),
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
                RhiError::DeviceCreation(
                    format!("Failed to create Metal residency set: {e}").into(),
                )
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
        let buffer_pool: SharedMetalBufferPool = Rc::new(RefCell::new(MetalBufferPool::new(
            device.clone(),
            residency_set.clone(),
            residency_dirty.clone(),
        )));
        let frame_fence_values: FrameFenceValues = RefCell::new([0u64; MAX_FRAMES_IN_FLIGHT]);
        let frame_fence_next = Cell::new(0u64);
        let in_flight_frame_commands: InFlightFrameCommands =
            RefCell::new([const { None }; MAX_FRAMES_IN_FLIGHT]);
        let pending_submissions: PendingSubmissions = RefCell::new(VecDeque::new());
        let frame_table_slots = Rc::new(RefCell::new([const { None }; MAX_FRAMES_IN_FLIGHT]));

        log::info!("Metal device created: {}", device.name());

        let create_heap = |len: usize, label: &str| {
            let heap = device
                .newBufferWithLength_options(
                    len * std::mem::size_of::<u64>(),
                    objc2_metal::MTLResourceOptions::StorageModeShared,
                )
                .ok_or_else(|| {
                    RhiError::DeviceCreation(
                        format!("Failed to allocate Metal {label} heap").into(),
                    )
                })?;
            {
                use objc2_metal::MTLResource;
                heap.setLabel(Some(&NSString::from_str(label)));
            }
            residency_set.addAllocation(as_allocation(&heap));
            Ok::<_, RhiError>(heap)
        };
        let texture_heap = create_heap(desc.bindless.textures as usize, "bindless-texture-heap")?;
        let sampler_heap = create_heap(desc.bindless.samplers as usize, "bindless-sampler-heap")?;

        let shared = Rc::new(MetalShared {
            device: device.clone(),
            residency_set,
            residency_dirty,
            textures: BindlessTable::new(texture_heap, desc.bindless.textures, "texture"),
            samplers: BindlessTable::new(sampler_heap, desc.bindless.samplers, "sampler"),
            allocations: RefCell::new(BTreeMap::new()),
        });

        let queue = Rc::new(MetalQueue {
            queue: queue.clone(),
            shared: shared.clone(),
            frame_fence_values,
            frame_fence_next,
            frame_event: frame_event.clone(),
            in_flight_frame_commands,
            pending_submissions,
            retired_resources: RetirementQueue::default(),
        });
        let rhi_queue = Queue {
            inner: queue.clone(),
        };

        let compiler_desc = MTL4CompilerDescriptor::new();
        let compiler = device
            .newCompilerWithDescriptor_error(&compiler_desc)
            .map_err(|e| {
                RhiError::DeviceCreation(format!("Metal MTL4 compiler creation failed: {e}").into())
            })?;

        let device = Self {
            shared,
            compiler,
            buffer_pool,
            queue,
            rhi_queue,
            mapped_allocations: Rc::new(RefCell::new(BTreeMap::new())),
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
        self.queue.wait_idle();
        // Heaps are only handed back here, never on the destroy path, so this cannot land
        // mid-frame. The queue is idle already.
        self.buffer_pool.borrow_mut().trim();
    }

    pub fn wait_for_frame(&self, frame_index: usize) {
        let q = &self.queue;
        let value = q.frame_fence_values.borrow()[frame_index];
        if value != 0 {
            assert!(
                self.frame_event
                    .waitUntilSignaledValue_timeoutMS(value, u64::MAX),
                "Failed to wait for Metal frame completion"
            );
        }
        q.in_flight_frame_commands.borrow_mut()[frame_index] = None;
        q.reclaim_completed_submissions();
    }
}
