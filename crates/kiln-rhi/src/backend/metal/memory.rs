use std::cell::{Cell, RefCell};
use std::rc::Rc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLDevice, MTLHeap, MTLHeapDescriptor, MTLHeapType, MTLResidencySet,
    MTLResourceOptions,
};

use objc2_foundation::NSString;

use super::as_allocation;
use super::device::{BufferAllocation, MetalDevice, MetalRetiredResource};
use crate::backend::mapped::{MappedAllocation, resolve_mapped_pointer};
use crate::backend::suballoc::BlockPool;
use crate::error::{RhiError, RhiResult};
use crate::memory::{Allocation, AllocationDesc, MemoryType};
use crate::types::GpuPtr;

pub(crate) type SharedMetalBufferPool = Rc<RefCell<MetalBufferPool>>;

pub struct MetalBuffer {
    pub(crate) buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) heap: Retained<ProtocolObject<dyn MTLHeap>>,
    pub(crate) size: u64,
    pub(crate) is_shared: bool,
    pool: SharedMetalBufferPool,
    block_index: usize,
    heap_offset: u64,
    heap_size: u64,
}

impl MetalBuffer {
    pub fn mapped_ptr(&self) -> Option<*mut u8> {
        if self.is_shared {
            Some(self.buffer.contents().as_ptr().cast::<u8>())
        } else {
            None
        }
    }

    pub fn gpu_address(&self) -> GpuPtr<u8> {
        GpuPtr::from_addr(self.buffer.gpuAddress())
    }

    /// Byte offset of this suballocation within its `MTLHeap`. Placed resources are positioned
    /// relative to the heap, not to the allocation, so texture placement needs this.
    pub(crate) fn heap_offset(&self) -> u64 {
        self.heap_offset
    }

    /// Return this buffer's placement range to the pool after GPU retirement.
    pub(crate) fn release_to_pool(self) {
        self.pool
            .borrow_mut()
            .release(self.block_index, self.heap_offset, self.heap_size);
    }
}

/// The native half of a pooled block: one `MTLHeap` with `Placement` type.
struct MetalHeapBlock {
    heap: Retained<ProtocolObject<dyn MTLHeap>>,
}

pub(crate) struct MetalBufferPool {
    residency_set: Retained<ProtocolObject<dyn MTLResidencySet>>,
    residency_dirty: Rc<Cell<bool>>,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    /// Keyed by memory type: two heaps are interchangeable when their storage mode matches.
    blocks: BlockPool<MemoryType, MetalHeapBlock>,
}

impl MetalBufferPool {
    pub(crate) fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        residency_set: Retained<ProtocolObject<dyn MTLResidencySet>>,
        residency_dirty: Rc<Cell<bool>>,
    ) -> Self {
        Self {
            device,
            residency_set,
            residency_dirty,
            blocks: BlockPool::default(),
        }
    }

    pub(crate) fn allocate_shared(
        pool: &SharedMetalBufferPool,
        length: usize,
        align: u64,
        memory: MemoryType,
    ) -> RhiResult<MetalBuffer> {
        let pool_handle = pool.clone();
        pool.borrow_mut()
            .allocate(length, align, memory, pool_handle)
    }

    /// `requested_align` is the caller's, which may be stricter than Metal's own placement
    /// requirement; the range is carved at whichever is larger, so the returned address satisfies
    /// both without over-allocating to leave room for a shift.
    fn allocate(
        &mut self,
        length: usize,
        requested_align: u64,
        memory: MemoryType,
        pool: SharedMetalBufferPool,
    ) -> RhiResult<MetalBuffer> {
        let options = resource_options(memory);
        let requirements = self
            .device
            .heapBufferSizeAndAlignWithLength_options(length, options);
        let size = requirements.size as u64;
        let align = (requirements.align as u64).max(requested_align).max(1);

        let device = self.device.clone();
        let residency_set = self.residency_set.clone();
        let residency_dirty = self.residency_dirty.clone();
        let (block_index, heap_offset) =
            self.blocks.allocate(memory, size, align, |block_size| {
                let heap = create_heap(&device, block_size, options)?;
                // Residency is tracked once per heap: everything placed inside it is covered,
                // so individual buffers and textures never touch the residency set.
                residency_set.addAllocation(as_allocation(&heap));
                residency_dirty.set(true);
                Ok(MetalHeapBlock { heap })
            })?;

        let heap = self.blocks.block(block_index).payload.heap.clone();
        // SAFETY: `heap_offset` was carved from this heap at an alignment Metal accepts.
        let buffer = unsafe {
            heap.newBufferWithLength_options_offset(length, options, heap_offset as usize)
        };
        let Some(buffer) = buffer else {
            self.blocks.release(block_index, heap_offset, size);
            return Err(RhiError::BufferCreation(
                "Metal placed buffer allocation failed".into(),
            ));
        };

        Ok(MetalBuffer {
            buffer,
            heap,
            size: length as u64,
            is_shared: matches!(memory, MemoryType::Upload | MemoryType::Readback),
            pool,
            block_index,
            heap_offset,
            heap_size: size,
        })
    }

    /// Return a range to its block. Never frees the block — see [`Self::trim`]. This runs on the
    /// per-frame retire path, and any request over `BLOCK_SIZE` gets a dedicated
    /// block, so freeing here would destroy and recreate a heap for every large transient.
    pub(crate) fn release(&mut self, block_index: usize, offset: u64, size: u64) {
        self.blocks.release(block_index, offset, size);
    }

    /// Release empty heaps, keeping one per memory type so usage oscillating around a block
    /// boundary still reuses. Call only where already synchronising, never per frame.
    pub(crate) fn trim(&mut self) {
        let residency_set = self.residency_set.clone();
        let residency_dirty = self.residency_dirty.clone();
        self.blocks.trim(|block| {
            residency_set.removeAllocation(as_allocation(&block.heap));
            residency_dirty.set(true);
        });
    }
}

/// One `MTLHeap` of at least `size` bytes, sized for placement.
fn create_heap(
    device: &ProtocolObject<dyn MTLDevice>,
    size: u64,
    options: MTLResourceOptions,
) -> RhiResult<Retained<ProtocolObject<dyn MTLHeap>>> {
    let heap_desc = MTLHeapDescriptor::new();
    heap_desc.setType(MTLHeapType::Placement);
    heap_desc.setSize(usize::try_from(size).map_err(|_| {
        RhiError::BufferCreation("Metal heap size exceeds the host address space".into())
    })?);
    heap_desc.setResourceOptions(options);
    device
        .newHeapWithDescriptor(&heap_desc)
        .ok_or_else(|| RhiError::BufferCreation("Metal buffer heap allocation failed".into()))
}

fn resource_options(memory: MemoryType) -> MTLResourceOptions {
    match memory {
        MemoryType::Upload => {
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeWriteCombined
        }
        MemoryType::Readback => {
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeDefaultCache
        }
        MemoryType::GpuOnly => MTLResourceOptions::StorageModePrivate,
    }
}

impl MetalDevice {
    pub fn create_allocation(&self, desc: &AllocationDesc) -> RhiResult<Allocation> {
        let length = usize::try_from(desc.size).map_err(|_| {
            RhiError::BufferCreation("Metal buffer size exceeds the host address space".into())
        })?;
        let metal_buffer =
            MetalBufferPool::allocate_shared(&self.buffer_pool, length, desc.align, desc.memory)?;

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
            inner: metal_buffer,
            _owner: None,
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

    pub fn destroy_allocation(&self, buffer: MetalBuffer) {
        self.shared
            .allocations
            .borrow_mut()
            .remove(&buffer.gpu_address().address);
        if let Some(mapped_ptr) = buffer.mapped_ptr() {
            self.mapped_allocations
                .borrow_mut()
                .remove(&(mapped_ptr as usize));
        }
        self.queue
            .release_resource(MetalRetiredResource::Buffer(buffer));
    }
}
