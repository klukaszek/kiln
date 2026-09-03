use std::cell::RefCell;
use std::rc::Rc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLDevice, MTLHeap, MTLHeapDescriptor, MTLHeapType, MTLResourceOptions,
};

use crate::backend::suballoc::FreeRanges;
use crate::error::{RhiError, RhiResult};
use crate::memory::MemoryType;
use crate::types::GpuPtr;

const BUFFER_HEAP_BLOCK_SIZE: u64 = 4 * 1024 * 1024;

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
            Some(self.buffer.contents().as_ptr() as *mut u8)
        } else {
            None
        }
    }

    pub fn gpu_address(&self) -> GpuPtr<u8> {
        GpuPtr::from_addr(self.buffer.gpuAddress())
    }

    /// Return this buffer's placement range to the pool after GPU retirement.
    pub(crate) fn release_to_pool(self) {
        self.pool
            .borrow_mut()
            .release(self.block_index, self.heap_offset, self.heap_size);
    }
}

/// One placement heap plus its free-range bookkeeping.
struct MetalHeapBlock {
    heap: Retained<ProtocolObject<dyn MTLHeap>>,
    memory: MemoryType,
    free_ranges: FreeRanges,
}

pub(crate) struct MetalBufferPool {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    /// Emptied, never removed, so a buffer's `block_index` stays valid for its whole life.
    blocks: Vec<Option<MetalHeapBlock>>,
}

impl MetalBufferPool {
    pub(crate) fn new(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        Self {
            device,
            blocks: Vec::new(),
        }
    }

    pub(crate) fn allocate_shared(
        pool: &SharedMetalBufferPool,
        length: usize,
        memory: MemoryType,
    ) -> RhiResult<MetalBuffer> {
        let pool_handle = pool.clone();
        pool.borrow_mut().allocate(length, memory, pool_handle)
    }

    fn allocate(
        &mut self,
        length: usize,
        memory: MemoryType,
        pool: SharedMetalBufferPool,
    ) -> RhiResult<MetalBuffer> {
        let options = resource_options(memory);
        let requirements = self
            .device
            .heapBufferSizeAndAlignWithLength_options(length, options);
        let size = requirements.size as u64;
        let align = requirements.align as u64;

        let existing = self
            .blocks
            .iter_mut()
            .enumerate()
            .find_map(|(index, slot)| {
                let block = slot.as_mut()?;
                (block.memory == memory)
                    .then(|| block.free_ranges.allocate(size, align))
                    .flatten()
                    .map(|offset| (index, offset))
            });
        let (block_index, heap_offset) = match existing {
            Some(allocation) => allocation,
            None => {
                let block_index = self.create_block(memory, size, options)?;
                let heap_offset = self.blocks[block_index]
                    .as_mut()
                    .expect("freshly created block")
                    .free_ranges
                    .allocate(size, align)
                    .ok_or_else(|| {
                        RhiError::AllocationFailed("Metal buffer heap is full".into())
                    })?;
                (block_index, heap_offset)
            }
        };

        let block = self.blocks[block_index]
            .as_mut()
            .expect("allocated block is present");
        let buffer = unsafe {
            block
                .heap
                .newBufferWithLength_options_offset(length, options, heap_offset as usize)
        };
        let Some(buffer) = buffer else {
            block.free_ranges.release(heap_offset, size);
            return Err(RhiError::BufferCreation(
                "Metal placed buffer allocation failed".into(),
            ));
        };

        Ok(MetalBuffer {
            buffer,
            heap: block.heap.clone(),
            size: length as u64,
            is_shared: matches!(memory, MemoryType::Upload | MemoryType::Readback),
            pool,
            block_index,
            heap_offset,
            heap_size: size,
        })
    }

    fn create_block(
        &mut self,
        memory: MemoryType,
        required_size: u64,
        options: MTLResourceOptions,
    ) -> RhiResult<usize> {
        let size = required_size.max(BUFFER_HEAP_BLOCK_SIZE);
        let heap_desc = MTLHeapDescriptor::new();
        heap_desc.setType(MTLHeapType::Placement);
        heap_desc.setSize(usize::try_from(size).map_err(|_| {
            RhiError::BufferCreation("Metal heap size exceeds the host address space".into())
        })?);
        heap_desc.setResourceOptions(options);
        let heap = self
            .device
            .newHeapWithDescriptor(&heap_desc)
            .ok_or_else(|| {
                RhiError::BufferCreation("Metal buffer heap allocation failed".into())
            })?;
        let block = MetalHeapBlock {
            heap,
            memory,
            free_ranges: FreeRanges::full(size),
        };
        Ok(match self.blocks.iter().position(Option::is_none) {
            Some(index) => {
                self.blocks[index] = Some(block);
                index
            }
            None => {
                self.blocks.push(Some(block));
                self.blocks.len() - 1
            }
        })
    }

    /// Return a range to its block. Never frees the block — see [`Self::trim`]. This runs on the
    /// per-frame retire path, and any request over `BUFFER_HEAP_BLOCK_SIZE` gets a dedicated
    /// block, so freeing here would destroy and recreate a heap for every large transient.
    pub(crate) fn release(&mut self, block_index: usize, offset: u64, size: u64) {
        let Some(block) = self.blocks.get_mut(block_index).and_then(Option::as_mut) else {
            debug_assert!(false, "Metal buffer pool block disappeared");
            return;
        };
        block.free_ranges.release(offset, size);
    }

    /// Release empty blocks, keeping one per memory type so usage oscillating around a block
    /// boundary still reuses. Call only where already synchronising, never per frame.
    pub(crate) fn trim(&mut self) {
        let mut kept_empty = Vec::new();
        for slot in &mut self.blocks {
            let Some(block) = slot.as_ref() else { continue };
            if !block.free_ranges.is_empty() {
                continue;
            }
            if kept_empty.contains(&block.memory) {
                *slot = None;
            } else {
                kept_empty.push(block.memory);
            }
        }
    }
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

// The free-range bookkeeping these tests used to cover now lives in `backend::suballoc`, shared
// with the Vulkan block allocator, and is tested there.
