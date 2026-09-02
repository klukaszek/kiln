use std::sync::{Arc, Mutex};

use crate::backend::suballoc::FreeRanges;
use crate::error::{RhiError, RhiResult};
use crate::types::GpuAddress;
use ash::vk;

/// Matches the Metal pool's block size, so both backends fragment the same way.
const MEMORY_BLOCK_SIZE: u64 = 4 * 1024 * 1024;

/// Vulkan buffer with buffer_device_address support.
pub struct VulkanBuffer {
    pub(crate) buffer: vk::Buffer,
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) size: u64,
    pub(crate) mapped_ptr: Option<*mut u8>,
    pub(crate) gpu_address: GpuAddress,
    /// Backing block and the padded range to hand back on destruction.
    pub(crate) block_index: usize,
    pub(crate) block_offset: u64,
    pub(crate) block_range: u64,
}

// SAFETY: VulkanBuffer's raw pointer is only used for CPU-side uploads
// and the underlying Vulkan memory is externally synchronized.
unsafe impl Send for VulkanBuffer {}
unsafe impl Sync for VulkanBuffer {}

impl VulkanBuffer {
    pub fn mapped_ptr(&self) -> Option<*mut u8> {
        self.mapped_ptr
    }

    pub fn gpu_address(&self) -> GpuAddress {
        self.gpu_address
    }
}

/// One `VkDeviceMemory` allocation, subdivided between many buffers.
struct MemoryBlock {
    memory: vk::DeviceMemory,
    memory_type_index: u32,
    /// Vulkan permits one mapping per `VkDeviceMemory`, so the block owns it and buffers point
    /// into it.
    mapped_ptr: Option<*mut u8>,
    free_ranges: FreeRanges,
}

/// Where a suballocation landed.
pub(crate) struct BlockSuballocation {
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) block_index: usize,
    pub(crate) offset: u64,
    pub(crate) range: u64,
    pub(crate) mapped_ptr: Option<*mut u8>,
}

/// Block allocator for buffer memory. `maxMemoryAllocationCount` (commonly 4096) makes one
/// allocation per buffer a hard ceiling on scene size, not just a slow path.
pub(crate) struct VulkanBufferPool {
    /// Emptied, never removed, so a buffer's `block_index` stays valid for its whole life.
    blocks: Vec<Option<MemoryBlock>>,
    buffer_image_granularity: u64,
}

// SAFETY: the mapped pointers are owned by the pool and only handed out behind the same external
// synchronization that guards every other device resource.
unsafe impl Send for VulkanBufferPool {}

pub(crate) type SharedBufferPool = Arc<Mutex<VulkanBufferPool>>;

impl VulkanBufferPool {
    pub(crate) fn new(buffer_image_granularity: u64) -> Self {
        Self {
            blocks: Vec::new(),
            buffer_image_granularity: buffer_image_granularity.max(1),
        }
    }

    /// Suballocate from a block of `memory_type_index`, creating one if needed. `host_visible`
    /// decides whether a new block is mapped up front.
    pub(crate) fn allocate(
        &mut self,
        device: &ash::Device,
        requirements: &vk::MemoryRequirements,
        memory_type_index: u32,
        host_visible: bool,
    ) -> RhiResult<BlockSuballocation> {
        let (align, range) = granularity_padded(
            requirements.alignment,
            requirements.size,
            self.buffer_image_granularity,
        )
        .ok_or_else(|| RhiError::AllocationFailed("buffer size overflows padding".into()))?;

        let existing = self
            .blocks
            .iter_mut()
            .enumerate()
            .find_map(|(index, slot)| {
                let block = slot.as_mut()?;
                (block.memory_type_index == memory_type_index)
                    .then(|| block.free_ranges.allocate(range, align))
                    .flatten()
                    .map(|offset| (index, offset))
            });
        let (block_index, offset) = match existing {
            Some(found) => found,
            None => {
                let block_index =
                    self.create_block(device, range, memory_type_index, host_visible)?;
                let offset = self.blocks[block_index]
                    .as_mut()
                    .expect("freshly created block")
                    .free_ranges
                    .allocate(range, align)
                    .ok_or_else(|| {
                        RhiError::AllocationFailed("Vulkan memory block is full".into())
                    })?;
                (block_index, offset)
            }
        };

        let block = self.blocks[block_index]
            .as_ref()
            .expect("allocated block is present");
        let mapped_ptr = block
            .mapped_ptr
            .map(|base| unsafe { base.add(offset as usize) });
        Ok(BlockSuballocation {
            memory: block.memory,
            block_index,
            offset,
            range,
            mapped_ptr,
        })
    }

    fn create_block(
        &mut self,
        device: &ash::Device,
        required_size: u64,
        memory_type_index: u32,
        host_visible: bool,
    ) -> RhiResult<usize> {
        let size = required_size.max(MEMORY_BLOCK_SIZE);
        let mut alloc_flags_info =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(memory_type_index)
            .push_next(&mut alloc_flags_info);
        let memory = unsafe {
            device
                .allocate_memory(&alloc_info, None)
                .map_err(|e| RhiError::AllocationFailed(e.to_string()))?
        };

        let mapped_ptr = if host_visible {
            match unsafe { device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty()) } {
                Ok(ptr) => Some(ptr as *mut u8),
                Err(error) => {
                    unsafe { device.free_memory(memory, None) };
                    return Err(RhiError::AllocationFailed(error.to_string()));
                }
            }
        } else {
            None
        };

        let block = MemoryBlock {
            memory,
            memory_type_index,
            mapped_ptr,
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
    /// per-frame retire path, and any request over `MEMORY_BLOCK_SIZE` gets a dedicated block, so
    /// freeing here would put a `vkAllocateMemory` on the critical path per large transient.
    pub(crate) fn release(&mut self, block_index: usize, offset: u64, range: u64) {
        let Some(block) = self.blocks.get_mut(block_index).and_then(Option::as_mut) else {
            debug_assert!(false, "Vulkan memory block disappeared");
            return;
        };
        block.free_ranges.release(offset, range);
    }

    /// Free empty blocks, keeping one per memory type so usage oscillating around a block boundary
    /// still reuses. Call only when the device is idle.
    pub(crate) fn trim(&mut self, device: &ash::Device) {
        let mut kept_empty: Vec<u32> = Vec::new();
        for slot in &mut self.blocks {
            let Some(block) = slot.as_ref() else { continue };
            if !block.free_ranges.is_empty() {
                continue;
            }
            if kept_empty.contains(&block.memory_type_index) {
                let block = slot.take().expect("block present immediately above");
                unsafe { Self::destroy_block(device, &block) };
            } else {
                kept_empty.push(block.memory_type_index);
            }
        }
    }

    /// Free every block. Call only once all GPU work has retired.
    pub(crate) fn destroy_all(&mut self, device: &ash::Device) {
        for block in self.blocks.drain(..).flatten() {
            unsafe { Self::destroy_block(device, &block) };
        }
    }

    /// # Safety
    /// No resource bound into `block` may still be alive.
    unsafe fn destroy_block(device: &ash::Device, block: &MemoryBlock) {
        unsafe {
            if block.mapped_ptr.is_some() {
                device.unmap_memory(block.memory);
            }
            device.free_memory(block.memory, None);
        }
    }
}

/// Widen alignment and size to whole `bufferImageGranularity` pages. Only resources of different
/// tiling classes need the separation, but the RHI lets a caller place an image into any
/// allocation after the fact, so every range is padded. Costs nothing where granularity is 1.
fn granularity_padded(alignment: u64, size: u64, granularity: u64) -> Option<(u64, u64)> {
    let granularity = granularity.max(1);
    let align = alignment.max(granularity).max(1);
    let range = size.checked_next_multiple_of(granularity)?;
    Some((align, range))
}

#[cfg(test)]
mod tests {
    use super::granularity_padded;

    #[test]
    fn granularity_of_one_costs_nothing() {
        assert_eq!(granularity_padded(256, 1000, 1), Some((256, 1000)));
    }

    #[test]
    fn allocations_are_widened_to_whole_granularity_pages() {
        assert_eq!(granularity_padded(256, 1000, 4096), Some((4096, 4096)));
        assert_eq!(granularity_padded(256, 4097, 4096), Some((4096, 8192)));
    }

    #[test]
    fn a_stricter_resource_alignment_still_wins() {
        assert_eq!(granularity_padded(65536, 1000, 4096), Some((65536, 4096)));
    }

    #[test]
    fn overflowing_padding_is_rejected_rather_than_wrapping() {
        assert_eq!(granularity_padded(1, u64::MAX, 4096), None);
    }
}
