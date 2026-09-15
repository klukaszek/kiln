use std::cell::RefCell;
use std::rc::Rc;

use crate::backend::suballoc::{BLOCK_SIZE, FreeRanges};
use crate::error::{RhiError, RhiResult};
use crate::types::GpuPtr;
use ash::vk;
use ash::vk::TaggedStructure as _;

/// The one buffer spanning each block declares every use an allocation carved from it might be
/// put to.
///
/// Deliberately narrow. Shaders reach buffer data through device addresses, not storage-buffer
/// descriptors, and vertices are pulled through pointers rather than bound, so neither
/// `STORAGE_BUFFER` nor `VERTEX_BUFFER` belongs here. Keeping `STORAGE_BUFFER` off also keeps
/// the address-taking commands free of `VkAddressCommandFlagsKHR` usage bits, which only exist to
/// name buffer usages that overlap a range.
pub(crate) const BLOCK_BUFFER_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS.as_raw()
        | vk::BufferUsageFlags::INDEX_BUFFER.as_raw()
        | vk::BufferUsageFlags::INDIRECT_BUFFER.as_raw()
        | vk::BufferUsageFlags::TRANSFER_SRC.as_raw()
        | vk::BufferUsageFlags::TRANSFER_DST.as_raw()
        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR.as_raw()
        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR.as_raw(),
);

/// Acceleration-structure build scratch is the one thing the spec insists come from a buffer
/// created with `STORAGE_BUFFER`. It gets its own blocks so that bit never lands on ordinary
/// allocations, which would in turn force every address-taking command to declare it.
pub(crate) const SCRATCH_BUFFER_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    BLOCK_BUFFER_USAGE.as_raw() | vk::BufferUsageFlags::STORAGE_BUFFER.as_raw(),
);

/// A suballocation of a pooled block. Under `VK_KHR_device_address_commands` nothing takes a
/// `VkBuffer`, so an allocation is just an address, an optional CPU pointer, and the block range
/// to hand back. `memory` is retained only because images still bind to `VkDeviceMemory`.
pub struct VulkanBuffer {
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) size: u64,
    pub(crate) mapped_ptr: Option<*mut u8>,
    pub(crate) gpu_address: GpuPtr<u8>,
    /// Backing block and the padded range to hand back on destruction.
    pub(crate) block_index: usize,
    pub(crate) block_offset: u64,
    pub(crate) block_range: u64,
}

impl VulkanBuffer {
    pub fn mapped_ptr(&self) -> Option<*mut u8> {
        self.mapped_ptr
    }

    pub fn gpu_address(&self) -> GpuPtr<u8> {
        self.gpu_address
    }
}

/// One `VkDeviceMemory` allocation, subdivided between many buffers.
struct MemoryBlock {
    memory: vk::DeviceMemory,
    /// Blocks are keyed by usage as well as memory type: one buffer spans the block, so every
    /// allocation inside it inherits exactly these usage flags.
    usage: vk::BufferUsageFlags,
    /// One buffer spans the whole block, purely to give the block a device address. Every
    /// allocation inside it is `base_address + offset`.
    buffer: vk::Buffer,
    base_address: u64,
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
    /// Device address of this suballocation: the block's base plus `offset`.
    pub(crate) address: u64,
}

/// Block allocator for buffer memory. `maxMemoryAllocationCount` (commonly 4096) makes one
/// allocation per buffer a hard ceiling on scene size, not just a slow path.
pub(crate) struct VulkanBufferPool {
    /// Emptied, never removed, so a buffer's `block_index` stays valid for its whole life.
    blocks: Vec<Option<MemoryBlock>>,
    buffer_image_granularity: u64,
}

pub(crate) type SharedBufferPool = Rc<RefCell<VulkanBufferPool>>;

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
        usage: vk::BufferUsageFlags,
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
                (block.memory_type_index == memory_type_index && block.usage == usage)
                    .then(|| block.free_ranges.allocate(range, align))
                    .flatten()
                    .map(|offset| (index, offset))
            });
        let (block_index, offset) = match existing {
            Some(found) => found,
            None => {
                let block_index =
                    self.create_block(device, range, memory_type_index, host_visible, usage)?;
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
            address: block.base_address + offset,
        })
    }

    fn create_block(
        &mut self,
        device: &ash::Device,
        required_size: u64,
        memory_type_index: u32,
        host_visible: bool,
        usage: vk::BufferUsageFlags,
    ) -> RhiResult<usize> {
        let size = required_size.max(BLOCK_SIZE);
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe {
            device
                .create_buffer(&buffer_info, None)
                .map_err(|e| RhiError::AllocationFailed(e.to_string()))?
        };
        let mut alloc_flags_info =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(memory_type_index)
            .push(&mut alloc_flags_info);
        let memory = match unsafe { device.allocate_memory(&alloc_info, None) } {
            Ok(memory) => memory,
            Err(error) => {
                unsafe { device.destroy_buffer(buffer, None) };
                return Err(RhiError::AllocationFailed(error.to_string()));
            }
        };
        if let Err(error) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            }
            return Err(RhiError::AllocationFailed(error.to_string()));
        }
        let base_address = unsafe {
            device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };

        let mapped_ptr = if host_visible {
            match unsafe { device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty()) } {
                Ok(ptr) => Some(ptr as *mut u8),
                Err(error) => {
                    unsafe {
                        device.destroy_buffer(buffer, None);
                        device.free_memory(memory, None);
                    }
                    return Err(RhiError::AllocationFailed(error.to_string()));
                }
            }
        } else {
            None
        };

        let block = MemoryBlock {
            memory,
            usage,
            buffer,
            base_address,
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
    /// per-frame retire path, and any request over `BLOCK_SIZE` gets a dedicated block, so
    /// freeing here would put a `vkAllocateMemory` on the critical path per large transient.
    pub(crate) fn release(&mut self, block_index: usize, offset: u64, range: u64) {
        // `trim` only reclaims empty blocks, so a block still holding this range is present.
        self.blocks[block_index]
            .as_mut()
            .expect("memory block outlives its suballocations")
            .free_ranges
            .release(offset, range);
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
            device.destroy_buffer(block.buffer, None);
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


/// Create a buffer, back it with memory of `properties`, bind it, and return its device address.
/// The `DEVICE_ADDRESS` allocate flag is required for any buffer whose address is taken and is
/// easy to omit, so every such allocation goes through here.
pub(crate) fn allocate_bound_buffer(
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
    what: &str,
) -> RhiResult<(vk::Buffer, vk::DeviceMemory, u64)> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    unsafe {
        let buffer = device
            .create_buffer(&buffer_info, None)
            .map_err(|e| RhiError::AllocationFailed(format!("{what}: {e}")))?;
        let reqs = device.get_buffer_memory_requirements(buffer);
        let memory_type = crate::backend::vulkan::device::find_memorytype_index(
            &reqs,
            memory_properties,
            properties,
        )
        .ok_or_else(|| RhiError::AllocationFailed(format!("No memory type for {what}")))?;
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(memory_type)
            .push(&mut flags);
        let memory = device
            .allocate_memory(&alloc_info, None)
            .map_err(|e| RhiError::AllocationFailed(format!("{what}: {e}")))?;
        device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(|e| RhiError::AllocationFailed(format!("{what}: {e}")))?;
        let address = device
            .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));
        Ok((buffer, memory, address))
    }
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
