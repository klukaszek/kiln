use std::cell::RefCell;
use std::rc::Rc;

use super::device::{BufferAllocation, VulkanDevice, buffer_memory_flags, find_memorytype_index};
use super::queue::VulkanRetiredResource;
use crate::backend::mapped::{MappedAllocation, resolve_mapped_pointer};
use crate::backend::suballoc::BlockPool;
use crate::error::{RhiError, RhiResult};
use crate::memory::{Allocation, MemoryType};
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

    /// An allocation is a range inside a shared block buffer, with no object of its own to name.
    pub fn set_label(&self, _label: &str) {}
}

/// The native half of a pooled block: one `VkDeviceMemory` plus the buffer spanning it.
struct MemoryBlock {
    memory: vk::DeviceMemory,
    /// One buffer spans the whole block, purely to give the block a device address. Every
    /// allocation inside it is `base_address + offset`.
    buffer: vk::Buffer,
    base_address: u64,
    /// Vulkan permits one mapping per `VkDeviceMemory`, so the block owns it and buffers point
    /// into it.
    mapped_ptr: Option<*mut u8>,
}

/// What makes two blocks interchangeable: one buffer spans a block, so every allocation inside it
/// inherits exactly these usage flags, and the memory type fixes where it lives.
#[derive(Clone, Copy, PartialEq, Eq)]
struct BlockKey {
    memory_type_index: u32,
    usage: vk::BufferUsageFlags,
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
    blocks: BlockPool<BlockKey, MemoryBlock>,
    buffer_image_granularity: u64,
}

pub(crate) type SharedBufferPool = Rc<RefCell<VulkanBufferPool>>;

impl VulkanBufferPool {
    pub(crate) fn new(buffer_image_granularity: u64) -> Self {
        Self {
            blocks: BlockPool::default(),
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

        let key = BlockKey {
            memory_type_index,
            usage,
        };
        let (block_index, offset) = self.blocks.allocate(key, range, align, |size| {
            create_block(device, size, memory_type_index, host_visible, usage)
        })?;

        let block = self.blocks.block(block_index);
        let mapped_ptr = block
            .payload
            .mapped_ptr
            // SAFETY: `offset` is inside the block, which is mapped for its whole length.
            .map(|base| unsafe { base.add(offset as usize) });
        Ok(BlockSuballocation {
            memory: block.payload.memory,
            block_index,
            offset,
            range,
            mapped_ptr,
            address: block.payload.base_address + offset,
        })
    }

    /// Return a range to its block. Never frees the block — see [`Self::trim`]. This runs on the
    /// per-frame retire path, and any request over `BLOCK_SIZE` gets a dedicated block, so
    /// freeing here would put a `vkAllocateMemory` on the critical path per large transient.
    pub(crate) fn release(&mut self, block_index: usize, offset: u64, range: u64) {
        self.blocks.release(block_index, offset, range);
    }

    /// Free empty blocks, keeping one per key so usage oscillating around a block boundary still
    /// reuses. Call only when the device is idle.
    pub(crate) fn trim(&mut self, device: &ash::Device) {
        // SAFETY: `trim` only yields blocks with nothing allocated in them, and the caller has
        // synchronised.
        self.blocks
            .trim(|block| unsafe { destroy_block(device, &block) });
    }

    /// Free every block. Call only once all GPU work has retired.
    pub(crate) fn destroy_all(&mut self, device: &ash::Device) {
        // SAFETY: the caller guarantees no resource bound into any block is still alive.
        self.blocks
            .destroy_all(|block| unsafe { destroy_block(device, &block) });
    }
}

/// Allocate one block: a buffer spanning it, its memory, and the mapping if it is host-visible.
///
/// Each failure past the first unwinds what it already holds, so nothing leaks on the way out.
fn create_block(
    device: &ash::Device,
    size: u64,
    memory_type_index: u32,
    host_visible: bool,
    usage: vk::BufferUsageFlags,
) -> RhiResult<MemoryBlock> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.create_buffer(&buffer_info, None) }
        .map_err(|e| RhiError::AllocationFailed(e.into()))?;
    let (memory, base_address) =
        match bind_device_address_memory(device, buffer, size, memory_type_index) {
            Ok(bound) => bound,
            Err(error) => {
                unsafe { device.destroy_buffer(buffer, None) };
                return Err(RhiError::AllocationFailed(error.into()));
            }
        };

    let mapped_ptr = if host_visible {
        match unsafe { device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty()) } {
            Ok(ptr) => Some(ptr.cast::<u8>()),
            Err(error) => {
                unsafe {
                    device.destroy_buffer(buffer, None);
                    device.free_memory(memory, None);
                }
                return Err(RhiError::AllocationFailed(error.into()));
            }
        }
    } else {
        None
    };

    Ok(MemoryBlock {
        memory,
        buffer,
        base_address,
        mapped_ptr,
    })
}

/// Allocate `size` bytes of `memory_type_index` with the `DEVICE_ADDRESS` flag, bind `buffer` to
/// it, and return the memory and the buffer's address. The flag is required for any buffer whose
/// address is taken and easy to omit, so every such allocation goes through here.
///
/// On failure the memory is freed; the buffer remains the caller's to destroy.
fn bind_device_address_memory(
    device: &ash::Device,
    buffer: vk::Buffer,
    size: u64,
    memory_type_index: u32,
) -> Result<(vk::DeviceMemory, u64), vk::Result> {
    let mut flags =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(size)
        .memory_type_index(memory_type_index)
        .push(&mut flags);
    let memory = unsafe { device.allocate_memory(&alloc_info, None) }?;
    if let Err(error) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
        unsafe { device.free_memory(memory, None) };
        return Err(error);
    }
    let address = unsafe {
        device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
    };
    Ok((memory, address))
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
    // SAFETY: `buffer_info` borrows nothing beyond this call.
    let buffer = unsafe { device.create_buffer(&buffer_info, None) }
        .map_err(|e| RhiError::AllocationFailed(format!("{what}: {e}").into()))?;

    let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    let bound = find_memorytype_index(&reqs, memory_properties, properties)
        .ok_or_else(|| RhiError::AllocationFailed(format!("No memory type for {what}").into()))
        .and_then(|memory_type| {
            bind_device_address_memory(device, buffer, reqs.size, memory_type)
                .map_err(|error| RhiError::AllocationFailed(format!("{what}: {error}").into()))
        });
    match bound {
        Ok((memory, address)) => Ok((buffer, memory, address)),
        Err(error) => {
            unsafe { device.destroy_buffer(buffer, None) };
            Err(error)
        }
    }
}

impl VulkanDevice {
    /// Memory requirements for a range of `size` inside a pooled block.
    ///
    /// Every allocation shares the block's usage flags, so alignment and the permitted memory
    /// types are properties of that usage rather than of any one allocation. They are probed once
    /// with a throwaway buffer and cached; afterwards this is pure arithmetic, which keeps scene
    /// loads from paying a create/destroy pair per allocation.
    pub(crate) fn buffer_requirements(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> RhiResult<vk::MemoryRequirements> {
        let cached = self
            .buffer_requirements_probe
            .borrow()
            .iter()
            .find(|(u, _, _)| *u == usage)
            .map(|&(_, a, b)| (a, b));
        let (alignment, memory_type_bits) = match cached {
            Some(cached) => cached,
            None => {
                let info = vk::BufferCreateInfo::default()
                    .size(1)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE);
                let probe = unsafe {
                    self.loaders
                        .device
                        .create_buffer(&info, None)
                        .map_err(|e| RhiError::BufferCreation(e.into()))?
                };
                let requirements =
                    unsafe { self.loaders.device.get_buffer_memory_requirements(probe) };
                unsafe { self.loaders.device.destroy_buffer(probe, None) };
                let probed = (requirements.alignment, requirements.memory_type_bits);
                self.buffer_requirements_probe
                    .borrow_mut()
                    .push((usage, probed.0, probed.1));
                probed
            }
        };
        Ok(vk::MemoryRequirements {
            size: size.max(1),
            alignment,
            memory_type_bits,
        })
    }

    pub fn create_allocation(
        &self,
        size: u64,
        align: u64,
        memory: MemoryType,
    ) -> RhiResult<Allocation> {
        // An allocation is a range inside a pooled block, so its requirements come from the
        // block's usage and are the same for every allocation.
        let mut mem_requirements = self.buffer_requirements(size, BLOCK_BUFFER_USAGE)?;
        // The caller's alignment may be stricter than the buffer's own requirement; the pool
        // carves at whichever is larger.
        mem_requirements.alignment = mem_requirements.alignment.max(align).max(1);

        let mem_flags = buffer_memory_flags(memory);

        let preferred_flags = match memory {
            MemoryType::Upload => vk::MemoryPropertyFlags::DEVICE_LOCAL,
            MemoryType::GpuOnly | MemoryType::Readback => vk::MemoryPropertyFlags::empty(),
        };
        // `Upload` prefers the host-visible device-local heap (resizable BAR), which is often
        // only 256 MiB, so a failed allocation retries in plain host memory.
        let mut candidates = [
            find_memorytype_index(
                &mem_requirements,
                &self.device_memory_properties,
                mem_flags | preferred_flags,
            ),
            find_memorytype_index(&mem_requirements, &self.device_memory_properties, mem_flags),
        ];
        if candidates[0] == candidates[1] {
            candidates[1] = None;
        }

        let mut attempt = Err(RhiError::AllocationFailed("No suitable memory type".into()));
        for candidate in candidates.into_iter().flatten() {
            // Follows the memory type's real properties, not the requested `MemoryType`: on UMA one
            // type serves both, and a block created by a `GpuOnly` buffer must still be mappable
            // for an `Upload` buffer landing in it later.
            let host_visible = self.device_memory_properties.memory_types[candidate as usize]
                .property_flags
                .contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
            let mut pool = self.buffer_pool.borrow_mut();
            attempt = pool.allocate(
                &self.loaders.device,
                &mem_requirements,
                candidate,
                host_visible,
                BLOCK_BUFFER_USAGE,
            );
            if attempt.is_ok() {
                break;
            }
        }
        let suballocation = attempt?;
        let gpu_addr = suballocation.address;

        // `GpuOnly` promises no CPU pointer even when it happens to land in host-visible memory.
        let mapped_ptr = match memory {
            MemoryType::Upload | MemoryType::Readback => suballocation.mapped_ptr,
            MemoryType::GpuOnly => None,
        };

        let vk_buffer = VulkanBuffer {
            memory: suballocation.memory,
            size,
            mapped_ptr,
            gpu_address: GpuPtr::from_addr(gpu_addr),
            block_index: suballocation.block_index,
            block_offset: suballocation.offset,
            block_range: suballocation.range,
        };

        {
            let mut allocations = self.allocations.borrow_mut();
            allocations.insert(
                vk_buffer.gpu_address.address,
                BufferAllocation {
                    base: vk_buffer.gpu_address,
                    size: vk_buffer.size,
                    memory: vk_buffer.memory,
                    memory_offset: vk_buffer.block_offset,
                },
            );
        }
        if let Some(mapped_ptr) = vk_buffer.mapped_ptr {
            self.mapped_allocations.borrow_mut().insert(
                mapped_ptr as usize,
                MappedAllocation {
                    gpu_base: vk_buffer.gpu_address,
                    size: vk_buffer.size,
                },
            );
        }

        Ok(Allocation {
            inner: vk_buffer,
            _owner: None,
            size,
            _type: std::marker::PhantomData,
        })
    }

    pub fn host_to_device_pointer(&self, cpu_ptr: *const u8) -> Option<GpuPtr<u8>> {
        if cpu_ptr.is_null() {
            return None;
        }
        let ptr = cpu_ptr as usize;
        resolve_mapped_pointer(&self.mapped_allocations.borrow(), ptr)
    }

    pub fn destroy_allocation(&self, buffer: VulkanBuffer) {
        self.allocations
            .borrow_mut()
            .remove(&buffer.gpu_address.address);
        if let Some(mapped_ptr) = buffer.mapped_ptr {
            self.mapped_allocations
                .borrow_mut()
                .remove(&(mapped_ptr as usize));
        }
        self.queue
            .release_resource(VulkanRetiredResource::Buffer(buffer));
    }
}

#[cfg(test)]
mod tests {
    use super::granularity_padded;

    #[test]
    fn allocations_are_widened_to_whole_granularity_pages() {
        assert_eq!(granularity_padded(256, 1000, 1), Some((256, 1000)));
        assert_eq!(granularity_padded(256, 1000, 4096), Some((4096, 4096)));
        assert_eq!(granularity_padded(256, 4097, 4096), Some((4096, 8192)));
        // A stricter resource alignment still wins.
        assert_eq!(granularity_padded(65536, 1000, 4096), Some((65536, 4096)));
        // Padding that would wrap is rejected instead.
        assert_eq!(granularity_padded(1, u64::MAX, 4096), None);
    }
}
