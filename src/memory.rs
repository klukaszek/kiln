use crate::types::GpuAddress;
use crate::{Device, RhiError, RhiResult};
use zerocopy::{FromBytes, IntoBytes};

/// Types copyable verbatim to/from GPU memory: `#[repr(C)]`, no padding, valid for any bit
/// pattern. Auto-implemented via the `zerocopy` derives (see `gpu_struct!`).
pub trait GpuPod: IntoBytes + FromBytes + zerocopy::Immutable {}
impl<T> GpuPod for T where T: IntoBytes + FromBytes + zerocopy::Immutable {}

/// Copy `bytes` into a CPU-mapped region, bounds-checked.
fn mapped_write(ptr: Option<*mut u8>, capacity: u64, bytes: &[u8]) -> RhiResult<()> {
    if bytes.len() as u64 > capacity {
        return Err(RhiError::AllocationFailed(format!(
            "upload of {} bytes exceeds mapped region ({capacity} bytes)",
            bytes.len()
        )));
    }
    let dst =
        ptr.ok_or_else(|| RhiError::AllocationFailed("allocation is not CPU-mapped".into()))?;
    // SAFETY: `dst` is valid for `capacity` bytes and `bytes.len() <= capacity`.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len()) };
    Ok(())
}

/// Read a `T` out of a CPU-mapped region, bounds-checked.
fn mapped_read<T: GpuPod>(ptr: Option<*mut u8>, capacity: u64) -> RhiResult<T> {
    let n = std::mem::size_of::<T>();
    let src =
        ptr.ok_or_else(|| RhiError::AllocationFailed("allocation is not CPU-mapped".into()))?;
    if n as u64 > capacity {
        return Err(RhiError::AllocationFailed(format!(
            "read of {n} bytes exceeds mapped region ({capacity} bytes)"
        )));
    }
    // SAFETY: `src` is valid for `capacity` >= `n` bytes; `T: FromBytes`.
    let bytes = unsafe { std::slice::from_raw_parts(src as *const u8, n) };
    T::read_from_bytes(bytes).map_err(|_| RhiError::AllocationFailed("read size mismatch".into()))
}

/// Memory residency for GPU allocations.
///
/// - `Default`: CPU-mapped, write-combined. Uniforms, staging, draw args, descriptors.
/// - `GpuOnly`: device-local, not CPU-mapped. Textures and large persistent buffers.
/// - `Readback`: GPU-writable, CPU-cached on read. Screenshots, feedback, GPGPU output.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MemoryType {
    #[default]
    Default,
    GpuOnly,
    Readback,
}

/// Description for creating a GPU buffer.
#[derive(Clone, Debug, Default)]
pub struct BufferDesc {
    pub size: u64,
    pub memory: MemoryType,
    pub label: Option<String>,
}

/// A GPU allocation: a CPU-mapped pointer + GPU address over a backing buffer.
pub struct GpuAllocation {
    pub(crate) buffer: GpuBuffer,
    pub(crate) size: u64,
}

impl GpuAllocation {
    /// CPU-mapped pointer (`None` for `GpuOnly`). The `.cpu` of the dual-pointer pair.
    pub fn cpu(&self) -> Option<*mut u8> {
        self.buffer.cpu()
    }

    /// GPU virtual address — the `.gpu` of the dual-pointer pair.
    pub fn gpu(&self) -> GpuAddress {
        self.buffer.gpu()
    }

    /// Allocation size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Consume the allocation and return the backing buffer.
    pub fn into_buffer(self) -> GpuBuffer {
        self.buffer
    }

    /// Upload a value into CPU-mapped memory (bounds-checked). Caller orders the write before
    /// the dependent submit.
    pub fn upload<T: GpuPod>(&self, value: &T) -> RhiResult<()> {
        mapped_write(self.cpu(), self.size, value.as_bytes())
    }

    /// Upload a slice into this allocation's CPU-mapped memory (bounds-checked).
    pub fn upload_slice<T: GpuPod>(&self, data: &[T]) -> RhiResult<()> {
        mapped_write(self.cpu(), self.size, data.as_bytes())
    }

    /// Read a value back from CPU-mapped memory (e.g. `Readback` after a GPU write).
    pub fn read<T: GpuPod>(&self) -> RhiResult<T> {
        mapped_read(self.cpu(), self.size)
    }

    /// View the mapped memory as `&[T]` (shared). Errors if not CPU-mapped or the size is
    /// not a whole number of `T`.
    pub fn as_slice<T: GpuPod>(&self) -> RhiResult<&[T]> {
        let ptr = self
            .cpu()
            .ok_or_else(|| RhiError::AllocationFailed("allocation is not CPU-mapped".into()))?;
        // SAFETY: `ptr` is valid for `self.size` bytes for the lifetime of `&self`.
        let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, self.size as usize) };
        <[T]>::ref_from_bytes(bytes)
            .map_err(|_| RhiError::AllocationFailed("size is not a multiple of element size".into()))
    }

    /// View the mapped memory as `&mut [T]` (exclusive). `&mut self` rules out CPU aliasing.
    pub fn as_mut_slice<T: GpuPod>(&mut self) -> RhiResult<&mut [T]> {
        let ptr = self
            .cpu()
            .ok_or_else(|| RhiError::AllocationFailed("allocation is not CPU-mapped".into()))?;
        // SAFETY: `ptr` is valid for `self.size` bytes; `&mut self` guarantees no other CPU
        // reference aliases it.
        let bytes = unsafe { std::slice::from_raw_parts_mut(ptr, self.size as usize) };
        <[T]>::mut_from_bytes(bytes)
            .map_err(|_| RhiError::AllocationFailed("size is not a multiple of element size".into()))
    }
}

/// Description for a user-land GPU allocator.
#[derive(Clone, Debug)]
pub struct GpuAllocatorDesc {
    /// Size of new backing blocks. Large enough blocks avoid runtime allocation churn.
    pub block_size: u64,
    /// Backing memory type.
    pub memory: MemoryType,
    /// Optional label prefix for backing allocations.
    pub label: Option<String>,
}

impl Default for GpuAllocatorDesc {
    fn default() -> Self {
        Self {
            block_size: 64 * 1024 * 1024,
            memory: MemoryType::Default,
            label: None,
        }
    }
}

/// A persistent suballocation from `GpuAllocator`.
#[derive(Debug)]
pub struct GpuSubAllocation {
    block: usize,
    offset: u64,
    size: u64,
    cpu_ptr: Option<*mut u8>,
    gpu_address: GpuAddress,
}

impl GpuSubAllocation {
    /// CPU-mapped pointer, if the backing memory is CPU visible.
    pub fn cpu(&self) -> Option<*mut u8> {
        self.cpu_ptr
    }

    /// GPU virtual address.
    pub fn gpu(&self) -> GpuAddress {
        self.gpu_address
    }

    /// Allocation size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Offset from the backing block base.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Upload a value into CPU-mapped memory (bounds-checked).
    pub fn upload<T: GpuPod>(&self, data: &T) -> RhiResult<()> {
        mapped_write(self.cpu_ptr, self.size, data.as_bytes())
    }

    /// Upload a slice into this suballocation's CPU-mapped memory (bounds-checked).
    pub fn upload_slice<T: GpuPod>(&self, data: &[T]) -> RhiResult<()> {
        mapped_write(self.cpu_ptr, self.size, data.as_bytes())
    }

    /// Read a value back from CPU-mapped memory.
    pub fn read<T: GpuPod>(&self) -> RhiResult<T> {
        mapped_read(self.cpu_ptr, self.size)
    }

    /// View the mapped memory as `&[T]` (shared).
    pub fn as_slice<T: GpuPod>(&self) -> RhiResult<&[T]> {
        let ptr = self
            .cpu_ptr
            .ok_or_else(|| RhiError::AllocationFailed("allocation is not CPU-mapped".into()))?;
        // SAFETY: `ptr` is valid for `self.size` bytes for the lifetime of `&self`.
        let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, self.size as usize) };
        <[T]>::ref_from_bytes(bytes)
            .map_err(|_| RhiError::AllocationFailed("size is not a multiple of element size".into()))
    }

    /// View the mapped memory as `&mut [T]` (exclusive). `&mut self` rules out CPU aliasing.
    pub fn as_mut_slice<T: GpuPod>(&mut self) -> RhiResult<&mut [T]> {
        let ptr = self
            .cpu_ptr
            .ok_or_else(|| RhiError::AllocationFailed("allocation is not CPU-mapped".into()))?;
        // SAFETY: `ptr` valid for `self.size` bytes; `&mut self` rules out CPU aliasing.
        let bytes = unsafe { std::slice::from_raw_parts_mut(ptr, self.size as usize) };
        <[T]>::mut_from_bytes(bytes)
            .map_err(|_| RhiError::AllocationFailed("size is not a multiple of element size".into()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FreeRange {
    offset: u64,
    size: u64,
}

struct GpuAllocatorBlock {
    allocation: Option<GpuAllocation>,
    cpu_base: Option<*mut u8>,
    gpu_base: GpuAddress,
    size: u64,
    free: Vec<FreeRange>,
}

/// User-land suballocator over large GPU memory blocks.
pub struct GpuAllocator<'a> {
    device: &'a Device,
    desc: GpuAllocatorDesc,
    blocks: Vec<GpuAllocatorBlock>,
}

impl<'a> GpuAllocator<'a> {
    /// Create an allocator. Backing memory is allocated lazily on first use.
    pub fn new(device: &'a Device, desc: GpuAllocatorDesc) -> Self {
        Self {
            device,
            desc,
            blocks: Vec::new(),
        }
    }

    /// Allocate a range with absolute GPU-address alignment.
    pub fn alloc(&mut self, size: u64, align: u64) -> RhiResult<GpuSubAllocation> {
        if size == 0 {
            return Err(RhiError::AllocationFailed(
                "allocation size must be non-zero".into(),
            ));
        }
        let align = align.max(1);
        if !align.is_power_of_two() {
            return Err(RhiError::AllocationFailed(format!(
                "alignment must be a power of two, got {align}"
            )));
        }

        for block_index in 0..self.blocks.len() {
            if let Some(allocation) =
                Self::alloc_from_block(&mut self.blocks[block_index], size, align, block_index)
            {
                return Ok(allocation);
            }
        }

        let block_index = self.blocks.len();
        self.blocks.push(self.create_block(size, align)?);
        Self::alloc_from_block(&mut self.blocks[block_index], size, align, block_index).ok_or_else(
            || {
                RhiError::AllocationFailed(
                    "new allocator block could not satisfy allocation".into(),
                )
            },
        )
    }

    /// Free a previous suballocation.
    pub fn free(&mut self, allocation: GpuSubAllocation) {
        let block = &mut self.blocks[allocation.block];
        insert_free_range(
            &mut block.free,
            FreeRange {
                offset: allocation.offset,
                size: allocation.size,
            },
        );
    }

    /// Release all suballocations without freeing backing blocks.
    pub fn reset(&mut self) {
        for block in &mut self.blocks {
            block.free.clear();
            block.free.push(FreeRange {
                offset: 0,
                size: block.size,
            });
        }
    }

    /// Bytes reserved in backing GPU blocks.
    pub fn reserved(&self) -> u64 {
        self.blocks.iter().map(|block| block.size).sum()
    }

    /// Bytes currently allocated from backing GPU blocks.
    pub fn used(&self) -> u64 {
        self.blocks
            .iter()
            .map(|block| block.size - block.free.iter().map(|range| range.size).sum::<u64>())
            .sum()
    }

    /// Number of backing blocks.
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    fn create_block(&self, size: u64, align: u64) -> RhiResult<GpuAllocatorBlock> {
        let worst_case_padding = align - 1;
        let min_block_size = size
            .checked_add(worst_case_padding)
            .ok_or_else(|| RhiError::AllocationFailed("allocator block size overflow".into()))?;
        let block_size = self.desc.block_size.max(min_block_size);
        let block_size = align_up_u64(block_size, 16)?;
        let allocation = self
            .device
            .malloc_aligned(block_size, 16, self.desc.memory)?;
        let cpu_base = allocation.cpu();
        let gpu_base = allocation.gpu();

        Ok(GpuAllocatorBlock {
            allocation: Some(allocation),
            cpu_base,
            gpu_base,
            size: block_size,
            free: vec![FreeRange {
                offset: 0,
                size: block_size,
            }],
        })
    }

    fn alloc_from_block(
        block: &mut GpuAllocatorBlock,
        size: u64,
        align: u64,
        block_index: usize,
    ) -> Option<GpuSubAllocation> {
        for range_index in 0..block.free.len() {
            let range = block.free[range_index];
            let aligned_gpu =
                align_up_u64(block.gpu_base.0.checked_add(range.offset)?, align).ok()?;
            let aligned_offset = aligned_gpu.checked_sub(block.gpu_base.0)?;
            let padding = aligned_offset.checked_sub(range.offset)?;
            let needed = padding.checked_add(size)?;
            if needed > range.size {
                continue;
            }

            let suffix_size = range.size - needed;
            if padding == 0 && suffix_size == 0 {
                block.free.swap_remove(range_index);
            } else if padding == 0 {
                block.free[range_index] = FreeRange {
                    offset: aligned_offset + size,
                    size: suffix_size,
                };
            } else if suffix_size == 0 {
                block.free[range_index].size = padding;
            } else {
                block.free[range_index].size = padding;
                block.free.insert(
                    range_index + 1,
                    FreeRange {
                        offset: aligned_offset + size,
                        size: suffix_size,
                    },
                );
            }

            let cpu_ptr = block
                .cpu_base
                .map(|ptr| unsafe { ptr.add(aligned_offset as usize) });
            return Some(GpuSubAllocation {
                block: block_index,
                offset: aligned_offset,
                size,
                cpu_ptr,
                gpu_address: GpuAddress(aligned_gpu),
            });
        }
        None
    }
}

impl Drop for GpuAllocator<'_> {
    fn drop(&mut self) {
        for block in &mut self.blocks {
            if let Some(allocation) = block.allocation.take() {
                self.device.free(allocation);
            }
        }
    }
}

fn align_up_u64(value: u64, align: u64) -> RhiResult<u64> {
    debug_assert!(align.is_power_of_two());
    value
        .checked_add(align - 1)
        .map(|v| v & !(align - 1))
        .ok_or_else(|| RhiError::AllocationFailed("alignment overflow".into()))
}

fn insert_free_range(free: &mut Vec<FreeRange>, range: FreeRange) {
    if range.size == 0 {
        return;
    }

    let index = free
        .binary_search_by_key(&range.offset, |existing| existing.offset)
        .unwrap_or_else(|index| index);
    free.insert(index, range);

    let mut i = index.saturating_sub(1);
    while i + 1 < free.len() {
        let current_end = free[i].offset + free[i].size;
        if current_end < free[i + 1].offset {
            i += 1;
            continue;
        }

        let next_end = free[i + 1].offset + free[i + 1].size;
        free[i].size = next_end.max(current_end) - free[i].offset;
        free.remove(i + 1);
    }
}

/// A persistent GPU buffer allocation.
///
/// Returns dual pointers: CPU mapped pointer (for upload buffers) + GPU address.
pub struct GpuBuffer {
    pub(crate) inner: GpuBufferInner,
}

pub(crate) enum GpuBufferInner {
    #[cfg(feature = "vulkan")]
    Vulkan(crate::backend::vulkan::memory::VulkanBuffer),
    #[cfg(feature = "metal")]
    Metal(crate::backend::metal::memory::MetalBuffer),
}

impl GpuBuffer {
    /// CPU-mapped pointer (`None` for `GpuOnly`).
    pub fn cpu(&self) -> Option<*mut u8> {
        backend_dispatch!(&self.inner, GpuBufferInner, b => b.mapped_ptr())
    }

    /// GPU virtual address for shader access.
    pub fn gpu(&self) -> GpuAddress {
        backend_dispatch!(&self.inner, GpuBufferInner, b => b.gpu_address())
    }

    /// Buffer size in bytes.
    pub fn size(&self) -> u64 {
        backend_dispatch!(&self.inner, GpuBufferInner, b => b.size())
    }
}

/// Dual-pointer transient allocation from the bump allocator (the doc's `{ cpu, gpu }`).
#[derive(Clone, Copy, Debug)]
pub struct TransientAllocation {
    pub cpu: *mut u8,
    pub gpu: GpuAddress,
    pub size: u64,
}

impl TransientAllocation {
    /// Write a value into CPU-mapped memory (bounds-checked).
    pub fn upload<T: GpuPod>(&self, data: &T) -> RhiResult<()> {
        mapped_write(Some(self.cpu), self.size, data.as_bytes())
    }

    /// Write a slice into CPU-mapped memory (bounds-checked).
    pub fn upload_slice<T: GpuPod>(&self, data: &[T]) -> RhiResult<()> {
        mapped_write(Some(self.cpu), self.size, data.as_bytes())
    }
}

/// Per-frame bump allocator over a large GpuBuffer.
pub struct BumpAllocator {
    buffer: GpuBuffer,
    offset: u64,
    capacity: u64,
}

impl BumpAllocator {
    /// Create a new bump allocator with the given buffer.
    pub fn new(buffer: GpuBuffer) -> Self {
        let capacity = buffer.size();
        Self {
            buffer,
            offset: 0,
            capacity,
        }
    }

    /// Allocate `size` bytes with the given alignment.
    /// Returns None if the allocator is full.
    pub fn alloc(&mut self, size: u64, align: u64) -> Option<TransientAllocation> {
        let align = align.max(1);
        if !align.is_power_of_two() {
            return None;
        }

        let aligned_offset = self.offset.checked_add(align - 1)? & !(align - 1);
        let end = aligned_offset.checked_add(size)?;
        if end > self.capacity {
            return None;
        }

        let cpu = self.buffer.cpu()?;
        let gpu = self.buffer.gpu().offset(aligned_offset);
        let cpu = unsafe { cpu.add(aligned_offset as usize) };

        self.offset = end;

        Some(TransientAllocation { cpu, gpu, size })
    }

    /// Reset the allocator for a new frame.
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// The underlying buffer's base GPU address.
    pub fn gpu(&self) -> GpuAddress {
        self.buffer.gpu()
    }

    /// How many bytes have been allocated so far.
    pub fn used(&self) -> u64 {
        self.offset
    }

    /// Total capacity in bytes.
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Consume the allocator and return the backing buffer.
    pub fn into_buffer(self) -> GpuBuffer {
        self.buffer
    }
}

// Allocator behaviour is covered by the black-box headless tests in `tests/memory.rs`.
