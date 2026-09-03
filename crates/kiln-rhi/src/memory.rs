//! GPU memory allocation and pointer arithmetic.

use crate::types::GpuPtr;
use crate::{RhiError, RhiResult};
use zerocopy::{FromBytes, IntoBytes};

/// A type that can be copied directly between CPU and GPU memory.
pub trait GpuPod: IntoBytes + FromBytes + zerocopy::Immutable {}
impl<T> GpuPod for T where T: IntoBytes + FromBytes + zerocopy::Immutable {}

/// Copy `bytes` into a CPU-mapped region, bounds-checked.
fn mapped_write(ptr: Option<*mut u8>, capacity: u64, bytes: &[u8]) -> RhiResult<()> {
    let byte_count = u64::try_from(bytes.len()).map_err(|_| {
        RhiError::AllocationFailed("upload size does not fit in a GPU allocation".into())
    })?;
    if byte_count > capacity {
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

/// Description for creating a GPU allocation.
#[derive(Clone, Debug, Default)]
pub struct AllocationDesc {
    pub size: u64,
    pub memory: MemoryType,
    pub label: Option<String>,
}

/// Owned GPU memory: an optional CPU mapping, a GPU address, and a byte length.
///
/// This is the only public buffer-memory object. Subranges are expressed by pointer arithmetic;
/// the backend buffer or heap that supplies the address is deliberately not part of the API.
pub struct Allocation {
    pub(crate) inner: AllocationInner,
    pub(crate) _owner: Option<std::rc::Rc<crate::device::DeviceInner>>,
    pub(crate) offset: u64,
    pub(crate) size: u64,
}

pub(crate) enum AllocationInner {
    #[cfg(feature = "vulkan")]
    Vulkan(crate::backend::vulkan::memory::VulkanBuffer),
    #[cfg(feature = "metal")]
    Metal(crate::backend::metal::memory::MetalBuffer),
}

impl Allocation {
    fn base_cpu(&self) -> Option<*mut u8> {
        backend_dispatch!(&self.inner, AllocationInner, allocation => allocation.mapped_ptr())
    }

    fn base_gpu(&self) -> GpuPtr<u8> {
        backend_dispatch!(&self.inner, AllocationInner, allocation => allocation.gpu_address())
    }

    /// CPU mapping, or `None` for `GpuOnly` memory.
    pub fn cpu(&self) -> Option<*mut u8> {
        self.base_cpu()
            .zip(usize::try_from(self.offset).ok())
            .map(|(ptr, offset)| unsafe { ptr.add(offset) })
    }

    /// GPU virtual address.
    pub fn gpu(&self) -> GpuPtr<u8> {
        self.base_gpu().byte_add(self.offset)
    }

    /// Typed GPU pointer to the first byte of this allocation.
    #[inline]
    pub fn ptr<T>(&self) -> crate::types::GpuPtr<T> {
        self.gpu().cast()
    }

    /// Allocation size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Upload a value into CPU-mapped memory (bounds-checked). Caller orders the write before
    /// the dependent submit.
    pub fn upload<T: GpuPod>(&mut self, value: &T) -> RhiResult<()> {
        mapped_write(self.cpu(), self.size, value.as_bytes())
    }

    /// Upload a slice into this allocation's CPU-mapped memory (bounds-checked).
    pub fn upload_slice<T: GpuPod>(&mut self, data: &[T]) -> RhiResult<()> {
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
        let size = usize::try_from(self.size).map_err(|_| {
            RhiError::AllocationFailed(
                "allocation size does not fit in the host address space".into(),
            )
        })?;
        // SAFETY: `ptr` is valid for `self.size` bytes for the lifetime of `&self`.
        let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, size) };
        <[T]>::ref_from_bytes(bytes).map_err(|_| {
            RhiError::AllocationFailed("size is not a multiple of element size".into())
        })
    }

    /// View the mapped memory as `&mut [T]` (exclusive). `&mut self` rules out CPU aliasing.
    pub fn as_mut_slice<T: GpuPod>(&mut self) -> RhiResult<&mut [T]> {
        let ptr = self
            .cpu()
            .ok_or_else(|| RhiError::AllocationFailed("allocation is not CPU-mapped".into()))?;
        let size = usize::try_from(self.size).map_err(|_| {
            RhiError::AllocationFailed(
                "allocation size does not fit in the host address space".into(),
            )
        })?;
        // SAFETY: `ptr` is valid for `self.size` bytes; `&mut self` guarantees no other CPU
        // reference aliases it.
        let bytes = unsafe { std::slice::from_raw_parts_mut(ptr, size) };
        <[T]>::mut_from_bytes(bytes).map_err(|_| {
            RhiError::AllocationFailed("size is not a multiple of element size".into())
        })
    }
}

/// A transient slice of mapped GPU memory.
#[derive(Clone, Copy, Debug)]
pub struct TransientAllocation {
    pub cpu: *mut u8,
    pub gpu: GpuPtr<u8>,
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

/// Linear allocator over a mapped GPU buffer.
pub struct BumpAllocator {
    allocation: Allocation,
    cpu_base: Option<*mut u8>,
    gpu_base: GpuPtr<u8>,
    offset: u64,
    capacity: u64,
}

impl BumpAllocator {
    /// Create a new bump allocator with the given buffer.
    pub fn new(allocation: Allocation) -> Self {
        let cpu_base = allocation.cpu();
        let gpu_base = allocation.gpu();
        let capacity = allocation.size();
        Self {
            allocation,
            cpu_base,
            gpu_base,
            offset: 0,
            capacity,
        }
    }

    /// Allocate `size` bytes with the given power-of-two alignment.
    /// Returns `None` when the allocator has no room.
    pub fn alloc(&mut self, size: u64, align: u64) -> Option<TransientAllocation> {
        let (aligned_offset, end) = aligned_bump_range(self.offset, size, align, self.capacity)?;

        let cpu = self.cpu_base?;
        let gpu = self.gpu_base.byte_add(aligned_offset);
        let cpu_offset = usize::try_from(aligned_offset).ok()?;
        let cpu = unsafe { cpu.add(cpu_offset) };

        self.offset = end;

        Some(TransientAllocation { cpu, gpu, size })
    }

    /// Allocate space for `count` values with their natural alignment.
    pub fn alloc_array<T>(&mut self, count: usize) -> Option<TransientAllocation> {
        let size = std::mem::size_of::<T>().checked_mul(count)? as u64;
        self.alloc(size, std::mem::align_of::<T>() as u64)
    }

    /// Reset the allocator for a new frame.
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// The underlying buffer's base GPU address.
    pub fn gpu(&self) -> GpuPtr<u8> {
        self.gpu_base
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
    pub fn into_allocation(self) -> Allocation {
        self.allocation
    }
}

fn aligned_bump_range(offset: u64, size: u64, align: u64, capacity: u64) -> Option<(u64, u64)> {
    if !align.is_power_of_two() {
        return None;
    }
    let remainder = offset % align;
    let padding = if remainder == 0 { 0 } else { align - remainder };
    let aligned_offset = offset.checked_add(padding)?;
    let end = aligned_offset.checked_add(size)?;
    (end <= capacity).then_some((aligned_offset, end))
}

#[cfg(test)]
mod tests {
    use super::aligned_bump_range;

    #[test]
    fn bump_range_rejects_overflow() {
        assert_eq!(aligned_bump_range(u64::MAX - 7, 16, 16, u64::MAX), None);
        assert_eq!(aligned_bump_range(8, u64::MAX, 1, u64::MAX), None);
        assert_eq!(aligned_bump_range(0, 8, 0, 64), None);
        assert_eq!(aligned_bump_range(0, 8, 3, 64), None);
    }

    #[test]
    fn bump_range_aligns_without_overflow() {
        assert_eq!(aligned_bump_range(17, 8, 16, 64), Some((32, 40)));
        assert_eq!(aligned_bump_range(17, 33, 16, 64), None);
    }
}
