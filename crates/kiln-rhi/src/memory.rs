//! GPU memory allocation: buffers, typed allocations, and the bump allocator.

use crate::types::GpuAddress;
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

/// Description for creating a GPU buffer.
#[derive(Clone, Debug, Default)]
pub struct BufferDesc {
    pub size: u64,
    pub memory: MemoryType,
    pub label: Option<String>,
}

/// A GPU memory allocation.
pub struct GpuAllocation {
    pub(crate) buffer: GpuBuffer,
    pub(crate) offset: u64,
    pub(crate) size: u64,
}

impl GpuAllocation {
    /// CPU mapping, or `None` for `GpuOnly` memory.
    pub fn cpu(&self) -> Option<*mut u8> {
        self.buffer
            .cpu()
            .zip(usize::try_from(self.offset).ok())
            .map(|(ptr, offset)| unsafe { ptr.add(offset) })
    }

    /// GPU virtual address.
    pub fn gpu(&self) -> GpuAddress {
        self.buffer.gpu().offset(self.offset)
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

/// A persistent GPU buffer.
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

/// A transient slice of mapped GPU memory.
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

/// Linear allocator over a mapped GPU buffer.
pub struct BumpAllocator {
    buffer: GpuBuffer,
    cpu_base: Option<*mut u8>,
    gpu_base: GpuAddress,
    offset: u64,
    capacity: u64,
}

impl BumpAllocator {
    /// Create a new bump allocator with the given buffer.
    pub fn new(buffer: GpuBuffer) -> Self {
        let cpu_base = buffer.cpu();
        let gpu_base = buffer.gpu();
        let capacity = buffer.size();
        Self {
            buffer,
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
        let gpu = self.gpu_base.offset(aligned_offset);
        let cpu_offset = usize::try_from(aligned_offset).ok()?;
        let cpu = unsafe { cpu.add(cpu_offset) };

        self.offset = end;

        Some(TransientAllocation { cpu, gpu, size })
    }

    /// Reset the allocator for a new frame.
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// The underlying buffer's base GPU address.
    pub fn gpu(&self) -> GpuAddress {
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
    pub fn into_buffer(self) -> GpuBuffer {
        self.buffer
    }
}

fn aligned_bump_range(offset: u64, size: u64, align: u64, capacity: u64) -> Option<(u64, u64)> {
    let align = align.max(1);
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
    }

    #[test]
    fn bump_range_aligns_without_overflow() {
        assert_eq!(aligned_bump_range(17, 8, 16, 64), Some((32, 40)));
        assert_eq!(aligned_bump_range(17, 33, 16, 64), None);
    }
}
