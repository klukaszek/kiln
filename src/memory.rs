use crate::types::GpuAddress;
use crate::{RhiError, RhiResult};
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
        <[T]>::ref_from_bytes(bytes).map_err(|_| {
            RhiError::AllocationFailed("size is not a multiple of element size".into())
        })
    }

    /// View the mapped memory as `&mut [T]` (exclusive). `&mut self` rules out CPU aliasing.
    pub fn as_mut_slice<T: GpuPod>(&mut self) -> RhiResult<&mut [T]> {
        let ptr = self
            .cpu()
            .ok_or_else(|| RhiError::AllocationFailed("allocation is not CPU-mapped".into()))?;
        // SAFETY: `ptr` is valid for `self.size` bytes; `&mut self` guarantees no other CPU
        // reference aliases it.
        let bytes = unsafe { std::slice::from_raw_parts_mut(ptr, self.size as usize) };
        <[T]>::mut_from_bytes(bytes).map_err(|_| {
            RhiError::AllocationFailed("size is not a multiple of element size".into())
        })
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

// Allocator behaviour is covered by the black-box headless tests in `tests/memory.rs`;
// the bump allocator's end-to-end use as per-draw root data is exercised by
// `graphics_root_from_bump_allocator` in `tests/graphics.rs`.
