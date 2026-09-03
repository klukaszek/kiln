//! GPU memory allocation and pointer arithmetic.

use crate::types::GpuPtr;
use crate::{RhiError, RhiResult};
use std::marker::PhantomData;
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

/// Where an allocation lives. No default: `Upload` for memory the GPU reads hot is a silent
/// bandwidth cost, not an error. (D3D12's `DEFAULT` is this `GpuOnly`.)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryType {
    /// Host-visible, coherent. Uniforms, staging, draw args, descriptors.
    Upload,
    /// Device-local, not mappable. Textures, vertex data, large persistent buffers.
    GpuOnly,
    /// Host-visible, cached on read. Screenshots, feedback, GPGPU output.
    Readback,
}

/// Description for creating a GPU allocation.
#[derive(Clone, Debug)]
pub struct AllocationDesc {
    pub size: u64,
    /// Power of two. The RHI pads the backing allocation so `gpu()` comes back aligned.
    pub align: u64,
    pub memory: MemoryType,
    pub label: Option<String>,
}

impl Default for AllocationDesc {
    fn default() -> Self {
        Self {
            size: 0,
            align: DEFAULT_ALIGN,
            // Fail-soft: always mappable, so a desc that forgets to choose still runs.
            memory: MemoryType::Upload,
            label: None,
        }
    }
}

/// Alignment used when a caller does not ask for one; wide enough for a `float4`.
pub const DEFAULT_ALIGN: u64 = 16;

/// Owned GPU memory: an optional CPU mapping, a GPU address, and a byte length. The backend
/// buffer or heap behind the address is deliberately not part of the API.
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

    fn cpu(&self) -> Option<*mut u8> {
        self.base_cpu()
            .zip(usize::try_from(self.offset).ok())
            .map(|(ptr, offset)| unsafe { ptr.add(offset) })
    }

    /// GPU virtual address; present even for `GpuOnly`.
    pub fn gpu(&self) -> GpuPtr<u8> {
        self.base_gpu().byte_add(self.offset)
    }

    /// Typed handle to the mapped bytes, or `None` for `GpuOnly`. Checked once here rather than
    /// on every write.
    pub fn mapped<T>(&self) -> Option<Mapped<'_, T>> {
        self.cpu().map(|cpu| Mapped {
            cpu,
            gpu: self.gpu().cast(),
            bytes: self.size,
            _borrow: PhantomData,
        })
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    /// Bounds-checked. The caller orders the write before the submit that reads it.
    pub fn upload<T: GpuPod>(&mut self, value: &T) -> RhiResult<()> {
        mapped_write(self.cpu(), self.size, value.as_bytes())
    }

    pub fn upload_slice<T: GpuPod>(&mut self, data: &[T]) -> RhiResult<()> {
        mapped_write(self.cpu(), self.size, data.as_bytes())
    }

    /// Read back, e.g. from `Readback` memory after a GPU write.
    pub fn read<T: GpuPod>(&self) -> RhiResult<T> {
        mapped_read(self.cpu(), self.size)
    }

    /// Errors if not mapped, or if the size is not a whole number of `T`.
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

    /// `&mut self` rules out CPU aliasing.
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

/// One mapped region, holding the CPU and GPU addresses for the same bytes. [`offset`](Self::offset)
/// advances both, so the two sides cannot drift apart and the stride comes from `T`.
///
/// The borrow pins a [`BumpAllocator`] against [`reset`](BumpAllocator::reset) while the handle
/// lives, making arena recycling under a live handle a compile error.
pub struct Mapped<'a, T> {
    cpu: *mut u8,
    gpu: GpuPtr<T>,
    /// Bytes to the end of the region; writes are checked against it.
    bytes: u64,
    _borrow: PhantomData<&'a ()>,
}

impl<'a, T> Mapped<'a, T> {
    /// GPU address of this position.
    #[inline]
    pub fn gpu(self) -> GpuPtr<T> {
        self.gpu
    }

    /// CPU address, for writes this type cannot express.
    #[inline]
    pub fn cpu(self) -> *mut u8 {
        self.cpu
    }

    #[inline]
    pub fn byte_len(self) -> u64 {
        self.bytes
    }

    /// Retype without moving either address.
    #[inline]
    pub fn cast<U>(self) -> Mapped<'a, U> {
        Mapped {
            cpu: self.cpu,
            gpu: self.gpu.cast(),
            bytes: self.bytes,
            _borrow: PhantomData,
        }
    }

    /// Advance both addresses, clamped to the region end so an over-long offset yields an empty
    /// handle rather than a pointer past the end.
    #[inline]
    pub fn byte_offset(self, bytes: u64) -> Self {
        let step = bytes.min(self.bytes);
        Self {
            // SAFETY: `step <= self.bytes`, so this stays within the mapped region.
            cpu: unsafe { self.cpu.add(step as usize) },
            gpu: self.gpu.cast::<u8>().byte_add(step).cast(),
            bytes: self.bytes - step,
            _borrow: PhantomData,
        }
    }
}

impl<'a, T> Mapped<'a, T> {
    #[inline]
    pub fn len(self) -> u64 {
        match size_of::<T>() as u64 {
            0 => 0,
            stride => self.bytes / stride,
        }
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    /// Advance both addresses by `count` elements.
    #[inline]
    pub fn offset(self, count: u64) -> Self {
        self.byte_offset((size_of::<T>() as u64).saturating_mul(count))
    }
}

impl<'a, T: GpuPod> Mapped<'a, T> {
    /// Bounds-checked. The caller orders the write before the submit that reads it.
    pub fn write(self, value: &T) -> RhiResult<()> {
        mapped_write(Some(self.cpu), self.bytes, value.as_bytes())
    }

    pub fn write_slice(self, values: &[T]) -> RhiResult<()> {
        mapped_write(Some(self.cpu), self.bytes, values.as_bytes())
    }

    /// Read back, e.g. from `Readback` memory after a GPU write.
    pub fn read(self) -> RhiResult<T> {
        mapped_read(Some(self.cpu), self.bytes)
    }

    pub fn as_slice(self) -> RhiResult<&'a [T]> {
        let size = usize::try_from(self.bytes).map_err(|_| {
            RhiError::AllocationFailed("mapped region does not fit the host address space".into())
        })?;
        // SAFETY: valid for `self.bytes` over this handle's borrow.
        let bytes = unsafe { std::slice::from_raw_parts(self.cpu as *const u8, size) };
        <[T]>::ref_from_bytes(bytes).map_err(|_| {
            RhiError::AllocationFailed("size is not a multiple of element size".into())
        })
    }

    /// `&mut self` stops one handle handing out two aliasing slices; overlapping handles are the
    /// caller's business.
    pub fn as_mut_slice(&mut self) -> RhiResult<&mut [T]> {
        let size = usize::try_from(self.bytes).map_err(|_| {
            RhiError::AllocationFailed("mapped region does not fit the host address space".into())
        })?;
        // SAFETY: valid for `self.bytes`; `&mut self` rules out a second slice.
        let bytes = unsafe { std::slice::from_raw_parts_mut(self.cpu, size) };
        <[T]>::mut_from_bytes(bytes).map_err(|_| {
            RhiError::AllocationFailed("size is not a multiple of element size".into())
        })
    }
}

impl<T> Copy for Mapped<'_, T> {}

impl<T> Clone for Mapped<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> std::fmt::Debug for Mapped<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mapped")
            .field("cpu", &self.cpu)
            .field("gpu", &self.gpu)
            .field("bytes", &self.bytes)
            .finish()
    }
}

/// Linear allocator over a mapped GPU buffer.
///
/// [`alloc`](Self::alloc) takes `&self` so handles can coexist; [`reset`](Self::reset) takes
/// `&mut self`, so the borrow checker refuses to recycle the arena under a live handle.
pub struct BumpAllocator {
    allocation: Allocation,
    cpu_base: Option<*mut u8>,
    gpu_base: GpuPtr<u8>,
    offset: std::cell::Cell<u64>,
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
            offset: std::cell::Cell::new(0),
            capacity,
        }
    }

    /// Allocate `size` bytes with the given power-of-two alignment.
    /// Returns `None` when the allocator has no room.
    pub fn alloc(&self, size: u64, align: u64) -> Option<Mapped<'_, u8>> {
        let (aligned_offset, end) =
            aligned_bump_range(self.offset.get(), size, align, self.capacity)?;

        let cpu = self.cpu_base?;
        let gpu = self.gpu_base.byte_add(aligned_offset);
        let cpu_offset = usize::try_from(aligned_offset).ok()?;
        // SAFETY: `aligned_bump_range` kept `end <= capacity`.
        let cpu = unsafe { cpu.add(cpu_offset) };

        self.offset.set(end);

        Some(Mapped {
            cpu,
            gpu,
            bytes: size,
            _borrow: PhantomData,
        })
    }

    /// Reset for a new frame. Will not compile while a [`Mapped`] from it is live.
    pub fn reset(&mut self) {
        self.offset.set(0);
    }

    /// The underlying buffer's base GPU address.
    pub fn gpu(&self) -> GpuPtr<u8> {
        self.gpu_base
    }

    /// How many bytes have been allocated so far.
    pub fn used(&self) -> u64 {
        self.offset.get()
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
