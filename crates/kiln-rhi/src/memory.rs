//! GPU memory allocation and pointer arithmetic.

use crate::types::{GpuPtr, MAX_FRAMES_IN_FLIGHT};
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
        return Err(RhiError::AllocationFailed(
            format!(
                "upload of {} bytes exceeds mapped region ({capacity} bytes)",
                bytes.len()
            )
            .into(),
        ));
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
        return Err(RhiError::AllocationFailed(
            format!("read of {n} bytes exceeds mapped region ({capacity} bytes)").into(),
        ));
    }
    // SAFETY: `src` is valid for `capacity` >= `n` bytes; `T: FromBytes`.
    let bytes = unsafe { std::slice::from_raw_parts(src.cast_const(), n) };
    T::read_from_bytes(bytes).map_err(|_| RhiError::AllocationFailed("read size mismatch".into()))
}

/// Reinterpret the `bytes` mapped bytes at `ptr` as `[T]`.
///
/// # Safety
/// `ptr` must be valid for `bytes` over `'a`, and nothing may write those bytes while the
/// returned slice lives.
unsafe fn slice_from_raw<'a, T: GpuPod>(ptr: *mut u8, bytes: u64) -> RhiResult<&'a [T]> {
    let len = usize::try_from(bytes).map_err(|_| {
        RhiError::AllocationFailed("mapped region does not fit the host address space".into())
    })?;
    let raw = unsafe { std::slice::from_raw_parts(ptr.cast_const(), len) };
    <[T]>::ref_from_bytes(raw)
        .map_err(|_| RhiError::AllocationFailed("size is not a multiple of element size".into()))
}

/// Mutable counterpart of [`slice_from_raw`].
///
/// # Safety
/// As [`slice_from_raw`], and the slice must have exclusive access for `'a`.
unsafe fn slice_from_raw_mut<'a, T: GpuPod>(ptr: *mut u8, bytes: u64) -> RhiResult<&'a mut [T]> {
    let len = usize::try_from(bytes).map_err(|_| {
        RhiError::AllocationFailed("mapped region does not fit the host address space".into())
    })?;
    let raw = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
    <[T]>::mut_from_bytes(raw)
        .map_err(|_| RhiError::AllocationFailed("size is not a multiple of element size".into()))
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

/// Alignment used when a caller does not ask for one; wide enough for a `float4`.
pub const DEFAULT_ALIGN: u64 = 16;

/// Owned GPU memory holding `T`s: an optional CPU mapping, a GPU address, and a byte length. The
/// backend buffer or heap behind the address is deliberately not part of the API.
///
/// `T` is what the allocation was created for, so [`gpu`](Self::gpu) and the CPU accessors are
/// typed without a cast. Raw byte regions are `Allocation<u8>`, the default; [`cast`](Self::cast)
/// reinterprets one as another type.
pub struct Allocation<T = u8> {
    pub(crate) inner: AllocationInner,
    pub(crate) _owner: Option<std::rc::Rc<crate::device::DeviceInner>>,
    pub(crate) size: u64,
    pub(crate) _type: PhantomData<fn() -> T>,
}

backend_enum!(AllocationInner {
    vulkan: crate::backend::vulkan::memory::VulkanBuffer,
    metal: crate::backend::metal::memory::MetalBuffer
});

impl<T> Allocation<T> {
    fn cpu(&self) -> Option<*mut u8> {
        self.inner.mapped_ptr()
    }

    /// GPU virtual address; present even for `GpuOnly`. Aligned to the alignment the allocation
    /// was created with.
    pub fn gpu(&self) -> GpuPtr<T> {
        self.inner.gpu_address().cast()
    }

    /// Size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// How many whole `T`s fit.
    pub fn len(&self) -> usize {
        (self.size / size_of::<T>().max(1) as u64) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The same memory, viewed as `U`s.
    pub fn cast<U>(self) -> Allocation<U> {
        Allocation {
            inner: self.inner,
            _owner: self._owner,
            size: self.size,
            _type: PhantomData,
        }
    }

    /// Name the allocation in GPU captures. A no-op on Vulkan, where an allocation is a range
    /// inside a shared buffer and has no object of its own to name.
    pub fn labeled(self, label: &str) -> Self {
        self.inner.set_label(label);
        self
    }

    /// Handle to the mapped memory, or `None` for `GpuOnly`. The mutable borrow excludes
    /// allocation reads while a writable handle is live.
    pub fn mapped(&mut self) -> Option<Mapped<'_, T>> {
        self.cpu().map(|cpu| Mapped {
            cpu,
            gpu: self.gpu(),
            bytes: self.size,
            _borrow: PhantomData,
        })
    }
}

impl<T: GpuPod> Allocation<T> {
    /// Write one `T` at the start. Bounds-checked; the caller orders the write before the submit
    /// that reads it.
    pub fn upload(&mut self, value: &T) -> RhiResult<()> {
        mapped_write(self.cpu(), self.size, value.as_bytes())
    }

    pub fn upload_slice(&mut self, data: &[T]) -> RhiResult<()> {
        mapped_write(self.cpu(), self.size, data.as_bytes())
    }

    /// Read the first `T` back, e.g. from `Readback` memory after a GPU write.
    pub fn read(&self) -> RhiResult<T> {
        mapped_read(self.cpu(), self.size)
    }

    /// Errors if not mapped, or if the size is not a whole number of `T`.
    pub fn as_slice(&self) -> RhiResult<&[T]> {
        let ptr = self
            .cpu()
            .ok_or_else(|| RhiError::AllocationFailed("allocation is not CPU-mapped".into()))?;
        // SAFETY: valid for `self.size`; `&self` rules out a concurrent CPU writer.
        unsafe { slice_from_raw(ptr, self.size) }
    }

    /// `&mut self` rules out CPU aliasing.
    pub fn as_mut_slice(&mut self) -> RhiResult<&mut [T]> {
        let ptr = self
            .cpu()
            .ok_or_else(|| RhiError::AllocationFailed("allocation is not CPU-mapped".into()))?;
        // SAFETY: valid for `self.size`; `&mut self` rules out any other CPU reference.
        unsafe { slice_from_raw_mut(ptr, self.size) }
    }
}

/// One mapped region, holding the CPU and GPU addresses for the same bytes. [`offset`](Self::offset)
/// advances both, so the two sides cannot drift apart and the stride comes from `T`.
///
/// The borrow pins a [`BumpAllocator`] against [`reset`](BumpAllocator::reset), so recycling the
/// arena under a live handle is a compile error.
///
/// `Copy`, so callers can offset repeatedly from one base. Reads and writes go through the
/// bounds-checked methods; borrow the whole [`Allocation`] for slice access.
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
    pub fn gpu(&self) -> GpuPtr<T> {
        self.gpu
    }

    /// CPU address, for writes this type cannot express.
    #[inline]
    pub fn cpu(&self) -> *mut u8 {
        self.cpu
    }

    #[inline]
    pub fn byte_len(&self) -> u64 {
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

impl<T> Mapped<'_, T> {
    /// Advance both addresses by `count` elements.
    #[inline]
    pub fn offset(self, count: u64) -> Self {
        self.byte_offset((size_of::<T>() as u64).saturating_mul(count))
    }
}

impl<T> Copy for Mapped<'_, T> {}

impl<T> Clone for Mapped<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: GpuPod> Mapped<'_, T> {
    /// Bounds-checked. The caller orders the write before the submit that reads it.
    pub fn write(&self, value: &T) -> RhiResult<()> {
        mapped_write(Some(self.cpu), self.bytes, value.as_bytes())
    }

    pub fn write_slice(&self, values: &[T]) -> RhiResult<()> {
        mapped_write(Some(self.cpu), self.bytes, values.as_bytes())
    }

    /// Read back, e.g. from `Readback` memory after a GPU write.
    pub fn read(&self) -> RhiResult<T> {
        mapped_read(Some(self.cpu), self.bytes)
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

    /// Room for one `T`, aligned to `T` and to at least [`DEFAULT_ALIGN`]. `None` when full.
    pub fn alloc<T>(&self) -> Option<Mapped<'_, T>> {
        self.alloc_array::<T>(1)
    }

    /// Room for `count` contiguous `T`s, aligned as [`alloc`](Self::alloc). `None` when full.
    pub fn alloc_array<T>(&self, count: usize) -> Option<Mapped<'_, T>> {
        let size = size_of::<T>().checked_mul(count)? as u64;
        let align = (align_of::<T>() as u64).max(DEFAULT_ALIGN);
        self.alloc_bytes(size, align).map(Mapped::cast)
    }

    /// Copy `value` into the arena and return its GPU address. `None` when full.
    pub fn upload<T: GpuPod>(&self, value: &T) -> Option<GpuPtr<T>> {
        let mapped = self.alloc::<T>()?;
        mapped.write(value).ok()?;
        Some(mapped.gpu())
    }

    /// Copy `values` into the arena and return the GPU address of the first. `None` when full.
    pub fn upload_slice<T: GpuPod>(&self, values: &[T]) -> Option<GpuPtr<T>> {
        let mapped = self.alloc_array::<T>(values.len())?;
        mapped.write_slice(values).ok()?;
        Some(mapped.gpu())
    }

    /// `size` bytes at a power-of-two `align`. `None` when full.
    pub fn alloc_bytes(&self, size: u64, align: u64) -> Option<Mapped<'_, u8>> {
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

/// One [`BumpAllocator`] per frame in flight, for per-frame root data and staging.
///
/// Reset a slot at the start of the frame that uses it, once that slot's previous submission has
/// completed: after [`Queue::acquire_image`](crate::Queue::acquire_image) or
/// [`Device::wait_for_frame`](crate::Device::wait_for_frame) for it. A
/// [`DeviceResource`](crate::DeviceResource): release it with
/// [`Device::destroy`](crate::Device::destroy).
pub struct FrameArena {
    pub(crate) slots: [BumpAllocator; MAX_FRAMES_IN_FLIGHT],
}

impl FrameArena {
    /// `bytes_per_frame` of upload memory for each frame in flight.
    pub fn new(device: &crate::Device, bytes_per_frame: u64) -> RhiResult<Self> {
        let mut slots = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            match device.allocate_bytes(bytes_per_frame, MemoryType::Upload) {
                Ok(allocation) => slots.push(BumpAllocator::new(allocation)),
                Err(error) => {
                    for slot in slots {
                        device.destroy(slot.into_allocation());
                    }
                    return Err(error);
                }
            }
        }
        let Ok(slots) = slots.try_into() else {
            unreachable!("one arena was made per frame in flight")
        };
        Ok(Self { slots })
    }

    /// Name each slot `"{label}-{slot}"` in GPU captures.
    pub fn labeled(self, label: &str) -> Self {
        for (index, slot) in self.slots.iter().enumerate() {
            slot.allocation.inner.set_label(&format!("{label}-{index}"));
        }
        self
    }

    /// Start a new frame in `slot`, reclaiming everything it held.
    pub fn reset(&mut self, slot: usize) {
        self.slots[slot].reset();
    }

    /// The arena behind `slot`, for [`alloc`](BumpAllocator::alloc) and friends.
    pub fn slot(&self, slot: usize) -> &BumpAllocator {
        &self.slots[slot]
    }

    /// Copy `value` into `slot` and return its GPU address. `None` when the slot is full.
    pub fn upload<T: GpuPod>(&self, slot: usize, value: &T) -> Option<GpuPtr<T>> {
        self.slots[slot].upload(value)
    }

    /// Copy `values` into `slot` and return the address of the first. `None` when it is full.
    pub fn upload_slice<T: GpuPod>(&self, slot: usize, values: &[T]) -> Option<GpuPtr<T>> {
        self.slots[slot].upload_slice(values)
    }
}
