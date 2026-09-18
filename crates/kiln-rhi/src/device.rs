//! Device creation and the top-level resource management API.

use crate::accel::AccelerationStructure;
use crate::command::CommandBuffer;
use crate::error::{RhiError, RhiResult};
use crate::memory::{Allocation, AllocationDesc, GpuPod, MemoryType};
use crate::pipeline::{
    ComputePso, ComputePsoDesc, GraphicsPso, GraphicsPsoDesc, MeshletPso, MeshletPsoDesc,
};
use crate::query::QueryPool;
use crate::queue::Queue;
use crate::sampler::{Sampler, SamplerDesc};
use crate::shader::{ShaderModule, ShaderModuleDesc};
use crate::surface::{Surface, SurfaceDesc};
use crate::swapchain::{Swapchain, SwapchainDesc};
use crate::sync::TimelineSemaphore;
use crate::texture::{Texture, TextureDesc, TextureSizeAlign, TextureViewDesc, ViewKind};
use crate::types::{BindlessCapacity, BlasDesc, GpuPtr, TextureHandle, TlasDesc, TlasInstance};
use std::rc::Rc;

/// Which GPU backend to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Vulkan 1.4, with the descriptor-heap and device-address command extensions.
    Vulkan,
    /// Metal 4 (Apple platforms only).
    Metal,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vulkan => write!(f, "Vulkan"),
            Self::Metal => write!(f, "Metal"),
        }
    }
}

/// Description for creating a device.
pub struct DeviceDesc<'a> {
    /// Enable Vulkan validation. Metal validation is configured at process launch through
    /// Xcode or `MTL_DEBUG_LAYER=1`; this flag cannot toggle Metal's process-wide layer.
    pub validation: bool,
    pub label: Option<&'a str>,
    /// Size of the bindless heaps, allocated in full at device creation.
    pub bindless: BindlessCapacity,
}

impl Default for DeviceDesc<'_> {
    fn default() -> Self {
        Self {
            validation: cfg!(debug_assertions),
            label: None,
            bindless: BindlessCapacity::default(),
        }
    }
}

/// The RHI device -- central object for resource creation.
pub struct Device {
    pub(crate) inner: Rc<DeviceInner>,
}

backend_enum!(DeviceInner { vulkan: Box<crate::backend::vulkan::device::VulkanDevice>, metal: Box<crate::backend::metal::device::MetalDevice> });

/// Keeps `DeviceInner` alive so a `Device` handle can be dropped while its resources still are.
/// Never read back; `Option` only because a backend cannot hand out an `Rc` to itself while
/// constructing, and `Device::create` stamps it before the resource escapes.
trait DeviceOwned {
    fn owner_mut(&mut self) -> &mut Option<Rc<DeviceInner>>;
}

macro_rules! impl_device_owned {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl DeviceOwned for $ty {
                fn owner_mut(&mut self) -> &mut Option<Rc<DeviceInner>> {
                    &mut self._owner
                }
            }
        )+
    };
}

impl_device_owned!(
    AccelerationStructure,
    CommandBuffer,
    ComputePso,
    Allocation,
    GraphicsPso,
    MeshletPso,
    QueryPool,
    Sampler,
    ShaderModule,
    Surface,
    Swapchain,
    Texture,
    TimelineSemaphore,
);

use crate::sealed;

/// A resource the RHI reclaims only on [`Device::destroy`]: dropping one leaks its memory, heap
/// slot, or bindless ID. Everything else frees on `Drop`.
///
/// These are the resources a submitted command buffer may still be reading, so release is
/// deferred to a fence — which `Drop` cannot see and [`Device::destroy`] can.
pub trait DeviceResource: sealed::Sealed + Sized {
    #[doc(hidden)]
    fn destroy_on(self, device: &Device);
}

/// `destroy_on` is one dispatch for every resource but `Texture`, which frees its views first.
macro_rules! impl_device_resource {
    ($($ty:ty => $destroy:ident),+ $(,)?) => {
        $(
            impl DeviceResource for $ty {
                fn destroy_on(self, device: &Device) {
                    device.inner.$destroy(self.inner)
                }
            }
        )+
    };
}

impl_device_resource!(
    Allocation => destroy_allocation,
    QueryPool => destroy_query_pool,
    AccelerationStructure => destroy_accel,
);

impl DeviceResource for Sampler {
    /// Carries a bindless id rather than a backend object, so the id is what is retired.
    fn destroy_on(self, device: &Device) {
        device.inner.destroy_sampler(self)
    }
}

impl DeviceResource for Texture {
    /// Views hold bindless slots of their own, so they go first.
    fn destroy_on(mut self, device: &Device) {
        for id in self.views.drain(..) {
            device.inner.destroy_texture_view(id);
        }
        device.inner.destroy_texture(self)
    }
}

/// Settle on one threadgroup size for a compute PSO; the module's and the descriptor's must agree
/// when both are present. `None` when neither states one, which only Metal rejects — Vulkan reads
/// the size out of the SPIR-V.
fn resolve_threadgroup_size(
    desc: &ComputePsoDesc,
    module: &ShaderModule,
) -> RhiResult<Option<[u32; 3]>> {
    let threads = match (desc.threads_per_threadgroup, module.threads_per_threadgroup) {
        (Some(from_desc), Some(from_module)) if from_desc != from_module => {
            return Err(RhiError::PipelineCreation(
                format!(
                    "ComputePsoDesc asks for {from_desc:?} threads per threadgroup but the \
                     shader declares [numthreads{from_module:?}]"
                )
                .into(),
            ));
        }
        (Some(threads), _) | (None, Some(threads)) => threads,
        (None, None) => return Ok(None),
    };
    if threads.contains(&0) {
        return Err(RhiError::PipelineCreation(
            format!("threads per threadgroup {threads:?} has a zero dimension").into(),
        ));
    }
    Ok(Some(threads))
}

/// Reject a shader module handed to the wrong `create_*_pso` argument.
impl Device {
    /// Opens the compiled-in backend.
    pub fn new(desc: &DeviceDesc) -> RhiResult<Self> {
        #[cfg(feature = "vulkan")]
        let inner = crate::backend::vulkan::device::VulkanDevice::new(desc)?;
        #[cfg(feature = "metal")]
        let inner = crate::backend::metal::device::MetalDevice::new(desc)?;
        Ok(Self::from_inner(Box::new(inner)))
    }

    fn from_inner(inner: DeviceInner) -> Self {
        Self {
            inner: Rc::new(inner),
        }
    }

    fn own<T: DeviceOwned>(&self, mut resource: T) -> T {
        *resource.owner_mut() = Some(Rc::clone(&self.inner));
        resource
    }

    /// Dispatch a backend `create_*` and stamp the owning device, so the second half cannot be
    /// forgotten.
    fn create<T: DeviceOwned>(
        &self,
        create: impl FnOnce(&DeviceInner) -> RhiResult<T>,
    ) -> RhiResult<T> {
        create(self.inner.as_ref()).map(|resource| self.own(resource))
    }

    /// The backend this build is compiled against.
    pub fn backend(&self) -> Backend {
        #[cfg(feature = "vulkan")]
        return Backend::Vulkan;
        #[cfg(feature = "metal")]
        return Backend::Metal;
    }

    pub fn create_surface(&self, desc: &SurfaceDesc) -> RhiResult<Surface> {
        self.create(|inner| inner.create_surface(desc))
    }

    pub fn create_swapchain(
        &self,
        surface: &Surface,
        desc: &SwapchainDesc,
    ) -> RhiResult<Swapchain> {
        self.create(|inner| inner.create_swapchain(&surface.inner, desc))
    }

    /// On resize.
    pub fn recreate_swapchain(
        &self,
        swapchain: &mut Swapchain,
        desc: &SwapchainDesc,
    ) -> RhiResult<()> {
        self.inner.recreate_swapchain(swapchain, desc)
    }

    /// The canonical allocation path; [`allocate`](Self::allocate) and
    /// [`allocate_aligned`](Self::allocate_aligned) are shorthands over it.
    ///
    /// `desc.align` reaches the backend's suballocator, which already carves ranges at a
    /// requested alignment. Nothing is over-allocated to make room for a shift.
    pub fn create_allocation(&self, desc: &AllocationDesc) -> RhiResult<Allocation> {
        if !desc.align.is_power_of_two() {
            return Err(RhiError::AllocationFailed(
                format!(
                    "allocation alignment {} is not a non-zero power of two",
                    desc.align
                )
                .into(),
            ));
        }
        let allocation = self
            .inner
            .create_allocation(desc)
            .map(|resource| self.own(resource))?;
        debug_assert!(
            allocation.gpu().is_aligned_to(desc.align),
            "backend returned {:#x} for a {}-byte alignment",
            allocation.gpu().addr(),
            desc.align
        );
        Ok(allocation)
    }

    /// Allocate with [`DEFAULT_ALIGN`](crate::memory::DEFAULT_ALIGN) alignment.
    pub fn allocate(&self, size: u64, memory: MemoryType) -> RhiResult<Allocation> {
        self.allocate_aligned(size, crate::memory::DEFAULT_ALIGN, memory)
    }

    pub fn allocate_aligned(
        &self,
        size: u64,
        align: u64,
        memory: MemoryType,
    ) -> RhiResult<Allocation> {
        self.create_allocation(&AllocationDesc {
            size,
            align,
            memory,
            label: None,
        })
    }

    /// Allocate mapped memory and upload `data`. Use `std::slice::from_ref` for a single value.
    pub fn upload_slice<T: GpuPod>(&self, data: &[T]) -> RhiResult<Allocation> {
        let size = std::mem::size_of_val(data).max(1) as u64;
        let mut alloc = self.allocate(size, MemoryType::Upload)?;
        if !data.is_empty() {
            alloc.upload_slice(data)?;
        }
        Ok(alloc)
    }

    /// Translate a CPU-mapped pointer to a GPU virtual address, if possible.
    pub fn host_to_device_pointer(&self, cpu_ptr: *const u8) -> Option<GpuPtr<u8>> {
        self.inner.host_to_device_pointer(cpu_ptr)
    }

    /// Query the size/alignment required for `create_texture`.
    pub fn texture_size_align(&self, desc: &TextureDesc) -> RhiResult<TextureSizeAlign> {
        desc.validate()?;
        self.inner.texture_size_align(desc)
    }

    /// Create a texture in caller-owned GPU memory.
    pub fn create_texture(
        &self,
        desc: &TextureDesc,
        texture_gpu: GpuPtr<u8>,
    ) -> RhiResult<Texture> {
        desc.validate()?;
        self.create(|inner| inner.create_texture(desc, texture_gpu))
    }

    /// Subresource view of `texture`, released with it.
    ///
    /// The source must carry `kind`'s usage and the mip/layer range must sit inside the texture;
    /// a format change also needs [`TextureUsage`](crate::TextureUsage)`::FORMAT_VIEW` and a
    /// compatible layout. The backend's validation reports violations, not this.
    pub fn create_texture_view(
        &self,
        texture: &mut Texture,
        kind: ViewKind,
        desc: &TextureViewDesc,
    ) -> RhiResult<TextureHandle> {
        let inner = self.inner.as_ref();
        let id = inner.create_texture_view(texture, desc, kind)?;
        texture.views.push(id);
        Ok(TextureHandle::from_raw(inner.texture_handle_raw(id)))
    }

    pub fn create_sampler(&self, desc: &SamplerDesc) -> RhiResult<Sampler> {
        self.create(|inner| inner.create_sampler(desc))
    }

    pub fn create_shader_module(&self, desc: &ShaderModuleDesc) -> RhiResult<ShaderModule> {
        self.create(|inner| inner.create_shader_module(desc))
    }

    pub fn create_graphics_pso(
        &self,
        desc: &GraphicsPsoDesc,
        vertex: &ShaderModule,
        pixel: &ShaderModule,
    ) -> RhiResult<GraphicsPso> {
        {
            let d = self.inner.as_ref();
            let v = &vertex.inner;
            let p = &pixel.inner;
            d.create_graphics_pso(desc, v, p)
        }
        .map(|resource| self.own(resource))
    }

    pub fn create_compute_pso(
        &self,
        desc: &ComputePsoDesc,
        compute: &ShaderModule,
    ) -> RhiResult<ComputePso> {
        let threads = resolve_threadgroup_size(desc, compute)?;
        {
            let d = self.inner.as_ref();
            let c = &compute.inner;
            d.create_compute_pso(desc, c, threads)
        }
        .map(|resource| self.own(resource))
    }

    pub fn create_meshlet_pso(
        &self,
        desc: &MeshletPsoDesc,
        mesh: &ShaderModule,
        pixel: &ShaderModule,
    ) -> RhiResult<MeshletPso> {
        {
            let d = self.inner.as_ref();
            let m = &mesh.inner;
            let p = &pixel.inner;
            d.create_meshlet_pso(desc, m, p)
        }
        .map(|resource| self.own(resource))
    }

    /// Allocates only; build it with `cmd.build_blas` before referencing it from a TLAS.
    pub fn create_blas(&self, desc: &BlasDesc<'_>) -> RhiResult<AccelerationStructure> {
        self.create(|inner| inner.create_blas(desc))
    }

    /// Allocates only; build it with `cmd.build_tlas`.
    pub fn create_tlas(&self, desc: &TlasDesc) -> RhiResult<AccelerationStructure> {
        self.create(|inner| inner.create_tlas(desc))
    }

    /// Size in bytes of one native TLAS instance descriptor for this backend. The instance
    /// buffer passed to `build_tlas` must use this stride; fill entries with
    /// [`write_tlas_instance`](Self::write_tlas_instance).
    pub fn tlas_instance_stride(&self) -> usize {
        self.inner.tlas_instance_stride()
    }

    /// Encode `instance` into slot `index` of a CPU-mapped instance buffer, using the active
    /// backend's native instance layout (Vulkan `VkAccelerationStructureInstanceKHR`; Metal
    /// indirect descriptor). Size the buffer as `instance_count * tlas_instance_stride()`.
    pub fn write_tlas_instance(
        &self,
        dst: &mut Allocation,
        index: usize,
        instance: &TlasInstance,
    ) -> RhiResult<()> {
        let stride = self.tlas_instance_stride() as u64;
        let base = dst.mapped::<u8>().ok_or_else(|| {
            RhiError::AllocationFailed("instance buffer is not CPU-mapped".into())
        })?;
        let slot = base.byte_offset(stride.saturating_mul(index as u64));
        if slot.byte_len() < stride {
            return Err(RhiError::AllocationFailed(
                format!("TLAS instance {index} (stride {stride}) exceeds the instance buffer")
                    .into(),
            ));
        }
        // SAFETY: `slot` covers at least `stride` bytes of a live mapping, checked just above,
        // and `&mut self` on `dst` rules out any other reference to them.
        let bytes = unsafe { std::slice::from_raw_parts_mut(slot.cpu(), stride as usize) };
        self.inner.write_tlas_instance(bytes, instance);
        Ok(())
    }

    pub fn create_command_buffer(&self) -> RhiResult<CommandBuffer> {
        self.create(|inner| inner.create_command_buffer())
    }

    /// Pre-wired with the swapchain's image views, for the main render loop.
    pub fn create_command_buffer_for_swapchain(
        &self,
        swapchain: &Swapchain,
        frame_index: usize,
    ) -> RhiResult<CommandBuffer> {
        self.create(|inner| {
            inner.create_command_buffer_for_swapchain(&swapchain.inner, frame_index)
        })
    }

    pub fn queue(&self) -> &Queue {
        self.inner.queue()
    }

    pub fn create_timeline_semaphore(&self, initial_value: u64) -> RhiResult<TimelineSemaphore> {
        self.create(|inner| inner.create_timeline_semaphore(initial_value))
    }

    pub fn create_query_pool(&self, count: u32) -> RhiResult<QueryPool> {
        self.create(|inner| inner.create_query_pool(count))
    }

    pub fn timestamp_period_ns(&self) -> f64 {
        self.inner.timestamp_period_ns()
    }

    /// Valid once the writing GPU work has completed.
    pub fn read_timestamps(&self, pool: &QueryPool) -> RhiResult<Vec<u64>> {
        let mut out = vec![0u64; pool.count as usize];
        self.read_timestamps_into(pool, &mut out)?;
        Ok(out)
    }

    /// `None` if either sample is unwritten.
    ///
    /// Resolves into a caller-owned buffer rather than allocating: this runs once a frame on a
    /// pool that is usually two slots wide.
    pub fn gpu_elapsed_ms(&self, pool: &QueryPool, begin: u32, end: u32) -> RhiResult<Option<f64>> {
        if let Some(bad) = [begin, end].iter().find(|&&i| i >= pool.count()) {
            return Err(RhiError::Backend(
                format!(
                    "gpu_elapsed_ms: query index {bad} out of range for a {}-slot pool",
                    pool.count()
                )
                .into(),
            ));
        }
        // Such pools are a few slots wide; resolve onto the stack.
        let mut ticks = [0u64; 16];
        let (b, e) = if pool.count() as usize <= ticks.len() {
            self.read_timestamps_into(pool, &mut ticks)?;
            (ticks[begin as usize], ticks[end as usize])
        } else {
            let all = self.read_timestamps(pool)?;
            (all[begin as usize], all[end as usize])
        };
        if b == 0 || e == 0 || e <= b {
            return Ok(None);
        }
        Ok(Some((e - b) as f64 * self.timestamp_period_ns() / 1.0e6))
    }

    /// Resolve the whole pool into `out` (at least `pool.count()` long), without allocating.
    pub fn read_timestamps_into(&self, pool: &QueryPool, out: &mut [u64]) -> RhiResult<()> {
        if out.len() < pool.count() as usize {
            return Err(RhiError::Backend(
                format!(
                    "read_timestamps_into needs room for {} slots, got {}",
                    pool.count(),
                    out.len()
                )
                .into(),
            ));
        }
        self.inner
            .read_timestamps_into(&pool.inner, pool.count, out)
    }

    pub fn wait_idle(&self) {
        self.inner.wait_idle()
    }

    /// Release a resource. Safe to call at any point, including mid-frame.
    ///
    /// The handle is consumed and stops resolving right away, but the storage and any bindless
    /// slot are held until every submission issued so far has retired, then reclaimed by the next
    /// [`Queue::submit`](crate::Queue::submit), [`acquire_image`](crate::Queue::acquire_image) or
    /// [`wait_idle`](Self::wait_idle). No fence of your own is required. Panics on a foreign
    /// device.
    pub fn destroy<R: DeviceResource>(&self, resource: R) {
        resource.destroy_on(self);
    }

    /// Blocks until that frame slot's prior work has retired.
    pub fn wait_for_frame(&self, frame_index: usize) {
        self.inner.wait_for_frame(frame_index)
    }
}
