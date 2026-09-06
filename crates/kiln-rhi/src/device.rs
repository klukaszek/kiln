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
use crate::shader::{ShaderModule, ShaderModuleDesc, ShaderModuleInner};
use crate::surface::{Surface, SurfaceDesc};
use crate::swapchain::{Swapchain, SwapchainDesc};
use crate::sync::TimelineSemaphore;
use crate::texture::{Texture, TextureDesc, TextureSizeAlign};
use crate::types::{BlasDesc, GpuPtr, TlasDesc, TlasInstance};
use std::rc::Rc;

/// Which GPU backend to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Vulkan 1.3+.
    Vulkan,
    /// Metal 4 (Apple platforms only).
    Metal,
}

/// Bindless implementation mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindlessMode {
    /// GPU-addressable descriptor heap (Vulkan descriptor buffer extension).
    DescriptorBuffer,
    /// Metal 4 argument tables (`MTL4ArgumentTable`).
    ArgumentTable,
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
pub struct DeviceDesc {
    /// Enable Vulkan validation. Metal validation is configured at process launch through
    /// Xcode or `MTL_DEBUG_LAYER=1`; this flag cannot toggle Metal's process-wide layer.
    pub validation: bool,
    pub label: Option<String>,
    /// Preferred backend. `None` uses the default for the platform.
    pub preferred_backend: Option<Backend>,
}

impl Default for DeviceDesc {
    fn default() -> Self {
        Self {
            validation: cfg!(debug_assertions),
            label: None,
            preferred_backend: None,
        }
    }
}

/// The RHI device -- central object for resource creation.
/// Uses enum dispatch for zero-cost backend selection.
pub struct Device {
    pub(crate) inner: Rc<DeviceInner>,
}

pub(crate) enum DeviceInner {
    #[cfg(feature = "vulkan")]
    Vulkan(Box<crate::backend::vulkan::device::VulkanDevice>),
    #[cfg(feature = "metal")]
    Metal(Box<crate::backend::metal::device::MetalDevice>),
}

/// Resources hold an `Rc` to the device that made them, so a `Device` handle can be dropped while
/// its resources are still alive. Nothing reads it back; it exists to keep `DeviceInner` alive.
/// `Option` only because a backend cannot hand out an `Rc` to the device that owns it during
/// construction; every public `create_*` stamps it before the resource escapes.
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

/// A resource whose storage the RHI reclaims only on [`Device::destroy`]: dropping one leaks its
/// memory, heap slot, or bindless ID. Everything else in the RHI frees on `Drop` like normal Rust.
pub trait DeviceResource: sealed::Sealed + Sized {
    #[doc(hidden)]
    fn destroy_on(self, device: &Device);
}

impl DeviceResource for Allocation {
    fn destroy_on(self, device: &Device) {
        backend_dispatch!(device.inner.as_ref(), DeviceInner, d => d.destroy_allocation(self))
    }
}

impl DeviceResource for Texture {
    fn destroy_on(mut self, device: &Device) {
        for id in self.views.drain(..) {
            backend_dispatch!(device.inner.as_ref(), DeviceInner, d => d.destroy_texture_view(id));
        }
        backend_dispatch!(device.inner.as_ref(), DeviceInner, d => d.destroy_texture(self))
    }
}

impl DeviceResource for Sampler {
    fn destroy_on(self, device: &Device) {
        backend_dispatch!(device.inner.as_ref(), DeviceInner, d => d.destroy_sampler(self))
    }
}

impl DeviceResource for QueryPool {
    fn destroy_on(self, device: &Device) {
        backend_dispatch!(device.inner.as_ref(), DeviceInner, d => d.destroy_query_pool(self))
    }
}

impl Device {
    /// Selects `desc.preferred_backend`, else Vulkan if compiled in, else Metal.
    pub fn new(desc: &DeviceDesc) -> RhiResult<Self> {
        let backend = desc.preferred_backend.unwrap_or_else(Self::default_backend);

        match backend {
            #[cfg(feature = "vulkan")]
            Backend::Vulkan => {
                let vk_device = crate::backend::vulkan::device::VulkanDevice::new(desc)?;
                Ok(Self::from_inner(DeviceInner::Vulkan(Box::new(vk_device))))
            }
            #[cfg(feature = "metal")]
            Backend::Metal => {
                let mtl_device = crate::backend::metal::device::MetalDevice::new(desc)?;
                Ok(Self::from_inner(DeviceInner::Metal(Box::new(mtl_device))))
            }
            #[allow(unreachable_patterns)]
            _ => Err(crate::error::RhiError::Unsupported(format!(
                "Backend '{backend}' is not compiled in. Enable the corresponding feature."
            ))),
        }
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

    /// The default backend for this build.
    fn default_backend() -> Backend {
        #[cfg(feature = "vulkan")]
        {
            Backend::Vulkan
        }
        #[cfg(all(feature = "metal", not(feature = "vulkan")))]
        {
            Backend::Metal
        }
        #[cfg(not(any(feature = "vulkan", feature = "metal")))]
        {
            compile_error!("At least one backend feature (vulkan or metal) must be enabled");
        }
    }

    pub fn backend(&self) -> Backend {
        match self.inner.as_ref() {
            #[cfg(feature = "vulkan")]
            DeviceInner::Vulkan(_) => Backend::Vulkan,
            #[cfg(feature = "metal")]
            DeviceInner::Metal(_) => Backend::Metal,
        }
    }

    /// Determined by the backend.
    pub fn bindless_mode(&self) -> BindlessMode {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.bindless_mode())
    }

    pub fn create_surface(&self, desc: &SurfaceDesc) -> RhiResult<Surface> {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.create_surface(desc))
            .map(|resource| self.own(resource))
    }

    pub fn create_swapchain(
        &self,
        surface: &Surface,
        desc: &SwapchainDesc,
    ) -> RhiResult<Swapchain> {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.create_swapchain(surface, desc))
            .map(|resource| self.own(resource))
    }

    /// On resize.
    pub fn recreate_swapchain(
        &self,
        swapchain: &mut Swapchain,
        desc: &SwapchainDesc,
    ) -> RhiResult<()> {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.recreate_swapchain(swapchain, desc))
    }

    /// The canonical allocation path; [`allocate`](Self::allocate) and
    /// [`allocate_aligned`](Self::allocate_aligned) are shorthands over it.
    pub fn create_allocation(&self, desc: &AllocationDesc) -> RhiResult<Allocation> {
        let align = desc.align;
        if !align.is_power_of_two() {
            return Err(RhiError::AllocationFailed(format!(
                "allocation alignment {align} is not a non-zero power of two"
            )));
        }

        // Over-allocate so an aligned address exists inside, then point the handle at it.
        // `size` still reports what was asked for, keeping bounds checks in usable terms.
        let backing = AllocationDesc {
            size: aligned_backing_size(desc.size, align)?,
            ..desc.clone()
        };
        let mut allocation =
            backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.create_allocation(&backing))
                .map(|resource| self.own(resource))?;
        allocation.offset = (align - allocation.gpu().address % align) % align;
        allocation.size = desc.size;
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
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.host_to_device_pointer(cpu_ptr))
    }

    /// Query the size/alignment required for `create_texture`.
    pub fn texture_size_align(&self, desc: &TextureDesc) -> RhiResult<TextureSizeAlign> {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.texture_size_align(desc))
    }

    /// Create a texture in caller-owned GPU memory.
    pub fn create_texture(
        &self,
        desc: &TextureDesc,
        texture_gpu: GpuPtr<u8>,
    ) -> RhiResult<Texture> {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.create_texture(desc, texture_gpu))
            .map(|resource| self.own(resource))
    }

    pub fn create_sampler(&self, desc: &SamplerDesc) -> RhiResult<Sampler> {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.create_sampler(desc))
            .map(|resource| self.own(resource))
    }

    pub fn create_shader_module(&self, desc: &ShaderModuleDesc) -> RhiResult<ShaderModule> {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.create_shader_module(desc))
            .map(|resource| self.own(resource))
    }

    pub fn create_graphics_pso(
        &self,
        desc: &GraphicsPsoDesc,
        vertex: &ShaderModule,
        pixel: &ShaderModule,
    ) -> RhiResult<GraphicsPso> {
        let result = match (self.inner.as_ref(), &vertex.inner, &pixel.inner) {
            #[cfg(feature = "vulkan")]
            (
                DeviceInner::Vulkan(d),
                ShaderModuleInner::Vulkan(v),
                ShaderModuleInner::Vulkan(p),
            ) => d.create_graphics_pso(desc, v, p),
            #[cfg(feature = "metal")]
            (DeviceInner::Metal(d), ShaderModuleInner::Metal(v), ShaderModuleInner::Metal(p)) => {
                d.create_graphics_pso(desc, v, p)
            }
            #[allow(unreachable_patterns)]
            _ => unreachable!("shader module backend does not match device backend"),
        };
        result.map(|resource| self.own(resource))
    }

    pub fn create_compute_pso(
        &self,
        desc: &ComputePsoDesc,
        compute: &ShaderModule,
    ) -> RhiResult<ComputePso> {
        let result = match (self.inner.as_ref(), &compute.inner) {
            #[cfg(feature = "vulkan")]
            (DeviceInner::Vulkan(d), ShaderModuleInner::Vulkan(c)) => d.create_compute_pso(desc, c),
            #[cfg(feature = "metal")]
            (DeviceInner::Metal(d), ShaderModuleInner::Metal(c)) => d.create_compute_pso(desc, c),
            #[allow(unreachable_patterns)]
            _ => unreachable!("shader module backend does not match device backend"),
        };
        result.map(|resource| self.own(resource))
    }

    pub fn create_meshlet_pso(
        &self,
        desc: &MeshletPsoDesc,
        mesh: &ShaderModule,
        pixel: &ShaderModule,
    ) -> RhiResult<MeshletPso> {
        let result = match (self.inner.as_ref(), &mesh.inner, &pixel.inner) {
            #[cfg(feature = "vulkan")]
            (
                DeviceInner::Vulkan(d),
                ShaderModuleInner::Vulkan(m),
                ShaderModuleInner::Vulkan(p),
            ) => d.create_meshlet_pso(desc, m, p),
            #[cfg(feature = "metal")]
            (DeviceInner::Metal(d), ShaderModuleInner::Metal(m), ShaderModuleInner::Metal(p)) => {
                d.create_meshlet_pso(desc, m, p)
            }
            #[allow(unreachable_patterns)]
            _ => unreachable!("shader module backend does not match device backend"),
        };
        result.map(|resource| self.own(resource))
    }

    /// Allocates only; build it with `cmd.build_blas` before referencing it from a TLAS.
    pub fn create_blas(&self, desc: &BlasDesc) -> RhiResult<AccelerationStructure> {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.create_blas(desc))
            .map(|resource| self.own(resource))
    }

    /// Allocates only; build it with `cmd.build_tlas`.
    pub fn create_tlas(&self, desc: &TlasDesc) -> RhiResult<AccelerationStructure> {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.create_tlas(desc))
            .map(|resource| self.own(resource))
    }

    /// Size in bytes of one native TLAS instance descriptor for this backend. The instance
    /// buffer passed to `build_tlas` must use this stride; fill entries with
    /// [`write_tlas_instance`](Self::write_tlas_instance).
    pub fn tlas_instance_stride(&self) -> usize {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.tlas_instance_stride())
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
            return Err(RhiError::AllocationFailed(format!(
                "TLAS instance {index} (stride {stride}) exceeds the instance buffer"
            )));
        }
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.write_tlas_instance(slot.cpu(), instance));
        Ok(())
    }

    pub fn create_command_buffer(&self) -> RhiResult<CommandBuffer> {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.create_command_buffer())
            .map(|resource| self.own(resource))
    }

    /// Pre-wired with the swapchain's image views, for the main render loop.
    pub fn create_command_buffer_for_swapchain(
        &self,
        swapchain: &Swapchain,
        frame_index: usize,
    ) -> RhiResult<CommandBuffer> {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.create_command_buffer_for_swapchain(swapchain, frame_index))
            .map(|resource| self.own(resource))
    }

    pub fn queue(&self) -> &Queue {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.queue())
    }

    pub fn create_timeline_semaphore(&self, initial_value: u64) -> RhiResult<TimelineSemaphore> {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.create_timeline_semaphore(initial_value))
            .map(|resource| self.own(resource))
    }

    pub fn create_query_pool(&self, count: u32) -> RhiResult<QueryPool> {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.create_query_pool(count))
            .map(|resource| self.own(resource))
    }

    pub fn timestamp_period_ns(&self) -> f64 {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.timestamp_period_ns())
    }

    /// Valid once the writing GPU work has completed.
    pub fn read_timestamps(&self, pool: &QueryPool) -> RhiResult<Vec<u64>> {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.read_timestamps(pool))
    }

    /// `None` if either sample is unwritten.
    pub fn gpu_elapsed_ms(&self, pool: &QueryPool, begin: u32, end: u32) -> RhiResult<Option<f64>> {
        let ticks = self.read_timestamps(pool)?;
        let (Some(&b), Some(&e)) = (ticks.get(begin as usize), ticks.get(end as usize)) else {
            return Err(RhiError::Backend(format!(
                "gpu_elapsed_ms: query index ({begin}, {end}) out of range for a {}-slot pool",
                ticks.len()
            )));
        };
        if b == 0 || e == 0 || e <= b {
            return Ok(None);
        }
        Ok(Some((e - b) as f64 * self.timestamp_period_ns() / 1.0e6))
    }

    pub fn wait_idle(&self) {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.wait_idle())
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
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.wait_for_frame(frame_index))
    }
}

fn aligned_backing_size(size: u64, align: u64) -> RhiResult<u64> {
    size.max(1).checked_add(align - 1).ok_or_else(|| {
        RhiError::AllocationFailed(format!(
            "allocation size {size} with alignment {align} overflows u64"
        ))
    })
}
