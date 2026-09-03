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
use crate::texture::{Texture, TextureDesc, TextureSizeAlign, TextureViewDesc};
use crate::types::{BlasDesc, ClipSpaceY, GpuPtr, TlasDesc, TlasInstance};
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
            Backend::Vulkan => write!(f, "Vulkan"),
            Backend::Metal => write!(f, "Metal"),
        }
    }
}

/// Description for creating a device.
pub struct DeviceDesc {
    /// Enable validation/debug layers.
    pub validation: bool,
    pub label: Option<String>,
    /// Preferred backend. `None` uses the default for the platform.
    pub preferred_backend: Option<Backend>,
    /// Preferred bindless mode. `None` lets the backend choose the best available mode.
    /// Vulkan requires descriptor buffers and mutable image descriptors; device creation fails if
    /// either is unavailable.
    pub bindless_mode: Option<BindlessMode>,
}

impl Default for DeviceDesc {
    fn default() -> Self {
        Self {
            validation: cfg!(debug_assertions),
            label: None,
            preferred_backend: None,
            bindless_mode: None,
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

trait DeviceOwned {
    fn owner(&self) -> &Option<Rc<DeviceInner>>;
    fn owner_mut(&mut self) -> &mut Option<Rc<DeviceInner>>;
}

macro_rules! impl_device_owned {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl DeviceOwned for $ty {
                fn owner(&self) -> &Option<Rc<DeviceInner>> {
                    &self._owner
                }

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

impl Device {
    /// Create a new device, selecting the backend based on `desc.preferred_backend`.
    ///
    /// If no preference is given, defaults to Vulkan (if available) then Metal.
    pub fn new(desc: &DeviceDesc) -> RhiResult<Self> {
        let backend = desc.preferred_backend.unwrap_or(Self::default_backend());

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
                "Backend '{}' is not compiled in. Enable the corresponding feature.",
                backend
            ))),
        }
    }

    fn from_inner(inner: DeviceInner) -> Self {
        let mut inner = Rc::new(inner);
        let device_id = Rc::as_ptr(&inner) as usize;
        match Rc::get_mut(&mut inner).expect("new device Rc unexpectedly shared") {
            #[cfg(feature = "vulkan")]
            DeviceInner::Vulkan(device) => device.set_device_id(device_id),
            #[cfg(feature = "metal")]
            DeviceInner::Metal(device) => device.set_device_id(device_id),
        }
        Self { inner }
    }

    fn own<T: DeviceOwned>(&self, mut resource: T) -> T {
        *resource.owner_mut() = Some(Rc::clone(&self.inner));
        resource
    }

    fn ensure_owns<T: DeviceOwned>(&self, resource: &T, kind: &str) -> RhiResult<()> {
        if resource
            .owner()
            .as_ref()
            .is_some_and(|owner| Rc::ptr_eq(owner, &self.inner))
        {
            Ok(())
        } else {
            Err(RhiError::Backend(format!(
                "{kind} belongs to a different device"
            )))
        }
    }

    fn assert_owns<T: DeviceOwned>(&self, resource: &T, kind: &str) {
        assert!(
            self.ensure_owns(resource, kind).is_ok(),
            "{kind} belongs to a different device"
        );
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

    /// The name of the active backend (e.g. "Vulkan", "Metal").
    pub fn backend_name(&self) -> &'static str {
        match self.inner.as_ref() {
            #[cfg(feature = "vulkan")]
            DeviceInner::Vulkan(_) => "Vulkan",
            #[cfg(feature = "metal")]
            DeviceInner::Metal(_) => "Metal",
        }
    }

    /// The active bindless mode selected by the backend.
    pub fn bindless_mode(&self) -> BindlessMode {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.bindless_mode())
    }

    /// Clip-space Y convention. The RHI normalizes both backends to Y-up.
    pub fn clip_space_y(&self) -> ClipSpaceY {
        ClipSpaceY::Up
    }

    /// Create a presentation surface from raw window handles.
    pub fn create_surface(&self, desc: &SurfaceDesc) -> RhiResult<Surface> {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.create_surface(desc))
            .map(|resource| self.own(resource))
    }

    /// Create a swapchain for the given surface.
    pub fn create_swapchain(
        &self,
        surface: &Surface,
        desc: &SwapchainDesc,
    ) -> RhiResult<Swapchain> {
        self.ensure_owns(surface, "surface")?;
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.create_swapchain(surface, desc))
            .map(|resource| self.own(resource))
    }

    /// Recreate swapchain (on resize).
    pub fn recreate_swapchain(
        &self,
        swapchain: &mut Swapchain,
        desc: &SwapchainDesc,
    ) -> RhiResult<()> {
        self.ensure_owns(swapchain, "swapchain")?;
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.recreate_swapchain(swapchain, desc))
    }

    /// Create an allocation with an explicit description.
    pub fn create_allocation(&self, desc: &AllocationDesc) -> RhiResult<Allocation> {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.create_allocation(desc))
            .map(|resource| self.own(resource))
    }

    /// Allocate GPU memory with the default 16-byte address alignment.
    pub fn allocate(&self, size: u64, memory: MemoryType) -> RhiResult<Allocation> {
        self.allocate_aligned(size, 16, memory)
    }

    /// Allocate GPU memory with explicit alignment.
    pub fn allocate_aligned(
        &self,
        size: u64,
        align: u64,
        memory: MemoryType,
    ) -> RhiResult<Allocation> {
        if !align.is_power_of_two() {
            return Err(RhiError::AllocationFailed(format!(
                "allocation alignment {align} is not a non-zero power of two"
            )));
        }

        let backing_size = aligned_backing_size(size, align)?;
        let mut allocation = self.create_allocation(&AllocationDesc {
            size: backing_size,
            memory,
            label: None,
        })?;
        allocation.offset = (align - allocation.gpu().address % align) % align;
        allocation.size = size;
        Ok(allocation)
    }

    /// Allocate space for `count` values with their natural alignment.
    pub fn allocate_array<T>(&self, count: usize, memory: MemoryType) -> RhiResult<Allocation> {
        let size = std::mem::size_of::<T>()
            .checked_mul(count)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| RhiError::AllocationFailed("typed allocation size overflow".into()))?;
        self.allocate_aligned(size.max(1), std::mem::align_of::<T>() as u64, memory)
    }

    /// Allocate mapped memory and upload one value.
    pub fn upload<T: GpuPod>(&self, value: &T) -> RhiResult<Allocation> {
        let mut allocation = self.allocate_array::<T>(1, MemoryType::Default)?;
        allocation.upload(value)?;
        Ok(allocation)
    }

    /// Allocate mapped memory and upload `data`.
    pub fn upload_slice<T: GpuPod>(&self, data: &[T]) -> RhiResult<Allocation> {
        let size = std::mem::size_of_val(data).max(1) as u64;
        let mut alloc = self.allocate(size, MemoryType::Default)?;
        if !data.is_empty() {
            alloc.upload_slice(data)?;
        }
        Ok(alloc)
    }

    /// Release an allocation. The caller guarantees that submitted GPU work no longer uses it.
    pub fn free(&self, allocation: Allocation) {
        self.destroy_allocation(allocation);
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

    /// Create a sampler.
    pub fn create_sampler(&self, desc: &SamplerDesc) -> RhiResult<Sampler> {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.create_sampler(desc))
            .map(|resource| self.own(resource))
    }

    /// Create a shader module.
    pub fn create_shader_module(&self, desc: &ShaderModuleDesc) -> RhiResult<ShaderModule> {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.create_shader_module(desc))
            .map(|resource| self.own(resource))
    }

    /// Create a graphics pipeline.
    pub fn create_graphics_pso(
        &self,
        desc: &GraphicsPsoDesc,
        vertex: &ShaderModule,
        pixel: &ShaderModule,
    ) -> RhiResult<GraphicsPso> {
        self.ensure_owns(vertex, "vertex shader")?;
        self.ensure_owns(pixel, "pixel shader")?;
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

    /// Create a compute pipeline.
    pub fn create_compute_pso(
        &self,
        desc: &ComputePsoDesc,
        compute: &ShaderModule,
    ) -> RhiResult<ComputePso> {
        self.ensure_owns(compute, "compute shader")?;
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

    /// Create a mesh-shader graphics pipeline.
    pub fn create_meshlet_pso(
        &self,
        desc: &MeshletPsoDesc,
        mesh: &ShaderModule,
        pixel: &ShaderModule,
    ) -> RhiResult<MeshletPso> {
        self.ensure_owns(mesh, "mesh shader")?;
        self.ensure_owns(pixel, "pixel shader")?;
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

    /// Allocate a Bottom-Level Acceleration Structure.
    ///
    /// The returned `AccelerationStructure` must be built via `cmd.build_blas(as, desc)`
    /// before it can be referenced in a TLAS instance.
    pub fn create_blas(&self, desc: &BlasDesc) -> RhiResult<AccelerationStructure> {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.create_blas(desc))
            .map(|resource| self.own(resource))
    }

    /// Allocate a Top-Level Acceleration Structure.
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
        dst: &Allocation,
        index: usize,
        instance: &TlasInstance,
    ) -> RhiResult<()> {
        self.ensure_owns(dst, "TLAS instance buffer")?;
        let stride = self.tlas_instance_stride();
        let offset = index.checked_mul(stride).ok_or_else(|| {
            RhiError::AllocationFailed(format!(
                "TLAS instance index {index} overflows the host address space"
            ))
        })?;
        let end = offset.checked_add(stride).ok_or_else(|| {
            RhiError::AllocationFailed(format!(
                "TLAS instance {index} end overflows the host address space"
            ))
        })?;
        let dst_size = usize::try_from(dst.size()).map_err(|_| {
            RhiError::AllocationFailed(
                "TLAS instance buffer size does not fit in the host address space".into(),
            )
        })?;
        if end > dst_size {
            return Err(RhiError::AllocationFailed(format!(
                "TLAS instance {index} (stride {stride}) exceeds the instance buffer"
            )));
        }
        let base = dst.cpu().ok_or_else(|| {
            RhiError::AllocationFailed("instance buffer is not CPU-mapped".into())
        })?;
        // SAFETY: bounds-checked above, and `base` is valid for `dst.size()` mapped bytes.
        let ptr = unsafe { base.add(offset) };
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.write_tlas_instance(ptr, instance));
        Ok(())
    }

    /// Create a transient command buffer for recording.
    pub fn create_command_buffer(&self) -> RhiResult<CommandBuffer> {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.create_command_buffer())
            .map(|resource| self.own(resource))
    }

    /// Create a command buffer pre-configured with swapchain image views.
    /// Use this for the main render loop where you need to render to swapchain images.
    pub fn create_command_buffer_for_swapchain(
        &self,
        swapchain: &Swapchain,
        frame_index: usize,
    ) -> RhiResult<CommandBuffer> {
        self.ensure_owns(swapchain, "swapchain")?;
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.create_command_buffer_for_swapchain(swapchain, frame_index))
            .map(|resource| self.own(resource))
    }

    /// Get the primary queue.
    pub fn queue(&self) -> &Queue {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.queue())
    }

    /// Create a timeline semaphore.
    pub fn create_timeline_semaphore(&self, initial_value: u64) -> RhiResult<TimelineSemaphore> {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.create_timeline_semaphore(initial_value))
            .map(|resource| self.own(resource))
    }

    /// Create a GPU timestamp pool.
    pub fn create_query_pool(&self, count: u32) -> RhiResult<QueryPool> {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.create_query_pool(count))
            .map(|resource| self.own(resource))
    }

    /// Destroy a query pool after its GPU work completes.
    pub fn destroy_query_pool(&self, pool: QueryPool) {
        self.assert_owns(&pool, "query pool");
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.destroy_query_pool(pool))
    }

    /// Nanoseconds per timestamp tick.
    pub fn timestamp_period_ns(&self) -> f64 {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.timestamp_period_ns())
    }

    /// Read timestamp ticks after the writing GPU work completes.
    pub fn read_timestamps(&self, pool: &QueryPool) -> RhiResult<Vec<u64>> {
        self.ensure_owns(pool, "query pool")?;
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.read_timestamps(pool))
    }

    /// Elapsed milliseconds between two timestamp slots, or `None` for invalid samples.
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

    /// Wait for the device to be idle.
    pub fn wait_idle(&self) {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.wait_idle())
    }

    /// Destroy a buffer, releasing its storage immediately.
    ///
    /// The RHI tracks no lifetimes: the caller guarantees the GPU is done, via
    /// [`wait_idle`](Self::wait_idle), [`wait_for_frame`](Self::wait_for_frame), or the
    /// swapchain's frames-in-flight fence. Destroying a resource an in-flight submission still
    /// references is a use-after-free.
    pub fn destroy_allocation(&self, allocation: Allocation) {
        self.assert_owns(&allocation, "allocation");
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.destroy_allocation(allocation))
    }

    /// Destroy a texture. Same contract as [`destroy_allocation`](Self::destroy_allocation); the
    /// Its shader handles are recycled at once, so a still-in-flight draw may read a new texture
    /// that reuses them.
    pub fn destroy_texture(&self, mut texture: Texture) {
        self.assert_owns(&texture, "texture");
        for id in texture.views.drain(..) {
            backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.destroy_texture_view(id));
        }
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.destroy_texture(texture))
    }

    /// Destroy a sampler. Same lifetime contract as
    /// [`destroy_allocation`](Self::destroy_allocation).
    pub fn destroy_sampler(&self, sampler: Sampler) {
        self.assert_owns(&sampler, "sampler");
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.destroy_sampler(sampler))
    }

    /// Wait for a specific frame's fence before reusing resources.
    pub fn wait_for_frame(&self, frame_index: usize) {
        backend_dispatch!(self.inner.as_ref(), DeviceInner, d => d.wait_for_frame(frame_index))
    }

    /// Get raw Vulkan handles for escape-hatch scenarios (e.g. ImGui).
    /// Only available with the vulkan feature.
    #[cfg(feature = "vulkan")]
    pub fn vulkan_handles(&self) -> crate::raw::VulkanHandles {
        backend_expect!(self.inner.as_ref(), DeviceInner::Vulkan).vulkan_handles()
    }
}

pub(crate) fn validate_texture_view(
    source: &Texture,
    view: &TextureViewDesc,
    required_usage: crate::texture::TextureUsage,
) -> RhiResult<()> {
    use crate::texture::{ALL_LAYERS, ALL_MIPS};

    if !source.desc().usage.contains(required_usage) {
        return Err(RhiError::Unsupported(format!(
            "texture view requires source usage {required_usage:?}"
        )));
    }
    if let Some(format) = view.format
        && format != source.desc().format
    {
        return Err(RhiError::Unsupported(
            "format-reinterpreting texture views are not in the portable Metal/Vulkan baseline"
                .into(),
        ));
    }

    let base_mip = u32::from(view.base_mip);
    let mip_count = if view.mip_count == ALL_MIPS {
        source.desc().mip_levels.saturating_sub(base_mip)
    } else {
        u32::from(view.mip_count)
    };
    if base_mip >= source.desc().mip_levels
        || mip_count == 0
        || mip_count > source.desc().mip_levels - base_mip
    {
        return Err(RhiError::Unsupported(
            "texture view mip range is outside the source texture".into(),
        ));
    }

    let base_layer = u32::from(view.base_layer);
    let layer_count = if view.layer_count == ALL_LAYERS {
        source.desc().array_layers.saturating_sub(base_layer)
    } else {
        u32::from(view.layer_count)
    };
    if base_layer >= source.desc().array_layers
        || layer_count == 0
        || layer_count > source.desc().array_layers - base_layer
    {
        return Err(RhiError::Unsupported(
            "texture view layer range is outside the source texture".into(),
        ));
    }

    Ok(())
}

fn aligned_backing_size(size: u64, align: u64) -> RhiResult<u64> {
    size.max(1).checked_add(align - 1).ok_or_else(|| {
        RhiError::AllocationFailed(format!(
            "allocation size {size} with alignment {align} overflows u64"
        ))
    })
}

impl CommandBuffer {
    /// Get the raw Vulkan command buffer handle for escape-hatch scenarios.
    #[cfg(feature = "vulkan")]
    pub fn vulkan_command_buffer(&self) -> ash::vk::CommandBuffer {
        backend_expect!(&self.inner, crate::command::CommandBufferInner::Vulkan).command_buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture::TextureUsage;
    use crate::types::{Format, TextureId};

    fn test_texture() -> Texture {
        Texture {
            id: TextureId(0),
            gpu_address: GpuPtr::NULL,
            handle: crate::TextureHandle::NULL,
            views: Vec::new(),
            desc: TextureDesc {
                mip_levels: 4,
                array_layers: 2,
                usage: TextureUsage::SAMPLED,
                ..Default::default()
            },
            _owner: None,
        }
    }

    #[test]
    fn portable_texture_views_reject_format_reinterpretation() {
        let source = test_texture();
        let view = TextureViewDesc {
            format: Some(Format::R32Float),
            ..Default::default()
        };
        assert!(validate_texture_view(&source, &view, TextureUsage::SAMPLED).is_err());
    }

    #[test]
    fn portable_texture_views_validate_subresource_ranges() {
        let source = test_texture();
        let valid = TextureViewDesc {
            base_mip: 1,
            mip_count: 3,
            base_layer: 1,
            layer_count: 1,
            ..Default::default()
        };
        assert!(validate_texture_view(&source, &valid, TextureUsage::SAMPLED).is_ok());

        let invalid = TextureViewDesc {
            base_mip: 4,
            ..Default::default()
        };
        assert!(validate_texture_view(&source, &invalid, TextureUsage::SAMPLED).is_err());
    }

    #[test]
    fn aligned_allocation_size_overflow_is_reported_before_backend_use() {
        assert!(aligned_backing_size(u64::MAX, 16).is_err());
        assert_eq!(aligned_backing_size(0, 1).unwrap(), 1);
        assert_eq!(aligned_backing_size(17, 16).unwrap(), 32);
    }
}
