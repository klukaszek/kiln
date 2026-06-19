//! Device creation and the top-level resource management API.

use crate::accel::AccelerationStructure;
use crate::command::CommandBuffer;
use crate::error::{RhiError, RhiResult};
use crate::memory::{BufferDesc, GpuAllocation, GpuBuffer, GpuPod, MemoryType};
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
use crate::texture::{GpuViewDesc, Texture, TextureDesc, TextureSizeAlign};
use crate::types::{BlasDesc, ClipSpaceY, GpuAddress, TlasDesc, TlasInstance};

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
    /// Vulkan requires DescriptorBuffer to align with Aaltonen; if unsupported, device creation fails.
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
    pub(crate) inner: DeviceInner,
}

pub(crate) enum DeviceInner {
    #[cfg(feature = "vulkan")]
    Vulkan(Box<crate::backend::vulkan::device::VulkanDevice>),
    #[cfg(feature = "metal")]
    Metal(Box<crate::backend::metal::device::MetalDevice>),
}

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
                Ok(Self {
                    inner: DeviceInner::Vulkan(Box::new(vk_device)),
                })
            }
            #[cfg(feature = "metal")]
            Backend::Metal => {
                let mtl_device = crate::backend::metal::device::MetalDevice::new(desc)?;
                Ok(Self {
                    inner: DeviceInner::Metal(Box::new(mtl_device)),
                })
            }
            #[allow(unreachable_patterns)]
            _ => Err(crate::error::RhiError::Unsupported(format!(
                "Backend '{}' is not compiled in. Enable the corresponding feature.",
                backend
            ))),
        }
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
        match &self.inner {
            #[cfg(feature = "vulkan")]
            DeviceInner::Vulkan(_) => "Vulkan",
            #[cfg(feature = "metal")]
            DeviceInner::Metal(_) => "Metal",
        }
    }

    /// The active bindless mode selected by the backend.
    pub fn bindless_mode(&self) -> BindlessMode {
        backend_dispatch!(&self.inner, DeviceInner, d => d.bindless_mode())
    }

    /// Clip-space Y convention — always [`ClipSpaceY::Up`].
    ///
    /// Kiln normalizes clip space to Y-up (Metal/D3D convention) on every backend, so
    /// the same NDC renders identically everywhere and a single Y-up projection works
    /// without per-backend branches. The Vulkan backend achieves this with a
    /// negative-height viewport (see `set_viewport`); Metal is Y-up natively.
    pub fn clip_space_y(&self) -> ClipSpaceY {
        ClipSpaceY::Up
    }

    /// Create a presentation surface from raw window handles.
    pub fn create_surface(&self, desc: &SurfaceDesc) -> RhiResult<Surface> {
        backend_dispatch!(&self.inner, DeviceInner, d => d.create_surface(desc))
    }

    /// Create a swapchain for the given surface.
    pub fn create_swapchain(
        &self,
        surface: &Surface,
        desc: &SwapchainDesc,
    ) -> RhiResult<Swapchain> {
        backend_dispatch!(&self.inner, DeviceInner, d => d.create_swapchain(surface, desc))
    }

    /// Recreate swapchain (on resize).
    pub fn recreate_swapchain(
        &self,
        swapchain: &mut Swapchain,
        desc: &SwapchainDesc,
    ) -> RhiResult<()> {
        backend_dispatch!(&self.inner, DeviceInner, d => d.recreate_swapchain(swapchain, desc))
    }

    /// Create a GPU buffer.
    pub fn create_buffer(&self, desc: &BufferDesc) -> RhiResult<GpuBuffer> {
        backend_dispatch!(&self.inner, DeviceInner, d => d.create_buffer(desc))
    }

    /// Allocate GPU memory and return a pointer-first allocation.
    pub fn malloc(&self, size: u64, memory: MemoryType) -> RhiResult<GpuAllocation> {
        self.malloc_aligned(size, 16, memory)
    }

    /// Allocate GPU memory with explicit alignment.
    pub fn malloc_aligned(
        &self,
        size: u64,
        align: u64,
        memory: MemoryType,
    ) -> RhiResult<GpuAllocation> {
        let align = align.max(1);
        assert!(align.is_power_of_two(), "alignment must be a power of two");

        let buffer = self.create_buffer(&BufferDesc {
            size,
            memory,
            label: None,
        })?;

        debug_assert_eq!(
            buffer.gpu().0 & (align - 1),
            0,
            "backend returned a misaligned GPU address for malloc_aligned",
        );

        Ok(GpuAllocation { buffer, size })
    }

    /// Allocate a [`MemoryType::Default`] buffer sized for `data` and upload the contents in
    /// one step. The one-shot form of `malloc` + `GpuAllocation::upload_slice` for persistent
    /// scene data (vertex buffers, material tables, lookup tables). For GPU-only resources or
    /// transient per-frame arguments use `malloc`/`malloc_aligned` or a bump allocator.
    pub fn upload_slice<T: GpuPod>(&self, data: &[T]) -> RhiResult<GpuAllocation> {
        let size = std::mem::size_of_val(data).max(1) as u64;
        let alloc = self.malloc(size, MemoryType::Default)?;
        if !data.is_empty() {
            alloc.upload_slice(data)?;
        }
        Ok(alloc)
    }

    /// Free a pointer-first allocation.
    pub fn free(&self, allocation: GpuAllocation) {
        self.destroy_buffer(allocation.into_buffer());
    }

    /// Translate a CPU-mapped pointer to a GPU virtual address, if possible.
    pub fn host_to_device_pointer(&self, cpu_ptr: *const u8) -> Option<GpuAddress> {
        backend_dispatch!(&self.inner, DeviceInner, d => d.host_to_device_pointer(cpu_ptr))
    }

    /// Query the size/alignment required for `create_texture`.
    pub fn texture_size_align(&self, desc: &TextureDesc) -> RhiResult<TextureSizeAlign> {
        backend_dispatch!(&self.inner, DeviceInner, d => d.texture_size_align(desc))
    }

    /// Create a texture in caller-owned GPU memory. `texture_gpu` must point to an allocation
    /// meeting `texture_size_align(desc)` and be kept alive while the texture is live.
    pub fn create_texture(
        &self,
        desc: &TextureDesc,
        texture_gpu: GpuAddress,
    ) -> RhiResult<Texture> {
        backend_dispatch!(&self.inner, DeviceInner, d => d.create_texture(desc, texture_gpu))
    }

    /// Register a sampled (SRV) view of `source` in the bindless heap, returning its `TextureId`.
    /// The view shares `source`'s storage; using the id after `source` is destroyed is UB.
    pub fn texture_view_descriptor(
        &self,
        source: &Texture,
        view: &GpuViewDesc,
    ) -> RhiResult<crate::types::TextureId> {
        backend_dispatch!(&self.inner, DeviceInner, d => d.texture_view_descriptor(source, view))
    }

    /// Register a storage (UAV) view of `source` in the bindless heap, returning its `TextureId`.
    pub fn rw_texture_view_descriptor(
        &self,
        source: &Texture,
        view: &GpuViewDesc,
    ) -> RhiResult<crate::types::TextureId> {
        backend_dispatch!(&self.inner, DeviceInner, d => d.rw_texture_view_descriptor(source, view))
    }

    /// Create a sampler.
    pub fn create_sampler(&self, desc: &SamplerDesc) -> RhiResult<Sampler> {
        backend_dispatch!(&self.inner, DeviceInner, d => d.create_sampler(desc))
    }

    /// Bindless handle to store in a [`TextureHandle`](crate::TextureHandle) root field for the
    /// sampled view `id` from [`texture_view_descriptor`](Self::texture_view_descriptor). This is
    /// the texture analogue of [`AccelerationStructure::gpu`](crate::AccelerationStructure::gpu):
    /// the heap index on Vulkan, the `gpuResourceID` on Metal. The shader samples it with
    /// `root.tex.Sample(root.smp, uv)`.
    pub fn bindless_texture_handle(&self, id: crate::types::TextureId) -> GpuAddress {
        backend_dispatch!(&self.inner, DeviceInner, d => d.bindless_texture_handle(id))
    }

    /// Bindless handle to store in a [`SamplerHandle`](crate::SamplerHandle) root field for the
    /// sampler `id` from [`create_sampler`](Self::create_sampler).
    pub fn bindless_sampler_handle(&self, id: crate::types::SamplerId) -> GpuAddress {
        backend_dispatch!(&self.inner, DeviceInner, d => d.bindless_sampler_handle(id))
    }

    /// Create a shader module.
    pub fn create_shader_module(&self, desc: &ShaderModuleDesc) -> RhiResult<ShaderModule> {
        backend_dispatch!(&self.inner, DeviceInner, d => d.create_shader_module(desc))
    }

    /// Create a graphics pipeline state object.
    ///
    /// Matches the spec's `gpuCreateGraphicsPipeline(vertexIR, pixelIR, desc)` — shaders are
    /// arguments, not part of `desc`. `vertex`/`pixel` must outlive only this call.
    pub fn create_graphics_pso(
        &self,
        desc: &GraphicsPsoDesc,
        vertex: &ShaderModule,
        pixel: &ShaderModule,
    ) -> RhiResult<GraphicsPso> {
        match (&self.inner, &vertex.inner, &pixel.inner) {
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
        }
    }

    /// Create a compute pipeline state object.
    ///
    /// Matches the spec's `gpuCreateComputePipeline(computeIR)`.
    pub fn create_compute_pso(
        &self,
        desc: &ComputePsoDesc,
        compute: &ShaderModule,
    ) -> RhiResult<ComputePso> {
        match (&self.inner, &compute.inner) {
            #[cfg(feature = "vulkan")]
            (DeviceInner::Vulkan(d), ShaderModuleInner::Vulkan(c)) => d.create_compute_pso(desc, c),
            #[cfg(feature = "metal")]
            (DeviceInner::Metal(d), ShaderModuleInner::Metal(c)) => d.create_compute_pso(desc, c),
            #[allow(unreachable_patterns)]
            _ => unreachable!("shader module backend does not match device backend"),
        }
    }

    /// Create a mesh-shader graphics pipeline (spec: `gpuCreateGraphicsMeshletPipeline`).
    /// Requires `VK_EXT_mesh_shader` on Vulkan; returns `RhiError::Unsupported` if mesh
    /// shaders are unavailable.
    pub fn create_meshlet_pso(
        &self,
        desc: &MeshletPsoDesc,
        mesh: &ShaderModule,
        pixel: &ShaderModule,
    ) -> RhiResult<MeshletPso> {
        match (&self.inner, &mesh.inner, &pixel.inner) {
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
        }
    }

    /// Allocate a Bottom-Level Acceleration Structure.
    ///
    /// The returned `AccelerationStructure` must be built via `cmd.build_blas(as, desc)`
    /// before it can be referenced in a TLAS instance.
    pub fn create_blas(&self, desc: &BlasDesc) -> RhiResult<AccelerationStructure> {
        backend_dispatch!(&self.inner, DeviceInner, d => d.create_blas(desc))
    }

    /// Allocate a Top-Level Acceleration Structure.
    pub fn create_tlas(&self, desc: &TlasDesc) -> RhiResult<AccelerationStructure> {
        backend_dispatch!(&self.inner, DeviceInner, d => d.create_tlas(desc))
    }

    /// Size in bytes of one native TLAS instance descriptor for this backend. The instance
    /// buffer passed to `build_tlas` must use this stride; fill entries with
    /// [`write_tlas_instance`](Self::write_tlas_instance).
    pub fn tlas_instance_stride(&self) -> usize {
        backend_dispatch!(&self.inner, DeviceInner, d => d.tlas_instance_stride())
    }

    /// Encode `instance` into slot `index` of a CPU-mapped instance buffer, using the active
    /// backend's native instance layout (Vulkan `VkAccelerationStructureInstanceKHR`; Metal
    /// indirect descriptor). Size the buffer as `instance_count * tlas_instance_stride()`.
    pub fn write_tlas_instance(
        &self,
        dst: &crate::memory::GpuAllocation,
        index: usize,
        instance: &TlasInstance,
    ) -> RhiResult<()> {
        let stride = self.tlas_instance_stride();
        let offset = index * stride;
        let capacity = dst.size() as usize;
        if offset + stride > capacity {
            return Err(RhiError::AllocationFailed(format!(
                "TLAS instance {index} (stride {stride}) exceeds instance buffer ({capacity} bytes)"
            )));
        }
        let base = dst.cpu().ok_or_else(|| {
            RhiError::AllocationFailed("instance buffer is not CPU-mapped".into())
        })?;
        // SAFETY: `offset + stride <= capacity`, and `base` is valid for `capacity` mapped
        // bytes, so `ptr` points to `stride` writable bytes for the backend to fill.
        let ptr = unsafe { base.add(offset) };
        backend_dispatch!(&self.inner, DeviceInner, d => d.write_tlas_instance(ptr, instance));
        Ok(())
    }

    /// Create a transient command buffer for recording.
    pub fn create_command_buffer(&self) -> RhiResult<CommandBuffer> {
        backend_dispatch!(&self.inner, DeviceInner, d => d.create_command_buffer())
    }

    /// Create a command buffer pre-configured with swapchain image views.
    /// Use this for the main render loop where you need to render to swapchain images.
    pub fn create_command_buffer_for_swapchain(
        &self,
        swapchain: &Swapchain,
    ) -> RhiResult<CommandBuffer> {
        backend_dispatch!(&self.inner, DeviceInner, d => d.create_command_buffer_for_swapchain(swapchain))
    }

    /// Get the primary queue.
    pub fn queue(&self) -> &Queue {
        backend_dispatch!(&self.inner, DeviceInner, d => d.queue())
    }

    /// Create a timeline semaphore.
    pub fn create_timeline_semaphore(&self, initial_value: u64) -> RhiResult<TimelineSemaphore> {
        backend_dispatch!(&self.inner, DeviceInner, d => d.create_timeline_semaphore(initial_value))
    }

    // -- GPU timestamp queries --

    /// Create a [`QueryPool`] with `count` GPU timestamp slots.
    ///
    /// Typical use is two slots per frame-in-flight: write a timestamp at the start and end of
    /// the frame's command buffer, then read the delta back once that frame's fence has signalled.
    /// See [`CommandBuffer::write_timestamp`], [`read_timestamps`](Self::read_timestamps), and
    /// [`gpu_elapsed_ms`](Self::gpu_elapsed_ms).
    pub fn create_query_pool(&self, count: u32) -> RhiResult<QueryPool> {
        backend_dispatch!(&self.inner, DeviceInner, d => d.create_query_pool(count))
    }

    /// Destroy a query pool. Like other RHI resources, pools are freed explicitly; the pool must
    /// not be in use by an in-flight command buffer.
    pub fn destroy_query_pool(&self, pool: QueryPool) {
        backend_dispatch!(&self.inner, DeviceInner, d => d.destroy_query_pool(pool))
    }

    /// Nanoseconds per timestamp tick for this device. Multiply a tick delta from
    /// [`read_timestamps`](Self::read_timestamps) by this to get nanoseconds.
    pub fn timestamp_period_ns(&self) -> f64 {
        backend_dispatch!(&self.inner, DeviceInner, d => d.timestamp_period_ns())
    }

    /// Read back all raw timestamp tick values from `pool`. The GPU work that wrote the
    /// timestamps must have completed (e.g. the frame's fence has been waited) before calling;
    /// stale or never-written slots read back as `0`.
    pub fn read_timestamps(&self, pool: &QueryPool) -> RhiResult<Vec<u64>> {
        backend_dispatch!(&self.inner, DeviceInner, d => d.read_timestamps(pool))
    }

    /// Convenience over [`read_timestamps`](Self::read_timestamps): elapsed GPU time in
    /// milliseconds between timestamp slots `begin` and `end`. Returns `Ok(None)` when either
    /// slot is unwritten (reads back `0`) or the values are non-monotonic, which happens for the
    /// first few frames before a slot's pool has been written and waited on.
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
        backend_dispatch!(&self.inner, DeviceInner, d => d.wait_idle())
    }

    /// Destroy a buffer.
    pub fn destroy_buffer(&self, buffer: GpuBuffer) {
        backend_dispatch!(&self.inner, DeviceInner, d => d.destroy_buffer(buffer))
    }

    /// Destroy a texture.
    pub fn destroy_texture(&self, texture: Texture) {
        backend_dispatch!(&self.inner, DeviceInner, d => d.destroy_texture(texture))
    }

    /// Wait for a specific frame's fence before reusing resources.
    pub fn wait_for_frame(&self, frame_index: usize) {
        backend_dispatch!(&self.inner, DeviceInner, d => d.wait_for_frame(frame_index))
    }

    /// Get raw Vulkan handles for escape-hatch scenarios (e.g. ImGui).
    /// Only available with the vulkan feature.
    #[cfg(feature = "vulkan")]
    pub fn vulkan_handles(&self) -> crate::backend::vulkan::device::VulkanHandles {
        match &self.inner {
            DeviceInner::Vulkan(d) => d.vulkan_handles(),
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}

impl CommandBuffer {
    /// Get the raw Vulkan command buffer handle for escape-hatch scenarios.
    #[cfg(feature = "vulkan")]
    pub fn vulkan_command_buffer(&self) -> ash::vk::CommandBuffer {
        match &self.inner {
            crate::command::CommandBufferInner::Vulkan(cmd) => cmd.command_buffer,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }
}
