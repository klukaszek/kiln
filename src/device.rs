use crate::accel::AccelerationStructure;
use crate::command::CommandBuffer;
use crate::error::{RhiError, RhiResult};
use crate::memory::{
    BufferDesc, GpuAllocation, GpuAllocator, GpuAllocatorDesc, GpuBuffer, MemoryType,
};
use crate::pipeline::{
    ComputePso, ComputePsoDesc, GraphicsPso, GraphicsPsoDesc, MeshletPso, MeshletPsoDesc,
};
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

    /// Clip-space Y convention for this backend.
    pub fn clip_space_y(&self) -> ClipSpaceY {
        match &self.inner {
            #[cfg(feature = "vulkan")]
            DeviceInner::Vulkan(_) => ClipSpaceY::Down,
            #[cfg(feature = "metal")]
            DeviceInner::Metal(_) => ClipSpaceY::Up,
        }
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

    /// Create a user-land GPU allocator over large pointer-first backing blocks.
    pub fn create_gpu_allocator(&self, desc: GpuAllocatorDesc) -> GpuAllocator<'_> {
        GpuAllocator::new(self, desc)
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
            buffer.gpu_address().0 & (align - 1),
            0,
            "backend returned a misaligned GPU address for malloc_aligned",
        );

        Ok(GpuAllocation { buffer, size })
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

    /// Create a texture in caller-owned GPU memory.
    ///
    /// `texture_gpu` must point to an allocation of at least
    /// `texture_size_align(desc).size` bytes and satisfy that alignment.
    /// The caller owns the allocation lifetime and must keep it alive while the
    /// returned texture is live.
    pub fn create_texture(
        &self,
        desc: &TextureDesc,
        texture_gpu: GpuAddress,
    ) -> RhiResult<Texture> {
        backend_dispatch!(&self.inner, DeviceInner, d => d.create_texture(desc, texture_gpu))
    }

    /// Create a sampled (SRV) view of an existing texture and register it in the bindless heap.
    ///
    /// Returns a new `TextureId` that can be used in shaders like any other texture.
    /// The view shares storage with the source texture — destroying the source while
    /// the view `TextureId` is in use is undefined behavior.
    /// Matches Aaltonen's `gpuTextureViewDescriptor(texture, viewDesc)`.
    pub fn texture_view_descriptor(
        &self,
        source: &Texture,
        view: &GpuViewDesc,
    ) -> RhiResult<crate::types::TextureId> {
        backend_dispatch!(&self.inner, DeviceInner, d => d.texture_view_descriptor(source, view))
    }

    /// Create a storage (UAV) view of an existing texture and register it in the bindless heap.
    ///
    /// Returns a new `TextureId`. On Metal, storage views share the same representation as
    /// sampled views — access mode is controlled by the shader binding type.
    /// Matches Aaltonen's `gpuRWTextureViewDescriptor(texture, viewDesc)`.
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

    /// Create a mesh-shader graphics pipeline.
    ///
    /// Matches Aaltonen's `gpuCreateGraphicsMeshletPipeline(meshletIR, pixelIR, desc)`.
    ///
    /// On Vulkan, requires `VK_EXT_mesh_shader` (enabled automatically when supported).
    /// On Metal, uses the Metal 4 mesh render pipeline path.
    /// Returns `RhiError::Unsupported` if the device does not support mesh shaders.
    ///
    /// Matches the spec's `gpuCreateGraphicsMeshletPipeline(meshletIR, pixelIR, desc)`.
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

    /// Return the GPU address of a built acceleration structure.
    ///
    /// On Vulkan: `vkGetAccelerationStructureDeviceAddressKHR`.
    /// On Metal: the opaque 64-bit `MTLResourceID` (cached at build time).
    ///
    /// Store this in a root struct field typed `u64` (via `.0`) and pass it to the
    /// intersection shader / ray generation kernel — same convention as
    /// [`GpuBuffer::gpu_address`](crate::GpuBuffer::gpu_address).
    pub fn accel_gpu_address(&self, accel: &AccelerationStructure) -> GpuAddress {
        match &accel.inner {
            #[cfg(feature = "vulkan")]
            crate::accel::AccelInner::Vulkan(a) => GpuAddress(a.device_address),
            #[cfg(feature = "metal")]
            crate::accel::AccelInner::Metal(a) => GpuAddress(a.gpu_resource_id),
        }
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
        let base = dst.mapped_ptr().ok_or_else(|| {
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
