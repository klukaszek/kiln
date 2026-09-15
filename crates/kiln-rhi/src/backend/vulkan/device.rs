use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, VecDeque};
use std::ffi::{CStr, CString, c_char};
use std::rc::Rc;

use ash::{
    Device, Entry, Instance,
    ext::{debug_utils, descriptor_heap, mesh_shader as vk_mesh_shader},
    khr::{
        acceleration_structure as vk_accel_structure, device_address_commands, surface, swapchain,
    },
    vk,
    vk::TaggedStructure as _,
};
use smallvec::SmallVec;

use crate::command::{CommandBuffer, CommandBufferInner};
use crate::device::DeviceDesc;
use crate::error::{RhiError, RhiResult};
use crate::memory::{Allocation, AllocationDesc, AllocationInner, MemoryType};
use crate::pipeline::{
    ComputePso, ComputePsoDesc, ComputePsoInner, GraphicsPso, GraphicsPsoDesc, GraphicsPsoInner,
    MeshletPso, MeshletPsoDesc, MeshletPsoInner,
};
use crate::query::{QueryPool, QueryPoolInner};
use crate::queue::{Queue, QueueInner};
use crate::shader::{ShaderModule, ShaderModuleDesc, ShaderModuleInner};
use crate::surface::{Surface, SurfaceDesc, SurfaceInner};
use crate::swapchain::{Swapchain, SwapchainInner};
use crate::sync::{TimelineSemaphore, TimelineSemaphoreInner};
use crate::types::{
    AddressMode, BuildAccelFlags, CompareOp, Format, GeometryFlags, GpuPtr, MAX_BINDLESS_SAMPLERS,
    MAX_BINDLESS_TEXTURES, MAX_FRAMES_IN_FLIGHT, SamplerId, TextureId,
};

use super::command::VulkanCommandBuffer;
use super::memory::{SharedBufferPool, VulkanBuffer, VulkanBufferPool};
use super::pipeline::{
    VulkanComputePso, VulkanGraphicsPso, VulkanGraphicsPsoDesc, VulkanMeshletPso,
    VulkanMeshletPsoDesc,
};
use super::query::VulkanQueryPool;
use super::queue::{VulkanQueue, VulkanRetiredResource};
use super::shader::VulkanShaderModule;
use super::surface::VulkanSurface;
use super::sync::VulkanTimelineSemaphore;
use super::texture::VulkanTexture;

#[derive(Clone)]
pub(crate) struct BufferAllocation {
    pub base: GpuPtr<u8>,
    pub size: u64,
    pub memory: vk::DeviceMemory,
    /// Offset within the pooled block; images placed here bind at this plus their own offset.
    pub memory_offset: u64,
}

/// Reverse index for CPU-mapped allocations: encoding looks up by GPU address, the public
/// pointer bridge by CPU address, and a second index keeps both a predecessor lookup.
pub(crate) struct MappedAllocation {
    gpu_base: GpuPtr<u8>,
    size: u64,
}

fn resolve_mapped_pointer(
    allocations: &BTreeMap<usize, MappedAllocation>,
    ptr: usize,
) -> Option<GpuPtr<u8>> {
    let (&base, allocation) = allocations.range(..=ptr).next_back()?;
    let offset = (ptr - base) as u64;
    (offset < allocation.size).then(|| allocation.gpu_base.offset(offset))
}

/// One app-owned descriptor heap: a mapped allocation the driver indexes by slot.
///
/// Descriptors occupy `[0, reserved_offset)` so slot `i` sits at `i * descriptor_size`; the
/// driver's reserved range is parked at the tail, keeping shader-visible indices identical to
/// the `TextureId`/`SamplerId` the RHI already hands out.
pub(crate) struct DescriptorHeap {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub mapped_ptr: *mut u8,
    pub size: u64,
    pub gpu_address: GpuPtr<u8>,
    pub descriptor_size: u64,
    pub reserved_offset: u64,
    pub reserved_size: u64,
}

impl DescriptorHeap {
    /// Byte range of slot `index`. The heap is sized from the same constant the id allocators
    /// cap at, so an id that exists is always in range.
    pub(crate) fn slot(&self, index: u32) -> std::ops::Range<usize> {
        let start = index as u64 * self.descriptor_size;
        let end = start + self.descriptor_size;
        debug_assert!(end <= self.reserved_offset, "descriptor slot {index} out of range");
        start as usize..end as usize
    }

    fn bind_info(&self) -> vk::BindHeapInfoEXT<'static> {
        vk::BindHeapInfoEXT::default()
            .heap_range(
                vk::DeviceAddressRangeEXT::default()
                    .address(self.gpu_address.address)
                    .size(self.size),
            )
            .reserved_range_offset(self.reserved_offset)
            .reserved_range_size(self.reserved_size)
    }
}

/// The two heaps every command buffer binds: resources (images) and samplers.
pub(crate) struct DescriptorHeaps {
    pub resource: DescriptorHeap,
    pub sampler: DescriptorHeap,
}

/// Buffer allocations keyed by GPU base address, enabling O(log n) address->buffer
/// resolution instead of a linear scan on every indirect/copy/index command.
pub(crate) type SharedAllocations = Rc<RefCell<BTreeMap<u64, BufferAllocation>>>;
type SharedMappedAllocations = Rc<RefCell<BTreeMap<usize, MappedAllocation>>>;
pub(crate) type SharedTextures = Rc<RefCell<Vec<Option<VulkanTexture>>>>;
pub(crate) type SharedTextureFreeIds = Rc<RefCell<Vec<TextureId>>>;
pub(crate) type SharedSamplerFreeIds = Rc<RefCell<Vec<SamplerId>>>;

pub struct VulkanDevice {
    pub(crate) entry: Entry,
    pub(crate) instance: Instance,
    pub(crate) device: Device,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) queue: Queue,
    pub(crate) present_queue: vk::Queue,
    pub(crate) command_pool: vk::CommandPool,
    pub(crate) pipeline_cache: vk::PipelineCache,
    /// Shared with the queue so retirement can hand ranges back.
    pub(crate) buffer_pool: SharedBufferPool,
    pub(crate) device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// Nanoseconds per timestamp tick (`VkPhysicalDeviceLimits::timestampPeriod`).
    pub(crate) timestamp_period: f32,

    pub(crate) surface_loader: surface::Instance,
    pub(crate) swapchain_loader: swapchain::Device,
    pub(crate) descriptor_heap_loader: descriptor_heap::Device,
    pub(crate) address_commands_loader: device_address_commands::Device,

    pub(crate) debug_utils_loader: Option<debug_utils::Instance>,
    pub(crate) debug_callback: vk::DebugUtilsMessengerEXT,
    /// Device-level `VK_EXT_debug_utils`: object names and pass label regions.
    pub(crate) debug_labels: Option<debug_utils::Device>,

    pub(crate) descriptor_heaps: DescriptorHeaps,
    /// `(usage, alignment, memory_type_bits)` for [`Self::buffer_requirements`], one entry per
    /// usage class, probed on first use.
    buffer_requirements_probe: RefCell<Vec<(vk::BufferUsageFlags, u64, u32)>>,
    pub(crate) textures: SharedTextures,
    pub(crate) next_texture_id: RefCell<u32>,
    pub(crate) free_texture_ids: SharedTextureFreeIds,
    pub(crate) allocations: SharedAllocations,
    pub(crate) mapped_allocations: SharedMappedAllocations,

    pub(crate) next_sampler_id: RefCell<u32>,
    pub(crate) free_sampler_ids: SharedSamplerFreeIds,

    /// User timeline semaphores stay alive until device teardown. This permits callers to use a
    /// temporary `SubmitDesc` without destroying a semaphore still referenced by queued work.
    pub(crate) timeline_semaphores: RefCell<Vec<vk::Semaphore>>,
    pub(crate) query_pools: RefCell<Vec<vk::QueryPool>>,

    pub(crate) setup_command_buffer: vk::CommandBuffer,
    /// Whether the batched setup command buffer is currently recording initial image layouts.
    pub(crate) setup_recording: Cell<bool>,

    /// Cached mesh-shader loader, cloned into command buffers. Built once — recreating it (or
    /// the descriptor-buffer loader) per command buffer re-runs `vkGetDeviceProcAddr` for every
    /// entry point on the hot per-frame path.
    pub(crate) mesh_shader_loader: vk_mesh_shader::Device,

    /// Present when VK_KHR_acceleration_structure was enabled (for BLAS/TLAS builds).
    pub(crate) acceleration_structure: vk_accel_structure::Device,
}
/// Debug callback for Vulkan validation layers.
unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = unsafe { *p_callback_data };
    let message = if callback_data.p_message.is_null() {
        std::borrow::Cow::from("")
    } else {
        unsafe { std::ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy() }
    };

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            log::error!("[Vulkan {:?}] {}", message_type, message);
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            log::warn!("[Vulkan {:?}] {}", message_type, message);
        }
        _ => {
            log::debug!("[Vulkan {:?}] {}", message_type, message);
        }
    }

    vk::FALSE
}

/// Helper: find memory type index.
pub(crate) fn find_memorytype_index(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    memory_prop.memory_types[..memory_prop.memory_type_count as _]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (1 << index) & memory_req.memory_type_bits != 0
                && memory_type.property_flags & flags == flags
        })
        .map(|(index, _)| index as u32)
}

/// Convert RHI Format to Vulkan format.
/// Convert an RHI Format to a Vulkan VkFormat.
pub fn format_to_vk(format: Format) -> vk::Format {
    match format {
        Format::R8Unorm => vk::Format::R8_UNORM,
        Format::R8G8Unorm => vk::Format::R8G8_UNORM,
        Format::R8G8B8A8Unorm => vk::Format::R8G8B8A8_UNORM,
        Format::R8G8B8A8Srgb => vk::Format::R8G8B8A8_SRGB,
        Format::B8G8R8A8Unorm => vk::Format::B8G8R8A8_UNORM,
        Format::B8G8R8A8Srgb => vk::Format::B8G8R8A8_SRGB,
        Format::R16Float => vk::Format::R16_SFLOAT,
        Format::R16G16Float => vk::Format::R16G16_SFLOAT,
        Format::R16G16B16A16Float => vk::Format::R16G16B16A16_SFLOAT,
        Format::R32Float => vk::Format::R32_SFLOAT,
        Format::R32G32Float => vk::Format::R32G32_SFLOAT,
        Format::R32G32B32A32Float => vk::Format::R32G32B32A32_SFLOAT,
        Format::R10G10B10A2Unorm => vk::Format::A2B10G10R10_UNORM_PACK32,
        Format::R11G11B10Float => vk::Format::B10G11R11_UFLOAT_PACK32,
        Format::D16Unorm => vk::Format::D16_UNORM,
        Format::D32Float => vk::Format::D32_SFLOAT,
        Format::D24UnormS8Uint => vk::Format::D24_UNORM_S8_UINT,
        Format::D32FloatS8Uint => vk::Format::D32_SFLOAT_S8_UINT,
        Format::R16Uint => vk::Format::R16_UINT,
        Format::R32Uint => vk::Format::R32_UINT,
    }
}

/// Convert Vulkan format to RHI Format.
pub(crate) fn vk_to_format(format: vk::Format) -> Format {
    match format {
        vk::Format::R8_UNORM => Format::R8Unorm,
        vk::Format::R8G8_UNORM => Format::R8G8Unorm,
        vk::Format::R8G8B8A8_UNORM => Format::R8G8B8A8Unorm,
        vk::Format::R8G8B8A8_SRGB => Format::R8G8B8A8Srgb,
        vk::Format::B8G8R8A8_UNORM => Format::B8G8R8A8Unorm,
        vk::Format::B8G8R8A8_SRGB => Format::B8G8R8A8Srgb,
        vk::Format::D32_SFLOAT => Format::D32Float,
        _ => Format::R8G8B8A8Unorm, // fallback
    }
}

pub(crate) fn build_accel_flags_to_vk(
    flags: BuildAccelFlags,
) -> vk::BuildAccelerationStructureFlagsKHR {
    let mut out = vk::BuildAccelerationStructureFlagsKHR::empty();
    if flags.contains(BuildAccelFlags::ALLOW_UPDATE) {
        out |= vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE;
    }
    if flags.contains(BuildAccelFlags::PREFER_FAST_TRACE) {
        out |= vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE;
    }
    if flags.contains(BuildAccelFlags::PREFER_FAST_BUILD) {
        out |= vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD;
    }
    if flags.contains(BuildAccelFlags::MINIMIZE_MEMORY) {
        out |= vk::BuildAccelerationStructureFlagsKHR::LOW_MEMORY;
    }
    out
}

pub(crate) fn geometry_flags_to_vk(flags: GeometryFlags) -> vk::GeometryFlagsKHR {
    let mut out = vk::GeometryFlagsKHR::empty();
    if flags.contains(GeometryFlags::OPAQUE) {
        out |= vk::GeometryFlagsKHR::OPAQUE;
    }
    if flags.contains(GeometryFlags::NO_DUPLICATE_ANYHIT) {
        out |= vk::GeometryFlagsKHR::NO_DUPLICATE_ANY_HIT_INVOCATION;
    }
    out
}

/// Pick the best physical device satisfying every feature the RHI requires, plus the queue family
/// to use with it. Filtering on the full feature set before scoring keeps adapter choice from
/// depending on enumeration order.
fn select_physical_device(
    instance: &ash::Instance,
) -> RhiResult<(vk::PhysicalDevice, u32)> {
    let physical_devices = unsafe {
        instance
            .enumerate_physical_devices()
            .map_err(|e| RhiError::DeviceCreation(format!("Enumerate devices: {e}")))?
    };

    // Filter on the complete required feature set before scoring devices. Selecting the
    // first Vulkan 1.3 queue and discovering missing descriptor-buffer features later makes
    // adapter choice depend on enumeration order and can fail on perfectly usable systems.
    let selected = physical_devices
        .iter()
        .filter_map(|pdevice| {
            let props = unsafe { instance.get_physical_device_properties(*pdevice) };
            let api_version = props.api_version;
            let supports_vulkan_13 = vk::api_version_major(api_version) > 1
                || (vk::api_version_major(api_version) == 1
                    && vk::api_version_minor(api_version) >= 3);
            if !supports_vulkan_13 {
                return None;
            }

            let extensions =
                unsafe { instance.enumerate_device_extension_properties(*pdevice) }.ok()?;
            let has_ext = |needle: &[u8]| {
                extensions.iter().any(|ext| {
                    let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                    name.to_bytes() == needle
                })
            };
            if !has_ext(b"VK_KHR_swapchain")
                || !has_ext(b"VK_EXT_descriptor_heap")
                || !has_ext(b"VK_KHR_shader_untyped_pointers")
                || !has_ext(b"VK_KHR_device_address_commands")
                || !has_ext(b"VK_EXT_mesh_shader")
                || !has_ext(b"VK_KHR_unified_image_layouts")
                || !has_ext(b"VK_KHR_acceleration_structure")
                || !has_ext(b"VK_KHR_deferred_host_operations")
                || !has_ext(b"VK_KHR_ray_query")
                || !has_ext(b"VK_KHR_ray_tracing_maintenance1")
            {
                return None;
            }

            let mut vulkan11 = vk::PhysicalDeviceVulkan11Features::default();
            let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut vulkan13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut vulkan14 = vk::PhysicalDeviceVulkan14Features::default();
            let mut descriptor_heap = vk::PhysicalDeviceDescriptorHeapFeaturesEXT::default();
            let mut untyped_pointers =
                vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR::default();
            let mut address_commands =
                vk::PhysicalDeviceDeviceAddressCommandsFeaturesKHR::default();
            let mut unified_layouts =
                vk::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR::default();

            let mut mesh = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();
            let mut accel = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
            let mut ray_query = vk::PhysicalDeviceRayQueryFeaturesKHR::default();
            let mut rt_maintenance1 =
                vk::PhysicalDeviceRayTracingMaintenance1FeaturesKHR::default();

            let mut features2 = vk::PhysicalDeviceFeatures2::default()
                .push(&mut vulkan11)
                .push(&mut vulkan12)
                .push(&mut vulkan13)
                .push(&mut vulkan14)
                .push(&mut descriptor_heap)
                .push(&mut untyped_pointers)
                .push(&mut address_commands)
                .push(&mut unified_layouts);
            features2 = features2
                .push(&mut mesh)
                .push(&mut accel)
                .push(&mut ray_query)
                .push(&mut rt_maintenance1);
            unsafe { instance.get_physical_device_features2(*pdevice, &mut features2) };

            let base = features2.features;
            let required = base.shader_int64 != 0
                && vulkan11.shader_draw_parameters != 0
                && vulkan12.buffer_device_address != 0
                && vulkan12.timeline_semaphore != 0
                && vulkan12.host_query_reset != 0
                && vulkan13.dynamic_rendering != 0
                && vulkan13.synchronization2 != 0
                && vulkan14.maintenance5 != 0
                && descriptor_heap.descriptor_heap != 0
                && untyped_pointers.shader_untyped_pointers != 0
                && address_commands.device_address_commands != 0
                && mesh.mesh_shader != 0
                && unified_layouts.unified_image_layouts != 0
                && accel.acceleration_structure != 0
                && ray_query.ray_query != 0
                && rt_maintenance1.ray_tracing_maintenance1 != 0;
            if !required {
                return None;
            }

            let queue_family =
                unsafe { instance.get_physical_device_queue_family_properties(*pdevice) }
                    .iter()
                    .enumerate()
                    .find(|(_, info)| {
                        info.queue_flags
                            .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                    })
                    .map(|(index, _)| index as u32)?;

            let type_score = match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 4u64,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                vk::PhysicalDeviceType::CPU => 1,
                _ => 0,
            };
            let memory_score = unsafe { instance.get_physical_device_memory_properties(*pdevice) }
                .memory_heaps[..]
                .iter()
                .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
                .map(|heap| heap.size / (1024 * 1024))
                .max()
                .unwrap_or(0);
            Some((
                *pdevice,
                queue_family,
                (type_score << 48) | memory_score.min((1 << 48) - 1),
            ))
        })
        .max_by_key(|candidate| candidate.2)
        .map(|(pdevice, family, _)| (pdevice, family))
        .ok_or(RhiError::NoSuitableGpu)?;
    Ok(selected)
}

/// Create the Vulkan instance and, under validation, its debug messenger.
///
/// The returned flag reports whether `VK_EXT_debug_utils` was enabled at all: the device-level
/// half (object names, pass labels) rides on the same instance extension.
fn create_instance(
    entry: &Entry,
    desc: &DeviceDesc,
) -> RhiResult<(
    ash::Instance,
    Option<debug_utils::Instance>,
    vk::DebugUtilsMessengerEXT,
    bool,
)> {
    let app_name = c"kiln-rhi";

    let mut layer_names_raw: Vec<*const c_char> = Vec::new();
    let layer_name_validation = c"VK_LAYER_KHRONOS_validation";
    if desc.validation {
        layer_names_raw.push(layer_name_validation.as_ptr());
    }

    // Enabled outside validation too: it carries the object-name and command-label entry
    // points, which a release-build capture still wants.
    let debug_utils_available = unsafe { entry.enumerate_instance_extension_properties(None) }
        .map(|properties| {
            properties
                .iter()
                .any(|property| property.extension_name_as_c_str() == Ok(debug_utils::NAME))
        })
        .unwrap_or(false);

    let mut extension_names: Vec<*const c_char> = Vec::new();
    if debug_utils_available {
        extension_names.push(debug_utils::NAME.as_ptr());
    }

    extension_names.push(ash::khr::surface::NAME.as_ptr());

    #[cfg(target_os = "linux")]
    {
        extension_names.push(ash::khr::xcb_surface::NAME.as_ptr());
        extension_names.push(ash::khr::xlib_surface::NAME.as_ptr());
        extension_names.push(ash::khr::wayland_surface::NAME.as_ptr());
    }
    #[cfg(target_os = "windows")]
    {
        extension_names.push(ash::khr::win32_surface::NAME.as_ptr());
    }

    let app_info = vk::ApplicationInfo::default()
        .application_name(app_name)
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(app_name)
        .engine_version(vk::make_api_version(0, 1, 0, 0))
        .api_version(vk::make_api_version(0, 1, 4, 0));

    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layer_names_raw)
        .enabled_extension_names(&extension_names);

    let instance = unsafe {
        entry
            .create_instance(&create_info, None)
            .map_err(|e| RhiError::DeviceCreation(format!("Failed to create instance: {e}")))?
    };

    let (debug_utils_loader, debug_callback) = if desc.validation && debug_utils_available {
        let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(vulkan_debug_callback));

        let loader = debug_utils::Instance::load(entry, &instance);
        let callback = unsafe {
            loader
                .create_debug_utils_messenger(&debug_info, None)
                .map_err(|e| RhiError::DeviceCreation(format!("Debug callback: {e}")))?
        };
        (Some(loader), callback)
    } else {
        (None, vk::DebugUtilsMessengerEXT::null())
    };
    Ok((
        instance,
        debug_utils_loader,
        debug_callback,
        debug_utils_available,
    ))
}

/// Negotiate optional device extensions against `capabilities`, enable the feature set the RHI
/// depends on, and create the logical device. Bindless requires descriptor buffers and mutable
/// descriptor types; both are hard requirements, so their absence fails here.
fn create_logical_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
) -> RhiResult<ash::Device> {
    // Adapter selection already rejected anything missing these, so they are simply enabled.
    let mut device_extension_names: Vec<*const c_char> = vec![swapchain::NAME.as_ptr()];

    device_extension_names.push(descriptor_heap::NAME.as_ptr());
    device_extension_names.push(ash::khr::shader_untyped_pointers::NAME.as_ptr());
    device_extension_names.push(ash::khr::device_address_commands::NAME.as_ptr());
    device_extension_names.push(ash::khr::unified_image_layouts::NAME.as_ptr());
    device_extension_names.push(vk_mesh_shader::NAME.as_ptr());
    device_extension_names.push(vk_accel_structure::NAME.as_ptr());
    device_extension_names.push(ash::khr::deferred_host_operations::NAME.as_ptr());
    device_extension_names.push(ash::khr::ray_query::NAME.as_ptr());
    device_extension_names.push(ash::khr::ray_tracing_maintenance1::NAME.as_ptr());

    // Vulkan 1.2/1.3 features use the consolidated core feature structs.
    let mut vulkan11_features =
        vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);
    let mut vulkan12_features = vk::PhysicalDeviceVulkan12Features::default()
        .buffer_device_address(true)
        .timeline_semaphore(true)
        // Query pools are reset from the host in `reset_queries`.
        .host_query_reset(true);
    let mut vulkan13_features = vk::PhysicalDeviceVulkan13Features::default()
        .dynamic_rendering(true)
        .synchronization2(true);
    // Core in 1.4, and `VK_EXT_descriptor_heap` requires it: it carries
    // `VkPipelineCreateFlags2CreateInfo`, the only place the descriptor-heap pipeline bit lives.
    let mut vulkan14_features = vk::PhysicalDeviceVulkan14Features::default().maintenance5(true);

    let mut descriptor_heap_features =
        vk::PhysicalDeviceDescriptorHeapFeaturesEXT::default().descriptor_heap(true);
    let mut untyped_pointer_features =
        vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR::default().shader_untyped_pointers(true);
    let mut address_command_features =
        vk::PhysicalDeviceDeviceAddressCommandsFeaturesKHR::default().device_address_commands(true);
    let mut unified_layout_features =
        vk::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR::default().unified_image_layouts(true);

    // Mesh only: task/amplification shaders are not exposed (see `pipeline.rs`), and asking
    // for one would exclude mesh-only devices. The adapter filter must agree.
    let mut mesh_shader_features =
        vk::PhysicalDeviceMeshShaderFeaturesEXT::default().mesh_shader(true);
    let mut accel_structure_features =
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default().acceleration_structure(true);
    let mut ray_query_features = vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);
    let mut rt_maintenance1_features =
        vk::PhysicalDeviceRayTracingMaintenance1FeaturesKHR::default()
            .ray_tracing_maintenance1(true);

    // 64-bit integers are the only core feature this RHI needs: buffer addresses and bindless
    // handles are both 64 bits wide. Fill mode, clip distance and multi-draw were inherited from
    // an earlier design and nothing reaches them now.
    let features = vk::PhysicalDeviceFeatures {
        shader_int64: 1,
        ..Default::default()
    };

    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .features(features)
        .push(&mut vulkan11_features)
        .push(&mut vulkan12_features)
        .push(&mut vulkan13_features)
        .push(&mut vulkan14_features)
        .push(&mut descriptor_heap_features)
        .push(&mut untyped_pointer_features)
        .push(&mut address_command_features)
        .push(&mut unified_layout_features);

    // Reassign each `push` result so the feature chain remains attached.
    features2 = features2.push(&mut mesh_shader_features);
    features2 = features2
        .push(&mut accel_structure_features)
        .push(&mut ray_query_features)
        .push(&mut rt_maintenance1_features);

    let priorities = [1.0f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&priorities);

    // SAFETY: `extend` rather than `push` because `features2` already carries the feature chain
    // assembled above; every link is a live, well-formed `TaggedStructure` borrowed until the
    // `create_device` call below.
    let device_create_info = unsafe {
        vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extension_names)
            .extend(&mut features2)
    };

    let device = unsafe {
        instance
            .create_device(physical_device, &device_create_info, None)
            .map_err(|e| RhiError::DeviceCreation(format!("Failed to create device: {e}")))?
    };

    Ok(device)
}

impl VulkanDevice {
    pub fn new(desc: &DeviceDesc) -> RhiResult<Self> {
        let entry = unsafe { Entry::load() }
            .map_err(|e| RhiError::DeviceCreation(format!("Failed to load Vulkan: {e}")))?;

        let (instance, debug_utils_loader, debug_callback, debug_utils_available) =
            create_instance(&entry, desc)?;

        let (physical_device, queue_family_index) = select_physical_device(&instance)?;

        let device_props = unsafe { instance.get_physical_device_properties(physical_device) };
        let device_name = unsafe {
            std::ffi::CStr::from_ptr(device_props.device_name.as_ptr())
                .to_string_lossy()
                .to_string()
        };
        log::info!("RHI: Selected GPU: {}", device_name);

        let device = create_logical_device(&instance, physical_device, queue_family_index)?;

        let present_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // Device-level half: object names and label regions. `debug_utils_loader` is the
        // instance-level messenger and only exists under validation.
        let debug_labels =
            debug_utils_available.then(|| debug_utils::Device::load(&instance, &device));

        let surface_loader = surface::Instance::load(&entry, &instance);
        let swapchain_loader = swapchain::Device::load(&instance, &device);
        let device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let pool_create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_index);
        let command_pool = unsafe {
            device
                .create_command_pool(&pool_create_info, None)
                .map_err(|e| RhiError::DeviceCreation(format!("Command pool: {e}")))?
        };

        let pipeline_cache = unsafe {
            device
                .create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)
                .map_err(|e| RhiError::DeviceCreation(format!("Pipeline cache: {e}")))?
        };

        let cmd_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY);
        let setup_command_buffer = unsafe {
            device
                .allocate_command_buffers(&cmd_alloc_info)
                .map_err(|e| RhiError::DeviceCreation(format!("Setup cmd buffer: {e}")))?
        }[0];

        let descriptor_heap_loader = descriptor_heap::Device::load(&instance, &device);
        let address_commands_loader = device_address_commands::Device::load(&instance, &device);

        let acceleration_structure = vk_accel_structure::Device::load(&instance, &device);
        let mesh_shader_loader = vk_mesh_shader::Device::load(&instance, &device);
        let descriptor_heaps = create_descriptor_heaps(
            &instance,
            &device,
            physical_device,
            &device_memory_properties,
        )?;
        let free_texture_ids = Rc::new(RefCell::new(Vec::new()));
        let free_sampler_ids = Rc::new(RefCell::new(Vec::new()));
        let mut completion_type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let completion_info =
            vk::SemaphoreCreateInfo::default().push(&mut completion_type_info);
        let completion_semaphore = unsafe {
            device
                .create_semaphore(&completion_info, None)
                .map_err(|e| RhiError::DeviceCreation(format!("Completion timeline: {e}")))?
        };

        // `bufferImageGranularity` is the separation buffers and images need when they share a
        // `VkDeviceMemory`; the pool pads every suballocation to it.
        let buffer_pool: SharedBufferPool = Rc::new(RefCell::new(VulkanBufferPool::new(
            device_props.limits.buffer_image_granularity,
        )));

        let queue = Queue {
            inner: QueueInner::Vulkan(Box::new(VulkanQueue {
                queue: present_queue,
                device: device.clone(),
                swapchain_loader: swapchain::Device::load(&instance, &device),
                command_pool,
                buffer_pool: buffer_pool.clone(),
                pending_commands: RefCell::new(VecDeque::new()),
                available_commands: RefCell::new(Vec::new()),
                completion_semaphore,
                next_completion_value: RefCell::new(0),
                frame_completion_values: RefCell::new([0; MAX_FRAMES_IN_FLIGHT]),
                frame_fence_armed: RefCell::new([false; MAX_FRAMES_IN_FLIGHT]),
                retired_resources: RefCell::new(VecDeque::new()),
                free_texture_ids: free_texture_ids.clone(),
                free_sampler_ids: free_sampler_ids.clone(),
            })),
        };

        Ok(Self {
            entry,
            instance,
            device,
            physical_device,

            queue,
            present_queue,
            command_pool,
            pipeline_cache,
            buffer_pool,
            device_memory_properties,
            timestamp_period: device_props.limits.timestamp_period,
            surface_loader,
            swapchain_loader,
            descriptor_heap_loader,
            address_commands_loader,
            debug_utils_loader,
            debug_labels,
            debug_callback,
            descriptor_heaps,
            buffer_requirements_probe: RefCell::new(Vec::new()),
            textures: Rc::new(RefCell::new(Vec::new())),
            next_texture_id: RefCell::new(0),
            free_texture_ids,
            allocations: Rc::new(RefCell::new(BTreeMap::new())),
            mapped_allocations: Rc::new(RefCell::new(BTreeMap::new())),
            next_sampler_id: RefCell::new(0),
            free_sampler_ids,
            timeline_semaphores: RefCell::new(Vec::new()),
            query_pools: RefCell::new(Vec::new()),
            setup_command_buffer,
            setup_recording: Cell::new(false),
            mesh_shader_loader,
            acceleration_structure,
        })
    }

    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    pub fn wait_idle(&self) {
        backend_expect!(&self.queue.inner, QueueInner::Vulkan).wait_idle();
    }

    pub fn wait_for_frame(&self, frame_index: usize) {
        backend_expect!(&self.queue.inner, QueueInner::Vulkan).wait_for_frame(frame_index);
    }

    pub fn create_surface(&self, desc: &SurfaceDesc) -> RhiResult<Surface> {
        let factory =
            ash_window::SurfaceFactory::new(&self.entry, &self.instance, desc.display_handle)
                .map_err(|e| RhiError::SurfaceCreation(e.to_string()))?;
        let surface = unsafe {
            factory
                .create_surface(desc.window_handle, None)
                .map_err(|e| RhiError::SurfaceCreation(e.to_string()))?
        };

        Ok(Surface {
            inner: SurfaceInner::Vulkan(VulkanSurface {
                surface,
                surface_loader: self.surface_loader.clone(),
            }),
            _owner: None,
        })
    }

    /// Name a Vulkan object for captures. No-op without `VK_EXT_debug_utils`.
    pub(crate) fn set_object_name<H: vk::Handle>(&self, handle: H, name: &str) {
        let Some(loader) = self.debug_labels.as_ref() else {
            return;
        };
        let name = std::ffi::CString::new(name).expect("debug label contains a NUL");
        let info = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(handle)
            .object_name(&name);
        unsafe { loader.set_debug_utils_object_name(&info) }.expect("set_debug_utils_object_name");
    }

    /// Memory requirements for a range of `size` inside a pooled block.
    ///
    /// Every allocation shares the block's usage flags, so alignment and the permitted memory
    /// types are properties of that usage rather than of any one allocation. They are probed once
    /// with a throwaway buffer and cached; afterwards this is pure arithmetic, which keeps scene
    /// loads from paying a create/destroy pair per allocation.
    pub(crate) fn buffer_requirements(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> RhiResult<vk::MemoryRequirements> {
        let cached = self
            .buffer_requirements_probe
            .borrow()
            .iter()
            .find(|(u, _, _)| *u == usage)
            .map(|&(_, a, b)| (a, b));
        let (alignment, memory_type_bits) = match cached {
            Some(cached) => cached,
            None => {
                let info = vk::BufferCreateInfo::default()
                    .size(1)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE);
                let probe = unsafe {
                    self.device
                        .create_buffer(&info, None)
                        .map_err(|e| RhiError::BufferCreation(e.to_string()))?
                };
                let requirements = unsafe { self.device.get_buffer_memory_requirements(probe) };
                unsafe { self.device.destroy_buffer(probe, None) };
                let probed = (requirements.alignment, requirements.memory_type_bits);
                self.buffer_requirements_probe
                    .borrow_mut()
                    .push((usage, probed.0, probed.1));
                probed
            }
        };
        Ok(vk::MemoryRequirements {
            size: size.max(1),
            alignment,
            memory_type_bits,
        })
    }

    pub fn create_allocation(&self, desc: &AllocationDesc) -> RhiResult<Allocation> {
        // Allocations no longer own a `VkBuffer`: the pool's block does, and this is a range
        // inside it. Requirements come from the block's usage, so they are the same for every
        // allocation and are queried once.
        let mem_requirements =
            self.buffer_requirements(desc.size, super::memory::BLOCK_BUFFER_USAGE)?;

        let mem_flags = buffer_memory_flags(desc.memory);

        let preferred_flags = match desc.memory {
            MemoryType::Upload => vk::MemoryPropertyFlags::DEVICE_LOCAL,
            MemoryType::GpuOnly | MemoryType::Readback => vk::MemoryPropertyFlags::empty(),
        };
        // The preferred type is tried first, then the bare requirement. `Upload` prefers a
        // host-visible *device-local* heap, which is the resizable-BAR window: ideal for the small
        // hot writes it mostly carries, but often only 256 MiB. Bulk staging fills it, so a failed
        // allocation has to retry in plain host memory rather than give up.
        let mut candidates = [
            find_memorytype_index(
                &mem_requirements,
                &self.device_memory_properties,
                mem_flags | preferred_flags,
            ),
            find_memorytype_index(&mem_requirements, &self.device_memory_properties, mem_flags),
        ];
        if candidates[0] == candidates[1] {
            candidates[1] = None;
        }
        if candidates.iter().all(Option::is_none) {
            return Err(RhiError::AllocationFailed("No suitable memory type".into()));
        }

        let mut attempt = Err(RhiError::AllocationFailed("No suitable memory type".into()));
        for candidate in candidates.into_iter().flatten() {
            // Follows the memory type's real properties, not the requested `MemoryType`: on UMA one
            // type serves both, and a block created by a `GpuOnly` buffer must still be mappable
            // for an `Upload` buffer landing in it later.
            let host_visible = self.device_memory_properties.memory_types[candidate as usize]
                .property_flags
                .contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
            let mut pool = self.buffer_pool.borrow_mut();
            attempt = pool.allocate(
                &self.device,
                &mem_requirements,
                candidate,
                host_visible,
                super::memory::BLOCK_BUFFER_USAGE,
            );
            if attempt.is_ok() {
                break;
            }
        }
        let suballocation = attempt?;
        let gpu_addr = suballocation.address;

        // `GpuOnly` promises no CPU pointer even when it happens to land in host-visible memory.
        let mapped_ptr = match desc.memory {
            MemoryType::Upload | MemoryType::Readback => suballocation.mapped_ptr,
            MemoryType::GpuOnly => None,
        };

        let vk_buffer = VulkanBuffer {
            memory: suballocation.memory,
            size: desc.size,
            mapped_ptr,
            gpu_address: GpuPtr::from_addr(gpu_addr),
            block_index: suballocation.block_index,
            block_offset: suballocation.offset,
            block_range: suballocation.range,
        };

        {
            let mut allocations = self.allocations.borrow_mut();
            allocations.insert(
                vk_buffer.gpu_address.address,
                BufferAllocation {
                    base: vk_buffer.gpu_address,
                    size: vk_buffer.size,
                    memory: vk_buffer.memory,
                    memory_offset: vk_buffer.block_offset,
                },
            );
        }
        if let Some(mapped_ptr) = vk_buffer.mapped_ptr {
            self.mapped_allocations.borrow_mut().insert(
                mapped_ptr as usize,
                MappedAllocation {
                    gpu_base: vk_buffer.gpu_address,
                    size: vk_buffer.size,
                },
            );
        }

        Ok(Allocation {
            inner: AllocationInner::Vulkan(vk_buffer),
            _owner: None,
            offset: 0,
            size: desc.size,
        })
    }

    pub fn host_to_device_pointer(&self, cpu_ptr: *const u8) -> Option<GpuPtr<u8>> {
        if cpu_ptr.is_null() {
            return None;
        }
        let ptr = cpu_ptr as usize;
        resolve_mapped_pointer(&self.mapped_allocations.borrow(), ptr)
    }

    pub(crate) fn recycle_sampler_id(&self, id: SamplerId) {
        self.free_sampler_ids.borrow_mut().push(id);
    }

    pub fn create_shader_module(&self, desc: &ShaderModuleDesc) -> RhiResult<ShaderModule> {
        let code: Vec<u32> = desc
            .code
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let shader_info = vk::ShaderModuleCreateInfo::default().code(&code);

        let module = unsafe {
            self.device
                .create_shader_module(&shader_info, None)
                .map_err(|e| RhiError::ShaderCompilation(e.to_string()))?
        };

        let entry_point = CString::new(desc.entry_point)
            .map_err(|e| RhiError::ShaderCompilation(e.to_string()))?;

        Ok(ShaderModule {
            inner: ShaderModuleInner::Vulkan(Box::new(VulkanShaderModule::new(
                self.device.clone(),
                module,
                entry_point,
            ))),
            stage: desc.stage,
            _owner: None,
        })
    }

    pub fn create_graphics_pso(
        &self,
        desc: &GraphicsPsoDesc,
        vert_module: &VulkanShaderModule,
        frag_module: &VulkanShaderModule,
    ) -> RhiResult<GraphicsPso> {
        let pso_desc = VulkanGraphicsPsoDesc {
            vert_module: vert_module.module,
            frag_module: frag_module.module,
            vert_entry: vert_module.entry_point.clone(),
            frag_entry: frag_module.entry_point.clone(),
            topology: desc.topology,
            color_targets: desc.color_targets.clone(),
            depth_format: desc.depth_format.map(format_to_vk),
            sample_count: desc.sample_count,
            alpha_to_coverage: desc.alpha_to_coverage,
            cull: desc.cull,
            depth: desc.depth,
        };

        let mut vk_pso = VulkanGraphicsPso {
            pipeline: vk::Pipeline::null(),
            device: self.device.clone(),
            pipeline_cache: self.pipeline_cache,
            desc: pso_desc,
        };
        let blend = desc.blendstate.as_ref().cloned().unwrap_or_default();
        vk_pso.pipeline = vk_pso.create_pipeline(&blend)?;
        if let Some(label) = desc.label.as_deref() {
            self.set_object_name(vk_pso.pipeline, label);
        }

        Ok(GraphicsPso {
            inner: GraphicsPsoInner::Vulkan(std::rc::Rc::new(vk_pso)),
            _owner: None,
        })
    }

    pub fn create_compute_pso(
        &self,
        desc: &ComputePsoDesc,
        shader: &VulkanShaderModule,
    ) -> RhiResult<ComputePso> {
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader.module)
            .name(&shader.entry_point);


        let mut flags2 = vk::PipelineCreateFlags2CreateInfo::default()
            .flags(vk::PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT);
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(vk::PipelineLayout::null())
            .push(&mut flags2);

        let pipelines = unsafe {
            self.device
                .create_compute_pipelines(self.pipeline_cache, &[pipeline_info], None)
                .map_err(|e| RhiError::PipelineCreation(format!("{e:?}")))?
        };

        if let Some(label) = desc.label.as_deref() {
            self.set_object_name(pipelines[0], label);
        }

        Ok(ComputePso {
            inner: ComputePsoInner::Vulkan(std::rc::Rc::new(VulkanComputePso {
                pipeline: pipelines[0],
                device: self.device.clone(),
            })),
            _owner: None,
        })
    }

    pub fn create_meshlet_pso(
        &self,
        desc: &MeshletPsoDesc,
        mesh_module: &VulkanShaderModule,
        frag_module: &VulkanShaderModule,
    ) -> RhiResult<MeshletPso> {
        let pso_desc = VulkanMeshletPsoDesc {
            mesh_module: mesh_module.module,
            frag_module: frag_module.module,
            mesh_entry: mesh_module.entry_point.clone(),
            frag_entry: frag_module.entry_point.clone(),
            color_targets: desc.color_targets.clone(),
            depth_format: desc.depth_format.map(format_to_vk),
            sample_count: desc.sample_count,
            alpha_to_coverage: desc.alpha_to_coverage,
            cull: desc.cull,
            depth: desc.depth,
        };


        let mut vk_pso = VulkanMeshletPso {
            pipeline: vk::Pipeline::null(),
            device: self.device.clone(),
            pipeline_cache: self.pipeline_cache,
            desc: pso_desc,
        };
        let blend = desc.blendstate.as_ref().cloned().unwrap_or_default();
        vk_pso.pipeline = vk_pso.create_pipeline(&blend)?;
        if let Some(label) = desc.label.as_deref() {
            self.set_object_name(vk_pso.pipeline, label);
        }

        Ok(MeshletPso {
            inner: MeshletPsoInner::Vulkan(std::rc::Rc::new(vk_pso)),
            _owner: None,
        })
    }

    /// `minAccelerationStructureScratchOffsetAlignment` — the required alignment for the
    /// `scratch_data` address passed to `vkCmdBuildAccelerationStructuresKHR`.
    pub(crate) fn accel_scratch_alignment(&self) -> u64 {
        let mut accel_props = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push(&mut accel_props);
        unsafe {
            self.instance
                .get_physical_device_properties2(self.physical_device, &mut props2);
        }
        accel_props
            .min_acceleration_structure_scratch_offset_alignment
            .max(1) as u64
    }

    fn acquire_command_buffer(&self) -> RhiResult<vk::CommandBuffer> {
        backend_expect!(&self.queue.inner, QueueInner::Vulkan).acquire_command_buffer()
    }

    pub(crate) fn recycle_command_buffer(&self, command_buffer: vk::CommandBuffer) {
        backend_expect!(&self.queue.inner, QueueInner::Vulkan)
            .recycle_command_buffer(command_buffer)
    }

    pub fn create_command_buffer(&self) -> RhiResult<CommandBuffer> {
        // Texture creation records initial-layout transitions into one reusable setup command
        // buffer. Flush the batch once before user work starts instead of queue-idling once per
        // texture.
        self.flush_setup_barriers()?;
        let cmd = self.acquire_command_buffer()?;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin_info)
                .map_err(|e| RhiError::CommandBuffer(e.to_string()))?;
        }

        // Both heaps are bound once here and stay bound: they are the only ones the device owns.
        self.bind_descriptor_heaps(cmd);
        let descriptor_heap_loader = self.descriptor_heap_loader.clone();
        let address_commands = self.address_commands_loader.clone();
        let mesh_shader = self.mesh_shader_loader.clone();
        let accel_loader_cmd = self.acceleration_structure.clone();

        Ok(CommandBuffer {
            inner: CommandBufferInner::Vulkan(Box::new(VulkanCommandBuffer {
                command_buffer: cmd,
                device: self.device.clone(),
                swapchain_image_views: Rc::from([]),
                swapchain_images: Rc::from([]),
                depth_image_view: vk::ImageView::null(),
                descriptor_heap_loader,
                address_commands,
                debug_labels: self.debug_labels.clone(),
                in_labelled_pass: false,
                pending_split_barrier: None,
                textures: self.textures.clone(),
                mesh_shader,
                acceleration_structure: accel_loader_cmd,
                rendered_swapchain_images: SmallVec::new(),
                retained_pipelines: SmallVec::new(),
                ended: false,
            })),
            _owner: None,
        })
    }

    /// Bind the resource and sampler heaps for the lifetime of `cmd`.
    fn bind_descriptor_heaps(&self, cmd: vk::CommandBuffer) {
        let heaps = &self.descriptor_heaps;
        unsafe {
            self.descriptor_heap_loader
                .cmd_bind_resource_heap(cmd, &heaps.resource.bind_info());
            self.descriptor_heap_loader
                .cmd_bind_sampler_heap(cmd, &heaps.sampler.bind_info());
        }
    }

    /// Create a command buffer pre-configured with swapchain image views for rendering.
    pub fn create_command_buffer_for_swapchain(
        &self,
        swapchain: &Swapchain,
        frame_index: usize,
    ) -> RhiResult<CommandBuffer> {
        let sc = backend_expect!(&swapchain.inner, SwapchainInner::Vulkan);
        if frame_index >= sc.in_flight_fences.len() {
            return Err(RhiError::CommandBuffer("invalid Vulkan frame index".into()));
        }
        self.flush_setup_barriers()?;

        let cmd = self.acquire_command_buffer()?;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin_info)
                .map_err(|e| RhiError::CommandBuffer(e.to_string()))?;
        }

        // Both heaps are bound once here and stay bound: they are the only ones the device owns.
        self.bind_descriptor_heaps(cmd);
        let descriptor_heap_loader = self.descriptor_heap_loader.clone();
        let address_commands = self.address_commands_loader.clone();
        let mesh_shader = self.mesh_shader_loader.clone();
        let accel_loader_cmd = self.acceleration_structure.clone();

        Ok(CommandBuffer {
            inner: CommandBufferInner::Vulkan(Box::new(VulkanCommandBuffer {
                command_buffer: cmd,
                device: self.device.clone(),
                swapchain_image_views: sc.image_views.clone(),
                swapchain_images: sc.images.clone(),
                depth_image_view: sc.depth_image_view,
                descriptor_heap_loader,
                address_commands,
                debug_labels: self.debug_labels.clone(),
                in_labelled_pass: false,
                pending_split_barrier: None,
                textures: self.textures.clone(),
                mesh_shader,
                acceleration_structure: accel_loader_cmd,
                rendered_swapchain_images: SmallVec::new(),
                retained_pipelines: SmallVec::new(),
                ended: false,
            })),
            _owner: None,
        })
    }

    pub fn create_timeline_semaphore(&self, initial_value: u64) -> RhiResult<TimelineSemaphore> {
        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(initial_value);

        let semaphore_info = vk::SemaphoreCreateInfo::default().push(&mut type_info);

        let semaphore = unsafe {
            self.device
                .create_semaphore(&semaphore_info, None)
                .map_err(|e| RhiError::SyncError(e.to_string()))?
        };
        self.timeline_semaphores.borrow_mut().push(semaphore);

        Ok(TimelineSemaphore {
            inner: TimelineSemaphoreInner::Vulkan(Box::new(VulkanTimelineSemaphore {
                semaphore,
                device: self.device.clone(),
            })),
            _owner: None,
        })
    }

    pub fn destroy_allocation(&self, buffer: Allocation) {
        match buffer.inner {
            AllocationInner::Vulkan(b) => {
                {
                    let mut allocations = self.allocations.borrow_mut();
                    allocations.remove(&b.gpu_address.address);
                }
                if let Some(mapped_ptr) = b.mapped_ptr {
                    self.mapped_allocations
                        .borrow_mut()
                        .remove(&(mapped_ptr as usize));
                }
                backend_expect!(&self.queue.inner, QueueInner::Vulkan)
                    .release_resource(VulkanRetiredResource::Buffer(b));
            }
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    pub fn create_query_pool(&self, count: u32) -> RhiResult<QueryPool> {
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(count);
        let pool = unsafe { self.device.create_query_pool(&info, None) }
            .map_err(|e| RhiError::Backend(format!("create_query_pool: {e}")))?;
        // Vulkan requires a query to be reset before its first use.
        unsafe { self.device.reset_query_pool(pool, 0, count) };
        self.query_pools.borrow_mut().push(pool);
        Ok(QueryPool {
            inner: QueryPoolInner::Vulkan(VulkanQueryPool { pool }),
            count,
            _owner: None,
        })
    }

    pub fn destroy_query_pool(&self, pool: QueryPool) {
        let p = match pool.inner {
            QueryPoolInner::Vulkan(p) => p,
            #[allow(unreachable_patterns)]
            _ => return,
        };
        self.query_pools
            .borrow_mut()
            .retain(|&entry| entry != p.pool);
        unsafe { self.device.destroy_query_pool(p.pool, None) };
    }

    pub fn timestamp_period_ns(&self) -> f64 {
        self.timestamp_period as f64
    }

    pub fn read_timestamps(&self, pool: &QueryPool) -> RhiResult<Vec<u64>> {
        let vk_pool = backend_expect!(&pool.inner, QueryPoolInner::Vulkan).pool;
        let mut data = vec![0u64; pool.count as usize];
        // The frame fence has already completed; unwritten slots remain zero.
        let result = unsafe {
            self.device
                .get_query_pool_results(vk_pool, 0, &mut data, vk::QueryResultFlags::TYPE_64)
        };
        match result {
            Ok(()) => Ok(data),
            Err(vk::Result::NOT_READY) => Ok(vec![0u64; pool.count as usize]),
            Err(e) => Err(RhiError::Backend(format!("get_query_pool_results: {e}"))),
        }
    }

    /// Write one image descriptor into the resource heap at slot `id`.
    ///
    /// `VK_EXT_descriptor_heap` describes the view inline, so this takes the
    /// `ImageViewCreateInfo` rather than a `VkImageView`; sampled and storage images share the
    /// slot and differ only in descriptor type and layout.
    pub(crate) fn write_image_descriptor(
        &self,
        id: TextureId,
        view_info: &vk::ImageViewCreateInfo<'_>,
        layout: vk::ImageLayout,
        storage: bool,
    ) -> RhiResult<()> {
        let heap = &self.descriptor_heaps.resource;
        let slot = heap.slot(id.0);
        let ty = if storage {
            vk::DescriptorType::STORAGE_IMAGE
        } else {
            vk::DescriptorType::SAMPLED_IMAGE
        };
        let image = vk::ImageDescriptorInfoEXT::default()
            .view(view_info)
            .layout(layout);
        let info = vk::ResourceDescriptorInfoEXT::default()
            .ty(ty)
            .data(vk::ResourceDescriptorDataEXT { p_image: &image });

        unsafe {
            let dst = std::slice::from_raw_parts_mut(heap.mapped_ptr.add(slot.start), slot.len());
            self.descriptor_heap_loader
                .write_resource_descriptors(&[info], &[vk::HostAddressRangeEXT::default().address(dst)])
                .map_err(|e| RhiError::Backend(format!("write image descriptor: {e}")))
        }
    }

    /// Move a freshly created image out of `UNDEFINED` into [`IMAGE_LAYOUT`], the only layout it
    /// will ever be in. This is the sole remaining image transition outside presentation.
    pub(crate) fn initialize_image_layout(
        &self,
        image: vk::Image,
        aspect: vk::ImageAspectFlags,
        mip_levels: u32,
        layer_count: u32,
    ) -> RhiResult<()> {
        let barrier = vk::ImageMemoryBarrier2::default()
            .image(image)
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(IMAGE_LAYOUT)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(aspect)
                    .level_count(mip_levels)
                    .layer_count(layer_count),
            );
        self.submit_setup_barrier(barrier)
    }

    /// Append an image barrier to the reusable setup command buffer. The batch is submitted when
    /// the next user command buffer is created, so loading a scene with many textures incurs one
    /// setup submission rather than one queue idle per texture.
    fn submit_setup_barrier(&self, barrier: vk::ImageMemoryBarrier2<'_>) -> RhiResult<()> {
        unsafe {
            if !self.setup_recording.get() {
                self.device
                    .reset_command_buffer(
                        self.setup_command_buffer,
                        vk::CommandBufferResetFlags::empty(),
                    )
                    .map_err(|e| RhiError::CommandBuffer(e.to_string()))?;
                let begin_info = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                self.device
                    .begin_command_buffer(self.setup_command_buffer, &begin_info)
                    .map_err(|e| RhiError::CommandBuffer(e.to_string()))?;
                self.setup_recording.set(true);
            }
            let dependency =
                vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));
            self.device
                .cmd_pipeline_barrier2(self.setup_command_buffer, &dependency);
        }
        Ok(())
    }

    /// Submit any batched setup barriers.
    ///
    /// Must be called before destroying an image that one of them might reference: the setup
    /// buffer stays in the recording state between batches, and destroying a referenced image
    /// invalidates the whole buffer, not just that barrier.
    pub(crate) fn flush_setup_barriers(&self) -> RhiResult<()> {
        if !self.setup_recording.replace(false) {
            return Ok(());
        }
        unsafe {
            self.device
                .end_command_buffer(self.setup_command_buffer)
                .map_err(|e| RhiError::CommandBuffer(e.to_string()))?;
            let submit_info = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.setup_command_buffer));
            self.device
                .queue_submit(self.present_queue, &[submit_info], vk::Fence::null())
                .map_err(|e| RhiError::QueueSubmit(e.to_string()))?;
            self.device
                .queue_wait_idle(self.present_queue)
                .map_err(|e| RhiError::QueueSubmit(e.to_string()))?;
        }
        Ok(())
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        let q = backend_expect!(&self.queue.inner, QueueInner::Vulkan);
        q.wait_idle();
        unsafe {
            self.device.destroy_semaphore(q.completion_semaphore, None);
            for semaphore in self.timeline_semaphores.get_mut().drain(..) {
                self.device.destroy_semaphore(semaphore, None);
            }
            for pool in self.query_pools.get_mut().drain(..) {
                self.device.destroy_query_pool(pool, None);
            }

            for t in self.textures.borrow_mut().drain(..).flatten() {
                self.device.destroy_image_view(t.image_view, None);
                if !t.is_view {
                    self.device.destroy_image(t.image, None);
                }
            }

            for heap in [&self.descriptor_heaps.resource, &self.descriptor_heaps.sampler] {
                self.device.unmap_memory(heap.memory);
                self.device.destroy_buffer(heap.buffer, None);
                self.device.free_memory(heap.memory, None);
            }

            self.device
                .destroy_pipeline_cache(self.pipeline_cache, None);
            self.device.destroy_command_pool(self.command_pool, None);

            // Buffers the application never destroyed are still bound into pool blocks, and
            // freeing block memory underneath a live buffer is invalid usage.
            // Allocations own no Vulkan object; the pool's blocks are freed just below.
            self.allocations.borrow_mut().clear();
            self.buffer_pool.borrow_mut().destroy_all(&self.device);
            self.device.destroy_device(None);

            if let Some(ref debug_loader) = self.debug_utils_loader {
                debug_loader.destroy_debug_utils_messenger(self.debug_callback, None);
            }

            self.instance.destroy_instance(None);
        }
    }
}

/// Allocate the resource and sampler heaps. Both are plain mapped allocations; under
/// `VK_EXT_descriptor_heap` there is no set layout, no binding list and no mutable-type dance.
fn create_descriptor_heaps(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
) -> RhiResult<DescriptorHeaps> {
    let mut props = vk::PhysicalDeviceDescriptorHeapPropertiesEXT::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push(&mut props);
    unsafe {
        instance.get_physical_device_properties2(physical_device, &mut props2);
    }

    // The resource heap holds only images: buffers reach the shader as device addresses, so the
    // stride is `imageDescriptorSize` and never the image/buffer maximum.
    let resource = create_descriptor_heap(
        device,
        mem_props,
        HeapLayout {
            slots: MAX_BINDLESS_TEXTURES,
            descriptor_size: props.image_descriptor_size,
            alignment: props.resource_heap_alignment,
            reserved_size: props.min_resource_heap_reserved_range,
            max_size: props.max_resource_heap_size,
        },
        "resource descriptor heap",
    )?;
    let sampler = create_descriptor_heap(
        device,
        mem_props,
        HeapLayout {
            slots: MAX_BINDLESS_SAMPLERS,
            descriptor_size: props.sampler_descriptor_size,
            alignment: props.sampler_heap_alignment,
            reserved_size: props.min_sampler_heap_reserved_range,
            max_size: props.max_sampler_heap_size,
        },
        "sampler descriptor heap",
    )?;

    Ok(DescriptorHeaps { resource, sampler })
}

/// Sizing inputs for one heap, all sourced from `VkPhysicalDeviceDescriptorHeapPropertiesEXT`.
struct HeapLayout {
    slots: u32,
    descriptor_size: u64,
    alignment: u64,
    reserved_size: u64,
    max_size: u64,
}

fn create_descriptor_heap(
    device: &Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    layout: HeapLayout,
    what: &str,
) -> RhiResult<DescriptorHeap> {
    let align = layout.alignment.max(1);
    let descriptors = (layout.slots as u64)
        .checked_mul(layout.descriptor_size)
        .ok_or_else(|| RhiError::AllocationFailed(format!("{what} size overflows")))?;
    // Park the driver's reserved range past the last descriptor so slot N stays at N * stride.
    let reserved_offset = align_up(descriptors, align);
    let size = align_up(reserved_offset + layout.reserved_size, align);

    if size > layout.max_size {
        return Err(RhiError::Unsupported(format!(
            "{what} needs {size} bytes but the device caps it at {}",
            layout.max_size
        )));
    }

    let (buffer, memory, gpu_address) = super::memory::allocate_bound_buffer(
        device,
        mem_props,
        size,
        vk::BufferUsageFlags::DESCRIPTOR_HEAP_EXT,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        what,
    )?;

    if gpu_address % align != 0 {
        unsafe {
            device.destroy_buffer(buffer, None);
            device.free_memory(memory, None);
        }
        return Err(RhiError::AllocationFailed(format!(
            "{what} address {gpu_address:#x} is not {align}-byte aligned"
        )));
    }

    let mapped_ptr = match unsafe { device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty()) }
    {
        Ok(ptr) => ptr as *mut u8,
        Err(error) => {
            unsafe {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            }
            return Err(RhiError::AllocationFailed(error.to_string()));
        }
    };

    Ok(DescriptorHeap {
        buffer,
        memory,
        mapped_ptr,
        size,
        gpu_address: GpuPtr::from_addr(gpu_address),
        descriptor_size: layout.descriptor_size,
        reserved_offset,
        reserved_size: layout.reserved_size,
    })
}

fn align_up(value: u64, align: u64) -> u64 {
    (value + align - 1) & !(align - 1)
}

pub(crate) fn address_mode_to_vk(mode: AddressMode) -> vk::SamplerAddressMode {
    match mode {
        AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
        AddressMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        AddressMode::ClampToBorder => vk::SamplerAddressMode::CLAMP_TO_BORDER,
    }
}

pub(crate) fn compare_op_to_vk(op: CompareOp) -> vk::CompareOp {
    match op {
        CompareOp::Never => vk::CompareOp::NEVER,
        CompareOp::Less => vk::CompareOp::LESS,
        CompareOp::Equal => vk::CompareOp::EQUAL,
        CompareOp::LessOrEqual => vk::CompareOp::LESS_OR_EQUAL,
        CompareOp::Greater => vk::CompareOp::GREATER,
        CompareOp::NotEqual => vk::CompareOp::NOT_EQUAL,
        CompareOp::GreaterOrEqual => vk::CompareOp::GREATER_OR_EQUAL,
        CompareOp::Always => vk::CompareOp::ALWAYS,
    }
}

fn buffer_memory_flags(memory: MemoryType) -> vk::MemoryPropertyFlags {
    match memory {
        MemoryType::GpuOnly => vk::MemoryPropertyFlags::DEVICE_LOCAL,
        MemoryType::Upload => {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        }
        MemoryType::Readback => {
            vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_CACHED
                | vk::MemoryPropertyFlags::HOST_COHERENT
        }
    }
}

/// Depth formats that carry a stencil aspect alongside depth.
pub(crate) fn format_has_stencil(format: Format) -> bool {
    matches!(format, Format::D24UnormS8Uint | Format::D32FloatS8Uint)
}

pub(crate) fn is_depth_format(format: Format) -> bool {
    matches!(
        format,
        Format::D16Unorm | Format::D32Float | Format::D24UnormS8Uint | Format::D32FloatS8Uint
    )
}

/// Every image lives in `GENERAL` for its whole life.
///
/// `VK_KHR_unified_image_layouts` guarantees `GENERAL` is as efficient as the specialized
/// layouts, so there is nothing to choose, nothing to track per resource, and no transition to
/// record except the one out of `UNDEFINED` at creation.
pub(crate) const IMAGE_LAYOUT: vk::ImageLayout = vk::ImageLayout::GENERAL;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapped_pointer_lookup_uses_predecessor_and_respects_end() {
        let mut allocations = BTreeMap::new();
        allocations.insert(
            0x1000,
            MappedAllocation {
                gpu_base: GpuPtr::from_addr(0x8000),
                size: 0x20,
            },
        );
        allocations.insert(
            0x2000,
            MappedAllocation {
                gpu_base: GpuPtr::from_addr(0x9000),
                size: 0x10,
            },
        );

        assert_eq!(
            resolve_mapped_pointer(&allocations, 0x100f),
            Some(GpuPtr::from_addr(0x800f))
        );
        assert_eq!(
            resolve_mapped_pointer(&allocations, 0x200f),
            Some(GpuPtr::from_addr(0x900f))
        );
        assert_eq!(resolve_mapped_pointer(&allocations, 0x1020), None);
        assert_eq!(resolve_mapped_pointer(&allocations, 0x1fff), None);
    }

    #[test]
    fn readback_memory_is_host_coherent() {
        let flags = buffer_memory_flags(MemoryType::Readback);
        assert!(flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE));
        assert!(flags.contains(vk::MemoryPropertyFlags::HOST_CACHED));
        assert!(flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT));
    }
}
