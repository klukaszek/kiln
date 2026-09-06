use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, VecDeque};
use std::ffi::{CStr, CString, c_char};
use std::rc::Rc;

use ash::{
    Device, Entry, Instance,
    ext::{debug_utils, descriptor_buffer, mesh_shader as vk_mesh_shader},
    khr::{acceleration_structure as vk_accel_structure, surface, swapchain},
    vk,
};
use smallvec::SmallVec;

use crate::command::{CommandBuffer, CommandBufferInner};
use crate::device::{BindlessMode, DeviceDesc};
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
use crate::texture::TextureDesc;
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
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    /// Offset within the pooled block; images placed here bind at this plus their own offset.
    pub memory_offset: u64,
    pub memory_type_index: u32,
}

/// Vulkan backend device.

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

pub(crate) struct DescriptorBufferHeap {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub mapped_ptr: *mut u8,
    pub size: u64,
    pub gpu_address: GpuPtr<u8>,
    pub layout: vk::DescriptorSetLayout,
    pub sampled_image_offset: u64,
    pub sampler_offset: u64,
    pub storage_image_offset: u64,
    /// Sampled and storage images share this mutable binding. Each slot uses the larger
    /// descriptor size.
    pub image_descriptor_stride: u64,
    pub sampled_image_descriptor_size: u64,
    pub storage_image_descriptor_size: u64,
    pub sampler_stride: u64,
}

/// Buffer allocations keyed by GPU base address, enabling O(log n) address->buffer
/// resolution instead of a linear scan on every indirect/copy/index command.
pub(crate) type SharedAllocations = Rc<RefCell<BTreeMap<u64, BufferAllocation>>>;
type SharedMappedAllocations = Rc<RefCell<BTreeMap<usize, MappedAllocation>>>;
pub(crate) type SharedTextures = Rc<RefCell<Vec<Option<VulkanTexture>>>>;
pub(crate) type SharedTextureFreeIds = Rc<RefCell<Vec<TextureId>>>;
pub(crate) type SharedSamplerFreeIds = Rc<RefCell<Vec<SamplerId>>>;

#[derive(Clone, Copy)]
struct VulkanCapabilities {
    mesh_shader: bool,
    acceleration_structure: bool,
    ray_query: bool,
    ray_tracing_maintenance1: bool,
    ray_tracing_pipeline: bool,
}

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
    pub(crate) bindless_mode: BindlessMode,
    /// Nanoseconds per timestamp tick (`VkPhysicalDeviceLimits::timestampPeriod`).
    pub(crate) timestamp_period: f32,

    pub(crate) surface_loader: surface::Instance,
    pub(crate) swapchain_loader: swapchain::Device,
    pub(crate) descriptor_buffer_loader: descriptor_buffer::Device,

    pub(crate) debug_utils_loader: Option<debug_utils::Instance>,
    pub(crate) debug_callback: vk::DebugUtilsMessengerEXT,
    /// Device-level `VK_EXT_debug_utils`: object names and pass label regions.
    pub(crate) debug_labels: Option<debug_utils::Device>,

    pub(crate) texture_descriptor_set_layout: vk::DescriptorSetLayout,
    pub(crate) descriptor_buffer_heap: DescriptorBufferHeap,
    pub(crate) textures: SharedTextures,
    pub(crate) texture_view_flags: RefCell<Vec<bool>>,
    pub(crate) next_texture_id: RefCell<u32>,
    pub(crate) free_texture_ids: SharedTextureFreeIds,
    pub(crate) allocations: SharedAllocations,
    pub(crate) mapped_allocations: SharedMappedAllocations,

    pub(crate) samplers: RefCell<Vec<Option<vk::Sampler>>>,
    pub(crate) next_sampler_id: RefCell<u32>,
    pub(crate) free_sampler_ids: SharedSamplerFreeIds,

    /// User timeline semaphores stay alive until device teardown. This permits callers to use a
    /// temporary `SubmitDesc` without destroying a semaphore still referenced by queued work.
    pub(crate) timeline_semaphores: RefCell<Vec<vk::Semaphore>>,
    pub(crate) query_pools: RefCell<Vec<vk::QueryPool>>,

    pub(crate) setup_command_buffer: vk::CommandBuffer,
    /// Whether the batched setup command buffer is currently recording initial image layouts.
    pub(crate) setup_recording: Cell<bool>,

    /// True when `VK_EXT_mesh_shader` was enabled at device creation.
    pub(crate) mesh_shader_supported: bool,
    /// Cached mesh-shader loader, cloned into command buffers. Built once — recreating it (or
    /// the descriptor-buffer loader) per command buffer re-runs `vkGetDeviceProcAddr` for every
    /// entry point on the hot per-frame path.
    pub(crate) mesh_shader_loader: Option<vk_mesh_shader::Device>,

    /// Present when VK_KHR_acceleration_structure was enabled (for BLAS/TLAS builds).
    pub(crate) acceleration_structure: Option<vk_accel_structure::Device>,
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
) -> RhiResult<(vk::PhysicalDevice, u32, VulkanCapabilities)> {
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
                || !has_ext(b"VK_EXT_descriptor_buffer")
                || !has_ext(b"VK_EXT_mutable_descriptor_type")
            {
                return None;
            }

            let mut vulkan11 = vk::PhysicalDeviceVulkan11Features::default();
            let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut vulkan13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut descriptor_buffer = vk::PhysicalDeviceDescriptorBufferFeaturesEXT::default();
            let mut mutable_descriptor =
                vk::PhysicalDeviceMutableDescriptorTypeFeaturesEXT::default();
            let has_mesh_ext = has_ext(b"VK_EXT_mesh_shader");
            let has_accel_ext = has_ext(b"VK_KHR_acceleration_structure")
                && has_ext(b"VK_KHR_deferred_host_operations");
            let has_ray_query_ext = has_accel_ext && has_ext(b"VK_KHR_ray_query");
            let has_rt_maintenance1_ext =
                has_ray_query_ext && has_ext(b"VK_KHR_ray_tracing_maintenance1");
            let has_rt_pipeline_ext = has_ray_query_ext && has_ext(b"VK_KHR_ray_tracing_pipeline");
            let mut mesh = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();
            let mut accel = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
            let mut ray_query = vk::PhysicalDeviceRayQueryFeaturesKHR::default();
            let mut rt_maintenance1 =
                vk::PhysicalDeviceRayTracingMaintenance1FeaturesKHR::default();
            let mut rt_pipeline = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default();

            let mut features2 = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut vulkan11)
                .push_next(&mut vulkan12)
                .push_next(&mut vulkan13)
                .push_next(&mut descriptor_buffer)
                .push_next(&mut mutable_descriptor);
            if has_mesh_ext {
                features2 = features2.push_next(&mut mesh);
            }
            if has_accel_ext {
                features2 = features2.push_next(&mut accel);
            }
            if has_ray_query_ext {
                features2 = features2.push_next(&mut ray_query);
            }
            if has_rt_maintenance1_ext {
                features2 = features2.push_next(&mut rt_maintenance1);
            }
            if has_rt_pipeline_ext {
                features2 = features2.push_next(&mut rt_pipeline);
            }
            unsafe { instance.get_physical_device_features2(*pdevice, &mut features2) };

            let base = features2.features;
            let required = base.shader_clip_distance != 0
                && base.fill_mode_non_solid != 0
                && base.multi_draw_indirect != 0
                && base.shader_int64 != 0
                && vulkan11.shader_draw_parameters != 0
                && vulkan12.buffer_device_address != 0
                && vulkan12.timeline_semaphore != 0
                && vulkan12.draw_indirect_count != 0
                && vulkan12.descriptor_binding_partially_bound != 0
                && vulkan12.runtime_descriptor_array != 0
                && vulkan12.host_query_reset != 0
                && vulkan13.dynamic_rendering != 0
                && vulkan13.synchronization2 != 0
                && descriptor_buffer.descriptor_buffer != 0
                && mutable_descriptor.mutable_descriptor_type != 0;
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

            let capabilities = VulkanCapabilities {
                mesh_shader: has_mesh_ext && mesh.mesh_shader != 0,
                acceleration_structure: has_accel_ext && accel.acceleration_structure != 0,
                ray_query: has_ray_query_ext && ray_query.ray_query != 0,
                ray_tracing_maintenance1: has_rt_maintenance1_ext
                    && rt_maintenance1.ray_tracing_maintenance1 != 0,
                ray_tracing_pipeline: has_rt_pipeline_ext && rt_pipeline.ray_tracing_pipeline != 0,
            };
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
                capabilities,
                (type_score << 48) | memory_score.min((1 << 48) - 1),
            ))
        })
        .max_by_key(|candidate| candidate.3)
        .map(|(pdevice, family, capabilities, _)| (pdevice, family, capabilities))
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

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        extension_names.push(ash::khr::portability_enumeration::NAME.as_ptr());
        extension_names.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
    }

    extension_names.push(ash::khr::surface::NAME.as_ptr());

    #[cfg(target_os = "macos")]
    {
        extension_names.push(ash::ext::metal_surface::NAME.as_ptr());
    }
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
        .api_version(vk::make_api_version(0, 1, 3, 0));

    let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::default()
    };

    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layer_names_raw)
        .enabled_extension_names(&extension_names)
        .flags(create_flags);

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

        let loader = debug_utils::Instance::new(entry, &instance);
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
    capabilities: &VulkanCapabilities,
) -> RhiResult<ash::Device> {
    let device_extension_props = unsafe {
        instance
            .enumerate_device_extension_properties(physical_device)
            .map_err(|e| RhiError::DeviceCreation(format!("Enumerate device extensions: {e}")))?
    };
    let has_ext = |needle: &[u8]| {
        device_extension_props.iter().any(|ext| {
            let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
            name.to_bytes() == needle
        })
    };
    let supports_descriptor_buffer = has_ext(b"VK_EXT_descriptor_buffer");
    let supports_mutable_descriptor_type = has_ext(b"VK_EXT_mutable_descriptor_type");
    let supports_mesh_shader = capabilities.mesh_shader;
    // BLAS/TLAS builds require acceleration structure and deferred-host-operation support.
    let supports_accel = capabilities.acceleration_structure;
    // Inline ray queries add VK_KHR_ray_query.
    let supports_ray_query = capabilities.ray_query;
    // The bindless TLAS lowering uses the address conversion provided by maintenance1.
    let supports_rt_maintenance1 = capabilities.ray_tracing_maintenance1;
    // Slang's bindless-TLAS lowering also requires the ray-tracing-pipeline extension, even
    // though the RHI only uses inline ray queries.
    let supports_ray_tracing_pipeline = capabilities.ray_tracing_pipeline;
    log::info!(
        "RHI: Optional extensions — mesh_shader={supports_mesh_shader} mutable_descriptors={supports_mutable_descriptor_type} acceleration_structure={supports_accel} ray_query={supports_ray_query} rt_maintenance1={supports_rt_maintenance1} rt_pipeline={supports_ray_tracing_pipeline}"
    );

    if !supports_descriptor_buffer {
        return Err(RhiError::Unsupported(
            "Vulkan descriptor buffer is required but not supported".into(),
        ));
    }
    if !supports_mutable_descriptor_type {
        return Err(RhiError::Unsupported(
            "Vulkan bindless storage textures require VK_EXT_mutable_descriptor_type".into(),
        ));
    }
    let mut device_extension_names: Vec<*const c_char> = vec![swapchain::NAME.as_ptr()];

    device_extension_names.push(descriptor_buffer::NAME.as_ptr());
    device_extension_names.push(ash::ext::mutable_descriptor_type::NAME.as_ptr());
    if supports_mesh_shader {
        device_extension_names.push(vk_mesh_shader::NAME.as_ptr());
    }
    if supports_accel {
        device_extension_names.push(vk_accel_structure::NAME.as_ptr());
        device_extension_names.push(ash::khr::deferred_host_operations::NAME.as_ptr());
    }
    if supports_ray_query {
        device_extension_names.push(ash::khr::ray_query::NAME.as_ptr());
    }
    if supports_rt_maintenance1 {
        device_extension_names.push(ash::khr::ray_tracing_maintenance1::NAME.as_ptr());
    }
    if supports_ray_tracing_pipeline {
        device_extension_names.push(ash::khr::ray_tracing_pipeline::NAME.as_ptr());
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        if has_ext(b"VK_KHR_portability_subset") {
            device_extension_names.push(ash::khr::portability_subset::NAME.as_ptr());
        }
    }

    // Vulkan 1.2/1.3 features use the consolidated core feature structs.
    let mut vulkan11_features =
        vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);
    let mut vulkan12_features = vk::PhysicalDeviceVulkan12Features::default()
        .buffer_device_address(true)
        .timeline_semaphore(true)
        .draw_indirect_count(true)
        .descriptor_binding_partially_bound(true)
        // Slang lowers bindless handles to an unbounded runtime descriptor array.
        .runtime_descriptor_array(true)
        .host_query_reset(true);
    let mut vulkan13_features = vk::PhysicalDeviceVulkan13Features::default()
        .dynamic_rendering(true)
        .synchronization2(true);

    let mut descriptor_buffer_features =
        vk::PhysicalDeviceDescriptorBufferFeaturesEXT::default().descriptor_buffer(true);
    let mut mutable_descriptor_type_features =
        vk::PhysicalDeviceMutableDescriptorTypeFeaturesEXT::default().mutable_descriptor_type(true);

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
    let mut rt_pipeline_features =
        vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default().ray_tracing_pipeline(true);

    let features = vk::PhysicalDeviceFeatures {
        shader_clip_distance: 1,
        fill_mode_non_solid: 1,
        multi_draw_indirect: 1,
        shader_int64: 1,
        ..Default::default()
    };

    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .features(features)
        .push_next(&mut vulkan11_features)
        .push_next(&mut vulkan12_features)
        .push_next(&mut vulkan13_features)
        .push_next(&mut descriptor_buffer_features);

    // Reassign each `push_next` result so the feature chain remains attached.
    if supports_mesh_shader {
        features2 = features2.push_next(&mut mesh_shader_features);
    }
    features2 = features2.push_next(&mut mutable_descriptor_type_features);
    if supports_accel {
        features2 = features2.push_next(&mut accel_structure_features);
    }
    if supports_ray_query {
        features2 = features2.push_next(&mut ray_query_features);
    }
    if supports_rt_maintenance1 {
        features2 = features2.push_next(&mut rt_maintenance1_features);
    }
    if supports_ray_tracing_pipeline {
        features2 = features2.push_next(&mut rt_pipeline_features);
    }

    let priorities = [1.0f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&priorities);

    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(std::slice::from_ref(&queue_info))
        .enabled_extension_names(&device_extension_names)
        .push_next(&mut features2);

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

        let (physical_device, queue_family_index, capabilities) =
            select_physical_device(&instance)?;

        let device_props = unsafe { instance.get_physical_device_properties(physical_device) };
        let device_name = unsafe {
            std::ffi::CStr::from_ptr(device_props.device_name.as_ptr())
                .to_string_lossy()
                .to_string()
        };
        log::info!("RHI: Selected GPU: {}", device_name);

        let device = create_logical_device(
            &instance,
            physical_device,
            queue_family_index,
            &capabilities,
        )?;
        let bindless_mode = BindlessMode::DescriptorBuffer;

        let present_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // Device-level half: object names and label regions. `debug_utils_loader` is the
        // instance-level messenger and only exists under validation.
        let debug_labels =
            debug_utils_available.then(|| debug_utils::Device::new(&instance, &device));

        let surface_loader = surface::Instance::new(&entry, &instance);
        let swapchain_loader = swapchain::Device::new(&instance, &device);
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

        let descriptor_buffer_loader = descriptor_buffer::Device::new(&instance, &device);

        let acceleration_structure_opt = if capabilities.acceleration_structure {
            Some(vk_accel_structure::Device::new(&instance, &device))
        } else {
            None
        };
        let mesh_shader_loader = if capabilities.mesh_shader {
            Some(vk_mesh_shader::Device::new(&instance, &device))
        } else {
            None
        };
        let (texture_descriptor_set_layout, descriptor_buffer_heap) = {
            let heap = create_descriptor_buffer_heap(
                &instance,
                &device,
                physical_device,
                &device_memory_properties,
                &descriptor_buffer_loader,
            )?;
            (heap.layout, heap)
        };
        let free_texture_ids = Rc::new(RefCell::new(Vec::new()));
        let free_sampler_ids = Rc::new(RefCell::new(Vec::new()));
        let mut completion_type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let completion_info =
            vk::SemaphoreCreateInfo::default().push_next(&mut completion_type_info);
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
                swapchain_loader: swapchain::Device::new(&instance, &device),
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
            bindless_mode,
            timestamp_period: device_props.limits.timestamp_period,
            surface_loader,
            swapchain_loader,
            descriptor_buffer_loader,
            debug_utils_loader,
            debug_labels,
            debug_callback,
            texture_descriptor_set_layout,
            descriptor_buffer_heap,
            textures: Rc::new(RefCell::new(Vec::new())),
            texture_view_flags: RefCell::new(Vec::new()),
            next_texture_id: RefCell::new(0),
            free_texture_ids,
            allocations: Rc::new(RefCell::new(BTreeMap::new())),
            mapped_allocations: Rc::new(RefCell::new(BTreeMap::new())),
            samplers: RefCell::new(Vec::new()),
            next_sampler_id: RefCell::new(0),
            free_sampler_ids,
            timeline_semaphores: RefCell::new(Vec::new()),
            query_pools: RefCell::new(Vec::new()),
            setup_command_buffer,
            setup_recording: Cell::new(false),
            mesh_shader_supported: capabilities.mesh_shader,
            mesh_shader_loader,
            acceleration_structure: acceleration_structure_opt,
        })
    }

    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    pub fn bindless_mode(&self) -> BindlessMode {
        self.bindless_mode
    }

    pub fn wait_idle(&self) {
        backend_expect!(&self.queue.inner, QueueInner::Vulkan).wait_idle();
    }

    pub fn wait_for_frame(&self, frame_index: usize) {
        backend_expect!(&self.queue.inner, QueueInner::Vulkan).wait_for_frame(frame_index);
    }

    pub fn create_surface(&self, desc: &SurfaceDesc) -> RhiResult<Surface> {
        let surface = unsafe {
            ash_window::create_surface(
                &self.entry,
                &self.instance,
                desc.display_handle,
                desc.window_handle,
                None,
            )
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

    pub fn create_allocation(&self, desc: &AllocationDesc) -> RhiResult<Allocation> {
        let mut usage_flags = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::INDIRECT_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        // Buffers are untyped in the RHI, so AS-capable devices give every buffer build-input use.
        if self.acceleration_structure.is_some() {
            usage_flags |= vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;
        }

        let buffer_info = vk::BufferCreateInfo::default()
            .size(desc.size)
            .usage(usage_flags)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            self.device
                .create_buffer(&buffer_info, None)
                .map_err(|e| RhiError::BufferCreation(e.to_string()))?
        };

        let mem_requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

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
            unsafe { self.device.destroy_buffer(buffer, None) };
            return Err(RhiError::AllocationFailed("No suitable memory type".into()));
        }

        let mut attempt = Err(RhiError::AllocationFailed("No suitable memory type".into()));
        let mut mem_type_index = 0;
        for candidate in candidates.into_iter().flatten() {
            mem_type_index = candidate;
            // Follows the memory type's real properties, not the requested `MemoryType`: on UMA one
            // type serves both, and a block created by a `GpuOnly` buffer must still be mappable
            // for an `Upload` buffer landing in it later.
            let host_visible = self.device_memory_properties.memory_types[candidate as usize]
                .property_flags
                .contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
            let mut pool = self.buffer_pool.borrow_mut();
            attempt = pool.allocate(&self.device, &mem_requirements, candidate, host_visible);
            if attempt.is_ok() {
                break;
            }
        }
        let suballocation = match attempt {
            Ok(suballocation) => suballocation,
            Err(error) => {
                unsafe { self.device.destroy_buffer(buffer, None) };
                return Err(error);
            }
        };

        if let Err(error) = unsafe {
            self.device
                .bind_buffer_memory(buffer, suballocation.memory, suballocation.offset)
        } {
            unsafe { self.device.destroy_buffer(buffer, None) };
            self.buffer_pool.borrow_mut().release(
                suballocation.block_index,
                suballocation.offset,
                suballocation.range,
            );
            return Err(RhiError::BufferCreation(error.to_string()));
        }

        let addr_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        let gpu_addr = unsafe { self.device.get_buffer_device_address(&addr_info) };

        // `GpuOnly` promises no CPU pointer even when it happens to land in host-visible memory.
        let mapped_ptr = match desc.memory {
            MemoryType::Upload | MemoryType::Readback => suballocation.mapped_ptr,
            MemoryType::GpuOnly => None,
        };

        if let Some(label) = desc.label.as_deref() {
            self.set_object_name(buffer, label);
        }

        let vk_buffer = VulkanBuffer {
            buffer,
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
                    buffer: vk_buffer.buffer,
                    memory: vk_buffer.memory,
                    memory_offset: vk_buffer.block_offset,
                    memory_type_index: mem_type_index,
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

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<GpuPtr<u8>>() as u32);

        let set_layouts = [self.texture_descriptor_set_layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        let pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|e| RhiError::PipelineCreation(e.to_string()))?
        };
        let mut vk_pso = VulkanGraphicsPso {
            pipeline: vk::Pipeline::null(),
            pipeline_layout,
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
            inner: GraphicsPsoInner::Vulkan(Box::new(vk_pso)),
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

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<GpuPtr<u8>>() as u32);

        let set_layouts = [self.texture_descriptor_set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        let pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&layout_info, None)
                .map_err(|e| RhiError::PipelineCreation(e.to_string()))?
        };

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            // Descriptor-buffer layouts require the matching pipeline flag.
            .flags(vk::PipelineCreateFlags::DESCRIPTOR_BUFFER_EXT)
            .stage(stage)
            .layout(pipeline_layout);

        let pipelines = unsafe {
            self.device
                .create_compute_pipelines(self.pipeline_cache, &[pipeline_info], None)
                .map_err(|e| RhiError::PipelineCreation(format!("{e:?}")))?
        };

        if let Some(label) = desc.label.as_deref() {
            self.set_object_name(pipelines[0], label);
        }

        Ok(ComputePso {
            inner: ComputePsoInner::Vulkan(Box::new(VulkanComputePso {
                pipeline: pipelines[0],
                pipeline_layout,
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
        if !self.mesh_shader_supported {
            return Err(RhiError::Unsupported(
                "VK_EXT_mesh_shader not available on this device".into(),
            ));
        }

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

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<GpuPtr<u8>>() as u32);

        let set_layouts = [self.texture_descriptor_set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        let pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&layout_info, None)
                .map_err(|e| RhiError::PipelineCreation(e.to_string()))?
        };

        let mut vk_pso = VulkanMeshletPso {
            pipeline: vk::Pipeline::null(),
            pipeline_layout,
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
            inner: MeshletPsoInner::Vulkan(Box::new(vk_pso)),
            _owner: None,
        })
    }

    pub(crate) fn require_accel_loader(&self) -> RhiResult<&vk_accel_structure::Device> {
        self.acceleration_structure.as_ref().ok_or_else(|| {
            RhiError::Unsupported(
                "VK_KHR_acceleration_structure not available on this device".into(),
            )
        })
    }

    /// `minAccelerationStructureScratchOffsetAlignment` — the required alignment for the
    /// `scratch_data` address passed to `vkCmdBuildAccelerationStructuresKHR`.
    pub(crate) fn accel_scratch_alignment(&self) -> u64 {
        let mut accel_props = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut accel_props);
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

        let heap = &self.descriptor_buffer_heap;
        let descriptor_buffer_binding = vk::DescriptorBufferBindingInfoEXT::default()
            .address(heap.gpu_address.address)
            .usage(
                vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT
                    | vk::BufferUsageFlags::SAMPLER_DESCRIPTOR_BUFFER_EXT,
            );
        let descriptor_buffer_loader = self.descriptor_buffer_loader.clone();
        let mesh_shader = self.mesh_shader_loader.clone();
        let accel_loader_cmd = self.acceleration_structure.clone();

        Ok(CommandBuffer {
            inner: CommandBufferInner::Vulkan(Box::new(VulkanCommandBuffer {
                command_buffer: cmd,
                device: self.device.clone(),
                swapchain_image_views: Rc::from([]),
                swapchain_images: Rc::from([]),
                depth_image_view: vk::ImageView::null(),
                pipeline_layout: vk::PipelineLayout::null(),
                descriptor_buffer_loader,
                descriptor_buffer_binding,
                push_constant_stages: vk::ShaderStageFlags::empty(),
                debug_labels: self.debug_labels.clone(),
                in_labelled_pass: false,
                pending_split_barrier: None,
                allocations: self.allocations.clone(),
                textures: self.textures.clone(),
                mesh_shader,
                acceleration_structure: accel_loader_cmd,
                rendered_swapchain_images: SmallVec::new(),
                ended: false,
            })),
            _owner: None,
        })
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

        let heap = &self.descriptor_buffer_heap;
        let descriptor_buffer_binding = vk::DescriptorBufferBindingInfoEXT::default()
            .address(heap.gpu_address.address)
            .usage(
                vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT
                    | vk::BufferUsageFlags::SAMPLER_DESCRIPTOR_BUFFER_EXT,
            );
        let descriptor_buffer_loader = self.descriptor_buffer_loader.clone();
        let mesh_shader = self.mesh_shader_loader.clone();
        let accel_loader_cmd = self.acceleration_structure.clone();

        Ok(CommandBuffer {
            inner: CommandBufferInner::Vulkan(Box::new(VulkanCommandBuffer {
                command_buffer: cmd,
                device: self.device.clone(),
                swapchain_image_views: sc.image_views.clone(),
                swapchain_images: sc.images.clone(),
                depth_image_view: sc.depth_image_view,
                pipeline_layout: vk::PipelineLayout::null(),
                descriptor_buffer_loader,
                descriptor_buffer_binding,
                push_constant_stages: vk::ShaderStageFlags::empty(),
                debug_labels: self.debug_labels.clone(),
                in_labelled_pass: false,
                pending_split_barrier: None,
                allocations: self.allocations.clone(),
                textures: self.textures.clone(),
                mesh_shader,
                acceleration_structure: accel_loader_cmd,
                rendered_swapchain_images: SmallVec::new(),
                ended: false,
            })),
            _owner: None,
        })
    }

    pub fn create_timeline_semaphore(&self, initial_value: u64) -> RhiResult<TimelineSemaphore> {
        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(initial_value);

        let semaphore_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);

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

    pub(crate) fn write_image_descriptor(
        &self,
        id: TextureId,
        image_view: vk::ImageView,
        layout: vk::ImageLayout,
        storage: bool,
    ) -> RhiResult<()> {
        let heap = &self.descriptor_buffer_heap;
        let loader = &self.descriptor_buffer_loader;

        let (base, descriptor_size, ty, kind) = if storage {
            (
                heap.storage_image_offset,
                heap.storage_image_descriptor_size,
                vk::DescriptorType::STORAGE_IMAGE,
                "Storage",
            )
        } else {
            (
                heap.sampled_image_offset,
                heap.sampled_image_descriptor_size,
                vk::DescriptorType::SAMPLED_IMAGE,
                "Sampled",
            )
        };

        let offset = base + (id.0 as u64) * heap.image_descriptor_stride;
        if offset + heap.image_descriptor_stride > heap.size {
            return Err(RhiError::Backend(format!(
                "{kind} image descriptor heap overflow"
            )));
        }

        let image_info = vk::DescriptorImageInfo::default()
            .image_view(image_view)
            .image_layout(layout);
        // Both p_sampled_image and p_storage_image are the same union variant — a pointer
        // to vk::DescriptorImageInfo. The descriptor type tag selects the layout written.
        let data = if storage {
            vk::DescriptorDataEXT {
                p_storage_image: &image_info,
            }
        } else {
            vk::DescriptorDataEXT {
                p_sampled_image: &image_info,
            }
        };
        let get_info = vk::DescriptorGetInfoEXT::default().ty(ty).data(data);

        unsafe {
            let dst = std::slice::from_raw_parts_mut(
                heap.mapped_ptr.add(offset as usize),
                descriptor_size as usize,
            );
            loader.get_descriptor(&get_info, dst);
        }
        Ok(())
    }

    pub(crate) fn transition_depth_image(&self, depth_image: vk::Image) -> RhiResult<()> {
        let barrier = vk::ImageMemoryBarrier::default()
            .image(depth_image)
            .dst_access_mask(
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
            .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .layer_count(1)
                    .level_count(1),
            );
        self.submit_setup_barrier(
            barrier,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        )
    }

    pub(crate) fn transition_image_to_layout(
        &self,
        image: vk::Image,
        aspect: vk::ImageAspectFlags,
        mip_levels: u32,
        layer_count: u32,
        new_layout: vk::ImageLayout,
    ) -> RhiResult<()> {
        let barrier = vk::ImageMemoryBarrier::default()
            .image(image)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(new_layout)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(aspect)
                    .level_count(mip_levels)
                    .layer_count(layer_count),
            );
        self.submit_setup_barrier(
            barrier,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::ALL_COMMANDS,
        )
    }

    /// Append an image barrier to the reusable setup command buffer. The batch is submitted when
    /// the next user command buffer is created, so loading a scene with many textures incurs one
    /// setup submission rather than one queue idle per texture.
    fn submit_setup_barrier(
        &self,
        barrier: vk::ImageMemoryBarrier<'_>,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
    ) -> RhiResult<()> {
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
            self.device.cmd_pipeline_barrier(
                self.setup_command_buffer,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
        Ok(())
    }

    fn flush_setup_barriers(&self) -> RhiResult<()> {
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

            for sampler in self.samplers.borrow_mut().drain(..).flatten() {
                self.device.destroy_sampler(sampler, None);
            }

            let heap = &self.descriptor_buffer_heap;
            self.device.unmap_memory(heap.memory);
            self.device.destroy_buffer(heap.buffer, None);
            self.device.free_memory(heap.memory, None);
            self.device.destroy_descriptor_set_layout(heap.layout, None);

            self.device
                .destroy_pipeline_cache(self.pipeline_cache, None);
            self.device.destroy_command_pool(self.command_pool, None);

            // Buffers the application never destroyed are still bound into pool blocks, and
            // freeing block memory underneath a live buffer is invalid usage.
            for (_, allocation) in std::mem::take(&mut *self.allocations.borrow_mut()) {
                self.device.destroy_buffer(allocation.buffer, None);
            }
            self.buffer_pool.borrow_mut().destroy_all(&self.device);
            self.device.destroy_device(None);

            if let Some(ref debug_loader) = self.debug_utils_loader {
                debug_loader.destroy_debug_utils_messenger(self.debug_callback, None);
            }

            self.instance.destroy_instance(None);
        }
    }
}

fn create_descriptor_buffer_heap(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    loader: &descriptor_buffer::Device,
) -> RhiResult<DescriptorBufferHeap> {
    let binding_flags = [
        vk::DescriptorBindingFlags::PARTIALLY_BOUND,
        vk::DescriptorBindingFlags::PARTIALLY_BOUND,
    ];
    let mut binding_flags_info =
        vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);

    let bindings = [
        vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::SAMPLER,
            descriptor_count: MAX_BINDLESS_SAMPLERS,
            stage_flags: vk::ShaderStageFlags::ALL,
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            binding: 2,
            descriptor_type: vk::DescriptorType::MUTABLE_EXT,
            descriptor_count: MAX_BINDLESS_TEXTURES,
            stage_flags: vk::ShaderStageFlags::ALL,
            ..Default::default()
        },
    ];

    let image_descriptor_types = [
        vk::DescriptorType::SAMPLED_IMAGE,
        vk::DescriptorType::STORAGE_IMAGE,
    ];
    let mutable_descriptor_lists = [
        vk::MutableDescriptorTypeListEXT::default(),
        vk::MutableDescriptorTypeListEXT::default().descriptor_types(&image_descriptor_types),
    ];
    let mut mutable_descriptor_info = vk::MutableDescriptorTypeCreateInfoEXT::default()
        .mutable_descriptor_type_lists(&mutable_descriptor_lists);

    let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
        .bindings(&bindings)
        .flags(vk::DescriptorSetLayoutCreateFlags::DESCRIPTOR_BUFFER_EXT)
        .push_next(&mut binding_flags_info);
    let mut layout_info = layout_info;
    layout_info = layout_info.push_next(&mut mutable_descriptor_info);

    let layout = unsafe {
        device
            .create_descriptor_set_layout(&layout_info, None)
            .map_err(|e| RhiError::DeviceCreation(format!("Descriptor buffer layout: {e}")))?
    };

    let mut props = vk::PhysicalDeviceDescriptorBufferPropertiesEXT::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut props);
    unsafe {
        instance.get_physical_device_properties2(physical_device, &mut props2);
    }

    let layout_size = unsafe { loader.get_descriptor_set_layout_size(layout) };
    let sampler_offset = unsafe { loader.get_descriptor_set_layout_binding_offset(layout, 0) };
    let sampled_image_offset =
        unsafe { loader.get_descriptor_set_layout_binding_offset(layout, 2) };
    let storage_image_offset = sampled_image_offset;

    let sampled_image_descriptor_size = props.sampled_image_descriptor_size as u64;
    let storage_image_descriptor_size = props.storage_image_descriptor_size as u64;
    let image_descriptor_stride = mutable_image_descriptor_stride(
        sampled_image_descriptor_size,
        storage_image_descriptor_size,
    );
    let sampler_stride = props.sampler_descriptor_size as u64;

    let align = props.descriptor_buffer_offset_alignment.max(1);
    let aligned_size = (layout_size + align - 1) & !(align - 1);

    let (buffer, memory, gpu_address) = super::memory::allocate_bound_buffer(
        device,
        mem_props,
        aligned_size,
        vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT
            | vk::BufferUsageFlags::SAMPLER_DESCRIPTOR_BUFFER_EXT,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        "descriptor buffer",
    )?;

    let mapped_ptr = unsafe {
        device
            .map_memory(memory, 0, aligned_size, vk::MemoryMapFlags::empty())
            .map_err(|e| RhiError::AllocationFailed(e.to_string()))?
    } as *mut u8;

    Ok(DescriptorBufferHeap {
        buffer,
        memory,
        mapped_ptr,
        size: aligned_size,
        gpu_address: GpuPtr::from_addr(gpu_address),
        layout,
        sampled_image_offset,
        sampler_offset,
        storage_image_offset,
        image_descriptor_stride,
        sampled_image_descriptor_size,
        storage_image_descriptor_size,
        sampler_stride,
    })
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

pub(crate) fn mutable_image_descriptor_stride(sampled_size: u64, storage_size: u64) -> u64 {
    sampled_size.max(storage_size)
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

/// The layout a newly-created image lives in for its whole life. There is no per-resource layout
/// tracker, so nothing transitions it afterwards except the round-trip inside a copy — which makes
/// this the single source of truth for both the image and its descriptors. Any usage mix needing
/// different layouts at different points must therefore settle on GENERAL.
pub(crate) fn initial_image_layout(desc: &TextureDesc) -> vk::ImageLayout {
    use crate::texture::TextureUsage;

    const ATTACHMENT: TextureUsage =
        TextureUsage::COLOR_ATTACHMENT.union(TextureUsage::DEPTH_STENCIL_ATTACHMENT);

    let usage = desc.usage;
    if usage
        .intersects(TextureUsage::STORAGE | TextureUsage::TRANSFER_SRC | TextureUsage::TRANSFER_DST)
    {
        return vk::ImageLayout::GENERAL;
    }
    // Render-then-sample would need an attachment layout and SHADER_READ_ONLY_OPTIMAL at
    // different times; with no tracker, GENERAL is the only one correct for both.
    if usage.contains(TextureUsage::SAMPLED) && usage.intersects(ATTACHMENT) {
        return vk::ImageLayout::GENERAL;
    }
    if usage.contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT)
        && !usage.contains(TextureUsage::COLOR_ATTACHMENT)
    {
        return vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }
    if usage.contains(TextureUsage::COLOR_ATTACHMENT)
        && !usage.contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT)
    {
        return vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
    }
    if usage.contains(TextureUsage::SAMPLED) {
        return vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    }
    vk::ImageLayout::GENERAL
}

/// The layout a sampled descriptor must declare — necessarily the one the image was created in.
pub(crate) fn sampled_image_layout(desc: &TextureDesc) -> vk::ImageLayout {
    initial_image_layout(desc)
}

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
    fn mutable_image_descriptors_use_the_union_stride() {
        assert_eq!(mutable_image_descriptor_stride(32, 64), 64);
        assert_eq!(mutable_image_descriptor_stride(64, 32), 64);
        assert_eq!(mutable_image_descriptor_stride(32, 32), 32);
    }

    #[test]
    fn readback_memory_is_host_coherent() {
        let flags = buffer_memory_flags(MemoryType::Readback);
        assert!(flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE));
        assert!(flags.contains(vk::MemoryPropertyFlags::HOST_CACHED));
        assert!(flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT));
    }

    fn texture_with(usage: crate::texture::TextureUsage) -> TextureDesc {
        TextureDesc {
            usage,
            ..Default::default()
        }
    }

    #[test]
    fn sampled_descriptors_declare_the_layout_the_image_is_actually_in() {
        use crate::texture::TextureUsage;

        for usage in [
            TextureUsage::SAMPLED,
            TextureUsage::SAMPLED | TextureUsage::TRANSFER_DST,
            TextureUsage::SAMPLED | TextureUsage::STORAGE,
            TextureUsage::SAMPLED | TextureUsage::COLOR_ATTACHMENT,
            TextureUsage::SAMPLED | TextureUsage::DEPTH_STENCIL_ATTACHMENT,
        ] {
            let desc = texture_with(usage);
            assert_eq!(
                sampled_image_layout(&desc),
                initial_image_layout(&desc),
                "sampled layout diverged from the created layout for {usage:?}"
            );
        }
    }

    #[test]
    fn attachments_that_are_also_sampled_stay_general() {
        use crate::texture::TextureUsage;

        assert_eq!(
            initial_image_layout(&texture_with(
                TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED
            )),
            vk::ImageLayout::GENERAL
        );
        assert_eq!(
            initial_image_layout(&texture_with(
                TextureUsage::DEPTH_STENCIL_ATTACHMENT | TextureUsage::SAMPLED
            )),
            vk::ImageLayout::GENERAL
        );
    }

    #[test]
    fn single_purpose_images_keep_optimal_layouts() {
        use crate::texture::TextureUsage;

        assert_eq!(
            initial_image_layout(&texture_with(TextureUsage::SAMPLED)),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        );
        assert_eq!(
            initial_image_layout(&texture_with(TextureUsage::COLOR_ATTACHMENT)),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        );
        assert_eq!(
            initial_image_layout(&texture_with(TextureUsage::DEPTH_STENCIL_ATTACHMENT)),
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        );
        assert_eq!(
            initial_image_layout(&texture_with(
                TextureUsage::COLOR_ATTACHMENT | TextureUsage::TRANSFER_SRC
            )),
            vk::ImageLayout::GENERAL
        );
    }
}
