use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, VecDeque};
use std::ffi::{CStr, c_char};
use std::rc::Rc;

use ash::{
    Entry, Instance,
    ext::{debug_utils, descriptor_heap, mesh_shader as vk_mesh_shader},
    khr::{
        acceleration_structure as vk_accel_structure, device_address_commands, surface, swapchain,
    },
    vk,
    vk::TaggedStructure as _,
};
use zerocopy::FromBytes;

use crate::device::DeviceDesc;
use crate::error::{RhiError, RhiResult};
use crate::memory::MemoryType;
use crate::queue::Queue;
use crate::types::{
    AddressMode, BuildAccelFlags, CompareOp, Format, GeometryFlags, GpuPtr, MAX_FRAMES_IN_FLIGHT,
    TextureId,
};

use super::heap::{DescriptorHeaps, create_descriptor_heaps};
use super::memory::{SharedBufferPool, VulkanBufferPool};
use super::queue::{FrameSubmission, VulkanQueue};
use super::texture::VulkanTexture;
use crate::backend::mapped::MappedAllocations;
use crate::backend::retire::RetirementQueue;
use crate::backend::slots::SlotTable;

#[derive(Clone)]
pub(crate) struct BufferAllocation {
    pub base: GpuPtr<u8>,
    pub size: u64,
    pub memory: vk::DeviceMemory,
    /// Offset within the pooled block; images placed here bind at this plus their own offset.
    pub memory_offset: u64,
}

/// Buffer allocations keyed by GPU base address, so a texture placement resolves its backing
/// memory in O(log n).
pub(crate) type SharedAllocations = Rc<RefCell<BTreeMap<u64, BufferAllocation>>>;
type SharedMappedAllocations = Rc<RefCell<MappedAllocations>>;
/// Texture slots, shared with the command buffer (which resolves ids) and the queue (which
/// recycles them once a frame retires).
pub(crate) type SharedTextures = Rc<SlotTable<VulkanTexture>>;
/// A Vulkan sampler under descriptor heaps is only a descriptor, so the table carries no payload
/// and exists purely to own which ids are live.
pub(crate) type SharedSamplerIds = Rc<SlotTable<()>>;

/// The device-level entry-point tables a command buffer needs. Shared behind one `Rc` because
/// they are ~2 KB of function pointers and a command buffer is built at least once a frame.
pub(crate) struct VulkanLoaders {
    pub(crate) device: ash::Device,
    pub(crate) descriptor_heap: descriptor_heap::Device,
    pub(crate) address_commands: device_address_commands::Device,
    pub(crate) mesh_shader: vk_mesh_shader::Device,
    pub(crate) acceleration_structure: vk_accel_structure::Device,
    /// Device-level `VK_EXT_debug_utils`: object names and pass label regions.
    pub(crate) debug_labels: Option<debug_utils::Device>,
}

pub struct VulkanDevice {
    /// Shared with every command buffer this device creates.
    pub(crate) loaders: Rc<VulkanLoaders>,
    pub(crate) entry: Entry,
    pub(crate) instance: Instance,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) queue: Rc<VulkanQueue>,
    /// `queue`, wrapped for [`Device::queue`](crate::Device::queue).
    rhi_queue: Queue,
    pub(crate) command_pool: vk::CommandPool,
    pub(crate) pipeline_cache: vk::PipelineCache,
    /// Shared with the queue so retirement can hand ranges back.
    pub(crate) buffer_pool: SharedBufferPool,
    pub(crate) device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// Nanoseconds per timestamp tick (`VkPhysicalDeviceLimits::timestampPeriod`).
    pub(crate) timestamp_period: f32,
    /// `minAccelerationStructureScratchOffsetAlignment`, probed once at device creation: it is a
    /// physical-device property, and every acceleration-structure build needs it.
    pub(crate) accel_scratch_alignment: u64,

    pub(crate) surface_loader: surface::Instance,
    pub(crate) swapchain_loader: swapchain::Device,

    pub(crate) debug_utils_loader: Option<debug_utils::Instance>,
    pub(crate) debug_callback: vk::DebugUtilsMessengerEXT,

    pub(crate) descriptor_heaps: DescriptorHeaps,
    /// `(usage, alignment, memory_type_bits)` for [`Self::buffer_requirements`], one entry per
    /// usage class, probed on first use.
    pub(crate) buffer_requirements_probe: RefCell<Vec<(vk::BufferUsageFlags, u64, u32)>>,
    pub(crate) textures: SharedTextures,
    pub(crate) allocations: SharedAllocations,
    pub(crate) mapped_allocations: SharedMappedAllocations,

    pub(crate) samplers: SharedSamplerIds,

    /// User timeline semaphores stay alive until device teardown. This permits callers to use a
    /// temporary `SubmitDesc` without destroying a semaphore still referenced by queued work.
    pub(crate) timeline_semaphores: RefCell<Vec<vk::Semaphore>>,
    pub(crate) query_pools: RefCell<Vec<vk::QueryPool>>,

    pub(crate) setup_command_buffer: vk::CommandBuffer,
    /// Whether the batched setup command buffer is currently recording initial image layouts.
    pub(crate) setup_recording: Cell<bool>,
}

/// Device extensions the RHI needs. One list, used both to reject an adapter that lacks them and
/// to enable them, so the two cannot disagree.
const REQUIRED_DEVICE_EXTENSIONS: &[&CStr] = &[
    swapchain::NAME,
    descriptor_heap::NAME,
    ash::khr::shader_untyped_pointers::NAME,
    device_address_commands::NAME,
    ash::khr::unified_image_layouts::NAME,
    vk_mesh_shader::NAME,
    vk_accel_structure::NAME,
    ash::khr::deferred_host_operations::NAME,
    ash::khr::ray_query::NAME,
    ash::khr::ray_tracing_maintenance1::NAME,
];

/// Device features the RHI needs, per feature struct.
///
/// Generates both halves from one list: the check that rejects an adapter, and the chain that
/// enables them at device creation. Written separately, requiring a feature without enabling it
/// produces a device that quietly lacks it.
macro_rules! required_features {
    ($( $name:ident : $ty:ty { $($feature:ident),+ $(,)? } ),+ $(,)?) => {
        /// Whether `pdevice` supports everything [`required_features!`] names.
        fn supports_required_features(
            instance: &ash::Instance,
            pdevice: vk::PhysicalDevice,
        ) -> bool {
            $( let mut $name = <$ty>::default(); )+
            let mut chain = vk::PhysicalDeviceFeatures2::default() $( .push(&mut $name) )+;
            // SAFETY: every link in the chain is a live, well-formed feature struct.
            unsafe { instance.get_physical_device_features2(pdevice, &mut chain) };
            chain.features.shader_int64 != 0 $($( && $name.$feature != 0 )+)+
        }

        /// Build the enabled-feature chain and hand it to `f`, which must use it before
        /// returning: the feature structs it links live only for the call.
        fn with_required_features<R>(
            f: impl FnOnce(&mut vk::PhysicalDeviceFeatures2<'_>) -> R,
        ) -> R {
            $( let mut $name = <$ty>::default() $( .$feature(true) )+; )+
            let mut chain = vk::PhysicalDeviceFeatures2::default()
                .features(vk::PhysicalDeviceFeatures {
                    // The only core feature needed: addresses and bindless handles are 64-bit.
                    shader_int64: 1,
                    ..Default::default()
                })
                $( .push(&mut $name) )+;
            f(&mut chain)
        }
    };
}

required_features! {
    vulkan11: vk::PhysicalDeviceVulkan11Features { shader_draw_parameters },
    vulkan12: vk::PhysicalDeviceVulkan12Features {
        buffer_device_address, timeline_semaphore, host_query_reset,
    },
    vulkan13: vk::PhysicalDeviceVulkan13Features { dynamic_rendering, synchronization2 },
    // Core in 1.4 and required by `VK_EXT_descriptor_heap`: it carries
    // `VkPipelineCreateFlags2CreateInfo`, the only place the descriptor-heap pipeline bit lives.
    vulkan14: vk::PhysicalDeviceVulkan14Features { maintenance5 },
    descriptor_heap: vk::PhysicalDeviceDescriptorHeapFeaturesEXT { descriptor_heap },
    untyped_pointers: vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR {
        shader_untyped_pointers,
    },
    address_commands: vk::PhysicalDeviceDeviceAddressCommandsFeaturesKHR {
        device_address_commands,
    },
    unified_layouts: vk::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR { unified_image_layouts },
    // Mesh only: task/amplification shaders are not exposed, and asking for one would exclude
    // mesh-only devices.
    mesh: vk::PhysicalDeviceMeshShaderFeaturesEXT { mesh_shader },
    accel: vk::PhysicalDeviceAccelerationStructureFeaturesKHR { acceleration_structure },
    ray_query: vk::PhysicalDeviceRayQueryFeaturesKHR { ray_query },
    rt_maintenance1: vk::PhysicalDeviceRayTracingMaintenance1FeaturesKHR {
        ray_tracing_maintenance1,
    },
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
        Cow::from("")
    } else {
        unsafe { CStr::from_ptr(callback_data.p_message).to_string_lossy() }
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

/// The first memory type allowed by `memory_req` that has all of `flags`.
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

/// Convert an RHI [`Format`] to a Vulkan `VkFormat`.
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

/// Convert a Vulkan format back to an RHI [`Format`], or `None` when the RHI has no name for it.
///
/// The exact inverse of [`format_to_vk`]; `vk_to_format_is_the_inverse_of_format_to_vk` keeps the
/// two in step. `None` rather than a fallback: the only caller reports a surface format back to
/// the application, and guessing there hands out a `Format` that will not match the attachment
/// any pipeline is built against.
pub(crate) fn vk_to_format(format: vk::Format) -> Option<Format> {
    Some(match format {
        vk::Format::R8_UNORM => Format::R8Unorm,
        vk::Format::R8G8_UNORM => Format::R8G8Unorm,
        vk::Format::R8G8B8A8_UNORM => Format::R8G8B8A8Unorm,
        vk::Format::R8G8B8A8_SRGB => Format::R8G8B8A8Srgb,
        vk::Format::B8G8R8A8_UNORM => Format::B8G8R8A8Unorm,
        vk::Format::B8G8R8A8_SRGB => Format::B8G8R8A8Srgb,
        vk::Format::R16_SFLOAT => Format::R16Float,
        vk::Format::R16G16_SFLOAT => Format::R16G16Float,
        vk::Format::R16G16B16A16_SFLOAT => Format::R16G16B16A16Float,
        vk::Format::R32_SFLOAT => Format::R32Float,
        vk::Format::R32G32_SFLOAT => Format::R32G32Float,
        vk::Format::R32G32B32A32_SFLOAT => Format::R32G32B32A32Float,
        vk::Format::A2B10G10R10_UNORM_PACK32 => Format::R10G10B10A2Unorm,
        vk::Format::B10G11R11_UFLOAT_PACK32 => Format::R11G11B10Float,
        vk::Format::D16_UNORM => Format::D16Unorm,
        vk::Format::D32_SFLOAT => Format::D32Float,
        vk::Format::D24_UNORM_S8_UINT => Format::D24UnormS8Uint,
        vk::Format::D32_SFLOAT_S8_UINT => Format::D32FloatS8Uint,
        vk::Format::R16_UINT => Format::R16Uint,
        vk::Format::R32_UINT => Format::R32Uint,
        _ => return None,
    })
}

pub(crate) fn build_accel_flags_to_vk(
    flags: BuildAccelFlags,
) -> vk::BuildAccelerationStructureFlagsKHR {
    let mut out = vk::BuildAccelerationStructureFlagsKHR::empty();
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
fn select_physical_device(instance: &ash::Instance) -> RhiResult<(vk::PhysicalDevice, u32)> {
    let physical_devices = unsafe {
        instance
            .enumerate_physical_devices()
            .map_err(|e| RhiError::DeviceCreation(format!("Enumerate devices: {e}").into()))?
    };

    let selected = physical_devices
        .iter()
        .filter_map(|pdevice| {
            let props = unsafe { instance.get_physical_device_properties(*pdevice) };
            if props.api_version < vk::API_VERSION_1_4 {
                return None;
            }

            let extensions =
                unsafe { instance.enumerate_device_extension_properties(*pdevice) }.ok()?;
            let supports_all = REQUIRED_DEVICE_EXTENSIONS.iter().all(|required| {
                extensions
                    .iter()
                    .any(|ext| ext.extension_name_as_c_str() == Ok(*required))
            });
            if !supports_all || !supports_required_features(instance, *pdevice) {
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
        entry.create_instance(&create_info, None).map_err(|e| {
            RhiError::DeviceCreation(format!("Failed to create instance: {e}").into())
        })?
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
                .map_err(|e| RhiError::DeviceCreation(format!("Debug callback: {e}").into()))?
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

/// Create the logical device with every required extension and feature enabled.
fn create_logical_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
) -> RhiResult<ash::Device> {
    // Adapter selection already rejected anything missing these, so they are simply enabled.
    let extension_names: Vec<*const c_char> = REQUIRED_DEVICE_EXTENSIONS
        .iter()
        .map(|name| name.as_ptr())
        .collect();

    let priorities = [1.0f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&priorities);

    with_required_features(|features| {
        // SAFETY: `extend` rather than `push` because `features` already carries the chain the
        // macro assembled; every link is live for this call.
        let create_info = unsafe {
            vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&queue_info))
                .enabled_extension_names(&extension_names)
                .extend(features)
        };
        unsafe { instance.create_device(physical_device, &create_info, None) }
    })
    .map_err(|e| RhiError::DeviceCreation(format!("Failed to create device: {e}").into()))
}

impl VulkanDevice {
    pub fn new(desc: &DeviceDesc) -> RhiResult<Self> {
        let entry = unsafe { Entry::load() }
            .map_err(|e| RhiError::DeviceCreation(format!("Failed to load Vulkan: {e}").into()))?;

        let (instance, debug_utils_loader, debug_callback, debug_utils_available) =
            create_instance(&entry, desc)?;

        let (physical_device, queue_family_index) = select_physical_device(&instance)?;

        let device_props = unsafe { instance.get_physical_device_properties(physical_device) };
        let device_name = unsafe { CStr::from_ptr(device_props.device_name.as_ptr()) };
        log::info!("RHI: Selected GPU: {}", device_name.to_string_lossy());

        let device = create_logical_device(&instance, physical_device, queue_family_index)?;

        // `minAccelerationStructureScratchOffsetAlignment`: the alignment the `scratch_data`
        // address passed to `vkCmdBuildAccelerationStructuresKHR` must satisfy.
        let accel_scratch_alignment = {
            let mut accel_props = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push(&mut accel_props);
            unsafe { instance.get_physical_device_properties2(physical_device, &mut props2) };
            accel_props
                .min_acceleration_structure_scratch_offset_alignment
                .max(1) as u64
        };

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
                .map_err(|e| RhiError::DeviceCreation(format!("Command pool: {e}").into()))?
        };

        let pipeline_cache = unsafe {
            device
                .create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)
                .map_err(|e| RhiError::DeviceCreation(format!("Pipeline cache: {e}").into()))?
        };

        let cmd_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY);
        let setup_command_buffer = unsafe {
            device
                .allocate_command_buffers(&cmd_alloc_info)
                .map_err(|e| RhiError::DeviceCreation(format!("Setup cmd buffer: {e}").into()))?
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
            desc.bindless,
        )?;
        let textures: SharedTextures = Rc::new(SlotTable::new(desc.bindless.textures, "texture"));
        let samplers: SharedSamplerIds = Rc::new(SlotTable::new(desc.bindless.samplers, "sampler"));
        let mut completion_type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let completion_info = vk::SemaphoreCreateInfo::default().push(&mut completion_type_info);
        let completion_semaphore = unsafe {
            device
                .create_semaphore(&completion_info, None)
                .map_err(|e| RhiError::DeviceCreation(format!("Completion timeline: {e}").into()))?
        };

        // `bufferImageGranularity` is the separation buffers and images need when they share a
        // `VkDeviceMemory`; the pool pads every suballocation to it.
        let buffer_pool: SharedBufferPool = Rc::new(RefCell::new(VulkanBufferPool::new(
            device_props.limits.buffer_image_granularity,
        )));

        let queue = Rc::new(VulkanQueue {
            queue: present_queue,
            device: device.clone(),
            swapchain_loader: swapchain_loader.clone(),
            command_pool,
            buffer_pool: buffer_pool.clone(),
            pending_commands: RefCell::new(VecDeque::new()),
            available_commands: RefCell::new(Vec::new()),
            completion_semaphore,
            next_completion_value: Cell::new(0),
            frames: RefCell::new([FrameSubmission::default(); MAX_FRAMES_IN_FLIGHT]),
            retired_resources: RetirementQueue::default(),
            textures: textures.clone(),
            samplers: samplers.clone(),
        });
        let rhi_queue = Queue {
            inner: queue.clone(),
        };

        let loaders = Rc::new(VulkanLoaders {
            device,
            descriptor_heap: descriptor_heap_loader,
            address_commands: address_commands_loader,
            mesh_shader: mesh_shader_loader,
            acceleration_structure,
            debug_labels,
        });

        Ok(Self {
            loaders,
            entry,
            instance,
            physical_device,

            queue,
            rhi_queue,
            command_pool,
            pipeline_cache,
            buffer_pool,
            device_memory_properties,
            timestamp_period: device_props.limits.timestamp_period,
            accel_scratch_alignment,
            surface_loader,
            swapchain_loader,
            debug_utils_loader,
            debug_callback,
            descriptor_heaps,
            buffer_requirements_probe: RefCell::new(Vec::new()),
            textures,
            allocations: Rc::new(RefCell::new(BTreeMap::new())),
            mapped_allocations: Rc::new(RefCell::new(BTreeMap::new())),
            samplers,
            timeline_semaphores: RefCell::new(Vec::new()),
            query_pools: RefCell::new(Vec::new()),
            setup_command_buffer,
            setup_recording: Cell::new(false),
        })
    }

    pub fn queue(&self) -> &Queue {
        &self.rhi_queue
    }

    pub fn wait_idle(&self) {
        self.queue.wait_idle();
    }

    pub fn wait_for_frame(&self, frame_index: usize) {
        self.queue.wait_for_frame(frame_index);
    }

    /// Name a Vulkan object for captures. No-op without `VK_EXT_debug_utils`.
    ///
    /// Never fails the caller: naming an object is a debugging aid, so a label with an interior
    /// NUL or a driver that refuses the call is logged and dropped rather than taking down a
    /// resource creation that otherwise succeeded.
    pub(crate) fn set_object_name<H: vk::Handle>(&self, handle: H, name: &str) {
        let Some(loader) = self.loaders.debug_labels.as_ref() else {
            return;
        };
        let Ok(name) = std::ffi::CString::new(name) else {
            log::warn!("debug label {name:?} contains an interior NUL; not naming the object");
            return;
        };
        let info = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(handle)
            .object_name(&name);
        if let Err(error) = unsafe { loader.set_debug_utils_object_name(&info) } {
            log::warn!("vkSetDebugUtilsObjectNameEXT failed: {error}");
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
            self.loaders
                .descriptor_heap
                .write_resource_descriptors(
                    &[info],
                    &[vk::HostAddressRangeEXT::default().address(dst)],
                )
                .map_err(|e| RhiError::Backend(format!("write image descriptor: {e}").into()))
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
                self.loaders
                    .device
                    .reset_command_buffer(
                        self.setup_command_buffer,
                        vk::CommandBufferResetFlags::empty(),
                    )
                    .map_err(|e| RhiError::CommandBuffer(e.into()))?;
                let begin_info = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                self.loaders
                    .device
                    .begin_command_buffer(self.setup_command_buffer, &begin_info)
                    .map_err(|e| RhiError::CommandBuffer(e.into()))?;
                self.setup_recording.set(true);
            }
            let dependency =
                vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));
            self.loaders
                .device
                .cmd_pipeline_barrier2(self.setup_command_buffer, &dependency);
        }
        Ok(())
    }

    /// Submit any batched setup barriers and wait for them.
    ///
    /// Must run before destroying an image one of them references: the setup buffer stays
    /// recording between batches, and a destroyed image invalidates the whole buffer. The wait is
    /// on this submission's timeline value, not `vkQueueWaitIdle`, so frames in flight are
    /// unaffected.
    pub(crate) fn flush_setup_barriers(&self) -> RhiResult<()> {
        if !self.setup_recording.replace(false) {
            return Ok(());
        }
        // SAFETY: the setup buffer is recording (the flag above) and belongs to this device.
        unsafe {
            self.loaders
                .device
                .end_command_buffer(self.setup_command_buffer)
                .map_err(|e| RhiError::CommandBuffer(e.into()))?;
        }
        self.queue.submit_setup_and_wait(self.setup_command_buffer)
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        let q = &self.queue;
        q.wait_idle();
        unsafe {
            self.loaders
                .device
                .destroy_semaphore(q.completion_semaphore, None);
            for semaphore in self.timeline_semaphores.get_mut().drain(..) {
                self.loaders.device.destroy_semaphore(semaphore, None);
            }
            for pool in self.query_pools.get_mut().drain(..) {
                self.loaders.device.destroy_query_pool(pool, None);
            }

            for t in self.textures.drain() {
                self.loaders.device.destroy_image_view(t.image_view, None);
                if !t.is_view {
                    self.loaders.device.destroy_image(t.image, None);
                }
            }

            for heap in [
                &self.descriptor_heaps.resource,
                &self.descriptor_heaps.sampler,
            ] {
                self.loaders.device.unmap_memory(heap.memory);
                self.loaders.device.destroy_buffer(heap.buffer, None);
                self.loaders.device.free_memory(heap.memory, None);
            }

            self.loaders
                .device
                .destroy_pipeline_cache(self.pipeline_cache, None);
            self.loaders
                .device
                .destroy_command_pool(self.command_pool, None);

            // Allocations own no Vulkan object, so ones the application never destroyed only
            // need forgetting before their blocks are freed.
            self.allocations.borrow_mut().clear();
            self.buffer_pool
                .borrow_mut()
                .destroy_all(&self.loaders.device);
            self.loaders.device.destroy_device(None);

            if let Some(ref debug_loader) = self.debug_utils_loader {
                debug_loader.destroy_debug_utils_messenger(self.debug_callback, None);
            }

            self.instance.destroy_instance(None);
        }
    }
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

/// SPIR-V's magic number, as the first word of a little-endian module.
const SPIRV_MAGIC: u32 = 0x0723_0203;

/// Reinterpret a SPIR-V blob as the `u32` words `vkCreateShaderModule` wants, borrowing when the
/// bytes are 4-byte aligned and copying only when they are not.
///
/// Length and magic are checked: `chunks_exact(4)` alone silently drops a trailing 1-3 bytes.
/// Words are read natively; a big-endian module fails the magic check rather than being swapped.
pub(crate) fn spirv_words(code: &[u8]) -> RhiResult<Cow<'_, [u32]>> {
    if code.len() < 4 || !code.len().is_multiple_of(4) {
        return Err(RhiError::ShaderCompilation(
            format!(
                "SPIR-V is {} bytes, which is not a whole number of 32-bit words",
                code.len()
            )
            .into(),
        ));
    }
    let magic = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
    if magic != SPIRV_MAGIC {
        return Err(RhiError::ShaderCompilation(
            format!(
                "not SPIR-V: first word is {magic:#010x}, expected {SPIRV_MAGIC:#010x} \
                 (big-endian modules are not supported)"
            )
            .into(),
        ));
    }
    Ok(match <[u32]>::ref_from_bytes(code) {
        Ok(words) => Cow::Borrowed(words),
        // Only reachable when the caller's slice is not 4-byte aligned.
        Err(_) => Cow::Owned(
            code.chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect(),
        ),
    })
}

pub(crate) fn buffer_memory_flags(memory: MemoryType) -> vk::MemoryPropertyFlags {
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
    fn vk_to_format_is_the_inverse_of_format_to_vk() {
        for &format in crate::types::ALL_FORMATS {
            assert_eq!(
                vk_to_format(format_to_vk(format)),
                Some(format),
                "{format:?} does not survive the round trip through vk::Format"
            );
        }
        // A format kiln cannot name reports so rather than guessing a near-match.
        assert_eq!(vk_to_format(vk::Format::R4G4_UNORM_PACK8), None);
        assert_eq!(vk_to_format(vk::Format::UNDEFINED), None);
    }

    #[test]
    fn spirv_words_rejects_malformed_blobs() {
        let magic = SPIRV_MAGIC.to_le_bytes();
        let valid = [magic.as_slice(), &[0u8; 4]].concat();
        assert!(spirv_words(&valid).is_ok());
        // `chunks_exact` alone would silently drop a trailing partial word.
        let truncated = [magic.as_slice(), &[0u8; 2]].concat();
        assert!(spirv_words(&truncated).is_err());
        assert!(spirv_words(&[]).is_err());
        assert!(spirv_words(&[0xDE, 0xAD, 0xBE, 0xEF]).is_err());
    }
}
