use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::ffi::{CStr, CString, c_char};
use std::sync::{Arc, Mutex};

use ash::{
    Device, Entry, Instance,
    ext::{debug_utils, descriptor_buffer, mesh_shader as vk_mesh_shader},
    khr::{
        acceleration_structure as vk_accel_structure, ray_tracing_pipeline as vk_rt_pipeline,
        surface, swapchain,
    },
    vk,
};

use crate::command::{
    CommandBuffer, CommandBufferInner, SignalOp, SignalValueDesc, WaitOp, WaitValueDesc,
};
use crate::device::{BindlessMode, DeviceDesc};
use crate::error::{RhiError, RhiResult};
use crate::memory::{BufferDesc, GpuBuffer, GpuBufferInner, MemoryType};
use crate::pipeline::*;
use crate::queue::{Queue, QueueInner, SubmitDesc};
use crate::sampler::{Sampler, SamplerDesc};
use crate::shader::{ShaderModule, ShaderModuleDesc, ShaderModuleInner};
use crate::surface::{Surface, SurfaceDesc, SurfaceInner};
use crate::swapchain::{AcquiredImage, Swapchain, SwapchainDesc, SwapchainInner};
use crate::sync::{TimelineSemaphore, TimelineSemaphoreInner};
use crate::texture::{Texture, TextureDesc, TextureSizeAlign};
use crate::types::*;

use super::accel::VulkanAccelerationStructure;
use super::command::VulkanCommandBuffer;
use super::memory::VulkanBuffer;
use super::pipeline::{
    VulkanComputePso, VulkanGraphicsPso, VulkanGraphicsPsoDesc, VulkanMeshletPso,
    VulkanMeshletPsoDesc, VulkanRayTracingPso,
};
use super::shader::VulkanShaderModule;
use super::surface::VulkanSurface;
use super::swapchain::VulkanSwapchain;
use super::sync::VulkanTimelineSemaphore;
use super::texture::VulkanTexture;
use crate::accel::{AccelInner, AccelerationStructure};

/// Raw Vulkan handles exposed for escape-hatch scenarios (e.g. ImGui integration).
pub struct VulkanHandles {
    pub instance: Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: Device,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub command_pool: vk::CommandPool,
}

#[derive(Clone)]
pub(crate) struct BufferAllocation {
    pub base: GpuAddress,
    pub size: u64,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub memory_type_index: u32,
    pub mapped_ptr: Option<*mut u8>,
}

// Mapped host pointers are stored only as address ranges for gpuHostToDevicePointer-style
// translation and are accessed behind the shared allocation mutex.
unsafe impl Send for BufferAllocation {}

pub(crate) struct DescriptorBufferHeap {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub mapped_ptr: *mut u8,
    pub size: u64,
    pub gpu_address: GpuAddress,
    pub layout: vk::DescriptorSetLayout,
    pub sampled_image_offset: u64,
    pub sampler_offset: u64,
    pub storage_image_offset: u64,
    pub sampled_image_stride: u64,
    pub sampler_stride: u64,
    pub storage_image_stride: u64,
}

/// Buffer allocations keyed by GPU base address, enabling O(log n) address->buffer
/// resolution instead of a linear scan on every indirect/copy/index command.
pub(crate) type SharedAllocations = Arc<Mutex<BTreeMap<u64, BufferAllocation>>>;
pub(crate) type SharedTextures = Arc<Mutex<Vec<Option<VulkanTexture>>>>;

/// Components produced by `build_swapchain_contents` (shared between create and recreate).
struct SwapchainContents {
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    extent: vk::Extent2D,
    depth_image: vk::Image,
    depth_image_view: vk::ImageView,
    depth_image_memory: vk::DeviceMemory,
    present_complete_semaphores: Vec<vk::Semaphore>,
    rendering_complete_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    in_flight_cmd_buffers: Vec<vk::CommandBuffer>,
}

/// Ray tracing pipeline properties — owned copy (no lifetime dependency).
#[derive(Clone, Copy)]
pub(crate) struct RayTracingProperties {
    pub shader_group_handle_size: u32,
    pub shader_group_handle_alignment: u32,
    pub max_ray_recursion_depth: u32,
}

/// Vulkan backend device.
pub struct VulkanDevice {
    pub(crate) entry: Entry,
    pub(crate) instance: Instance,
    pub(crate) device: Device,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) queue_family_index: u32,
    pub(crate) queue: Queue,
    pub(crate) present_queue: vk::Queue,
    pub(crate) command_pool: vk::CommandPool,
    pub(crate) device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub(crate) bindless_mode: BindlessMode,
    pub(crate) max_draw_indirect_count: u32,

    // Extension loaders
    pub(crate) surface_loader: surface::Instance,
    pub(crate) swapchain_loader: swapchain::Device,
    pub(crate) descriptor_buffer_loader: Option<descriptor_buffer::Device>,

    // Debug
    pub(crate) debug_utils_loader: Option<debug_utils::Instance>,
    pub(crate) debug_callback: vk::DebugUtilsMessengerEXT,

    // Bindless texture heap
    pub(crate) texture_descriptor_set_layout: vk::DescriptorSetLayout,
    pub(crate) descriptor_buffer_heap: Option<DescriptorBufferHeap>,
    pub(crate) textures: SharedTextures,
    pub(crate) next_texture_id: RefCell<u32>,
    pub(crate) allocations: SharedAllocations,

    // Sampler storage
    pub(crate) samplers: RefCell<Vec<vk::Sampler>>,
    pub(crate) next_sampler_id: RefCell<u32>,

    // Shader module storage
    pub(crate) shader_modules: RefCell<Vec<VulkanShaderModule>>,

    // Setup command buffer
    pub(crate) setup_command_buffer: vk::CommandBuffer,

    // Mesh shader support
    /// True when `VK_EXT_mesh_shader` was enabled at device creation.
    pub(crate) mesh_shader_supported: bool,

    // Ray tracing support (VK_KHR_ray_tracing_pipeline + VK_KHR_acceleration_structure)
    /// Present when both RT extensions were enabled. Contains the loader + pipeline properties.
    pub(crate) ray_tracing: Option<(vk_rt_pipeline::Device, RayTracingProperties)>,
    /// Present when VK_KHR_acceleration_structure was enabled.
    pub(crate) acceleration_structure: Option<vk_accel_structure::Device>,
    /// Monotonic counter for AccelerationStructureId assignment.
    pub(crate) accel_counter: RefCell<u32>,
}

/// Vulkan queue wrapper.
pub struct VulkanQueue {
    pub(crate) queue: vk::Queue,
    pub(crate) device: Device,
    pub(crate) swapchain_loader: swapchain::Device,
    pub(crate) command_pool: vk::CommandPool,
    value_sync: Mutex<HashMap<u64, VulkanValueSyncState>>,
}

#[derive(Clone, Copy)]
struct VulkanValueSyncState {
    semaphore: vk::Semaphore,
    scheduled_value: u64,
}

impl VulkanQueue {
    pub fn submit(&self, cmd: VulkanCommandBuffer) -> RhiResult<()> {
        self.submit_with_desc(cmd, &SubmitDesc::default())
    }

    fn ensure_value_sync_state<'a>(
        &self,
        value_ptr: GpuAddress,
        map: &'a mut HashMap<u64, VulkanValueSyncState>,
    ) -> RhiResult<&'a mut VulkanValueSyncState> {
        let key = value_ptr.0;
        if let std::collections::hash_map::Entry::Vacant(entry) = map.entry(key) {
            let mut type_info = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let semaphore_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
            let semaphore = unsafe {
                self.device
                    .create_semaphore(&semaphore_info, None)
                    .map_err(|e| {
                        RhiError::SyncError(format!(
                            "Failed to create Vulkan value semaphore for {:#x}: {e}",
                            key
                        ))
                    })?
            };
            entry.insert(VulkanValueSyncState {
                semaphore,
                scheduled_value: 0,
            });
        }
        Ok(map
            .get_mut(&key)
            .expect("value sync state should exist after insertion"))
    }

    fn collect_value_waits(
        &self,
        waits: &[WaitValueDesc],
    ) -> RhiResult<HashMap<vk::Semaphore, u64>> {
        let mut map = self.value_sync.lock().expect("value sync lock poisoned");
        let mut merged: HashMap<vk::Semaphore, u64> = HashMap::new();
        for wait in waits {
            let state = self.ensure_value_sync_state(wait.value_ptr, &mut map)?;
            let required = match wait.wait_op {
                WaitOp::GreaterOrEqual => wait.value,
                WaitOp::Equal => {
                    if wait.value < state.scheduled_value {
                        return Err(RhiError::SyncError(format!(
                            "Vulkan wait_before_value(Equal) on {:#x} requested {}, but value already advanced to {}",
                            wait.value_ptr.0, wait.value, state.scheduled_value
                        )));
                    }
                    wait.value
                }
                WaitOp::MaskedEqual => {
                    if wait.mask == u64::MAX {
                        if wait.value < state.scheduled_value {
                            return Err(RhiError::SyncError(format!(
                                "Vulkan wait_before_value(MaskedEqual full-mask) on {:#x} requested {}, but value already advanced to {}",
                                wait.value_ptr.0, wait.value, state.scheduled_value
                            )));
                        }
                        wait.value
                    } else {
                        let masked_current = state.scheduled_value & wait.mask;
                        let masked_target = wait.value & wait.mask;
                        if masked_current != masked_target {
                            return Err(RhiError::SyncError(format!(
                                "Vulkan wait_before_value(MaskedEqual) for {:#x} cannot be represented with timeline wait: current masked value {:#x} != target {:#x}",
                                wait.value_ptr.0, masked_current, masked_target
                            )));
                        }
                        state.scheduled_value
                    }
                }
            };
            state.scheduled_value = state.scheduled_value.max(required);
            merged
                .entry(state.semaphore)
                .and_modify(|v| *v = (*v).max(required))
                .or_insert(required);
        }
        Ok(merged)
    }

    fn collect_value_signals(
        &self,
        signals: &[SignalValueDesc],
    ) -> RhiResult<HashMap<vk::Semaphore, u64>> {
        let mut map = self.value_sync.lock().expect("value sync lock poisoned");
        let mut merged: HashMap<vk::Semaphore, u64> = HashMap::new();
        for signal in signals {
            let state = self.ensure_value_sync_state(signal.value_ptr, &mut map)?;
            let target = match signal.signal_op {
                SignalOp::AtomicSet => signal.value,
                SignalOp::AtomicMax => signal.value.max(state.scheduled_value),
                SignalOp::AtomicOr => state.scheduled_value | signal.value,
            };
            if target < state.scheduled_value {
                return Err(RhiError::SyncError(format!(
                    "Vulkan signal_after_value would decrease {:#x} from {} to {}",
                    signal.value_ptr.0, state.scheduled_value, target
                )));
            }
            state.scheduled_value = target;
            merged.insert(state.semaphore, target);
        }
        Ok(merged)
    }

    pub fn submit_with_desc(
        &self,
        cmd: VulkanCommandBuffer,
        desc: &SubmitDesc<'_>,
    ) -> RhiResult<()> {
        let mut waits = timeline_pairs(desc.wait_semaphores, "wait")?;
        waits.extend(self.collect_value_waits(&cmd.pending_value_waits)?);
        let mut signals = timeline_pairs(desc.signal_semaphores, "signal")?;
        signals.extend(self.collect_value_signals(&cmd.pending_value_signals)?);
        let wait_stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; waits.len()];
        self.submit_timeline(
            cmd.command_buffer,
            &waits,
            &wait_stages,
            &signals,
            vk::Fence::null(),
        )
    }

    /// Encode a `vkQueueSubmit` with timeline-semaphore wait/signal pairs.
    ///
    /// `wait_stages` must have the same length as `waits`. Pass `vk::Fence::null()` when
    /// no completion fence is needed.
    fn submit_timeline(
        &self,
        cmd: vk::CommandBuffer,
        waits: &[(vk::Semaphore, u64)],
        wait_stages: &[vk::PipelineStageFlags],
        signals: &[(vk::Semaphore, u64)],
        fence: vk::Fence,
    ) -> RhiResult<()> {
        let command_buffers = [cmd];
        let wait_semaphores: Vec<vk::Semaphore> = waits.iter().map(|(s, _)| *s).collect();
        let wait_values: Vec<u64> = waits.iter().map(|(_, v)| *v).collect();
        let signal_semaphores: Vec<vk::Semaphore> = signals.iter().map(|(s, _)| *s).collect();
        let signal_values: Vec<u64> = signals.iter().map(|(_, v)| *v).collect();
        let mut submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);
        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(&signal_values);
        if !wait_values.is_empty() || !signal_values.is_empty() {
            submit_info = submit_info.push_next(&mut timeline_info);
        }
        unsafe {
            self.device
                .queue_submit(self.queue, &[submit_info], fence)
                .map_err(|e| RhiError::QueueSubmit(e.to_string()))?;
        }
        Ok(())
    }

    pub fn acquire_image(
        &self,
        sc: &VulkanSwapchain,
        frame_index: usize,
    ) -> RhiResult<AcquiredImage> {
        unsafe {
            // Wait for this frame's fence
            let fence = sc.in_flight_fences[frame_index];
            self.device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|e| RhiError::SyncError(e.to_string()))?;
            self.device
                .reset_fences(&[fence])
                .map_err(|e| RhiError::SyncError(e.to_string()))?;

            // Reclaim the previous command buffer used for this frame.
            {
                let mut cmd_buffers = sc.in_flight_cmd_buffers.borrow_mut();
                if let Some(prev) = cmd_buffers.get_mut(frame_index)
                    && *prev != vk::CommandBuffer::null()
                {
                    self.device
                        .free_command_buffers(self.command_pool, std::slice::from_ref(prev));
                    *prev = vk::CommandBuffer::null();
                }
            }

            let semaphore = sc.present_complete_semaphores[frame_index];
            let (image_index, _suboptimal) = self
                .swapchain_loader
                .acquire_next_image(sc.swapchain, u64::MAX, semaphore, vk::Fence::null())
                .map_err(|e| match e {
                    vk::Result::ERROR_OUT_OF_DATE_KHR => RhiError::SwapchainOutOfDate,
                    _ => RhiError::SwapchainCreation(e.to_string()),
                })?;

            Ok(AcquiredImage {
                index: image_index,
                format: sc.format,
                width: sc.extent.width,
                height: sc.extent.height,
            })
        }
    }

    pub fn present(
        &self,
        sc: &VulkanSwapchain,
        image_index: u32,
        _frame_index: usize,
    ) -> RhiResult<()> {
        let wait_semaphores = [sc.rendering_complete_semaphores[image_index as usize]];
        let swapchains = [sc.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe {
            self.swapchain_loader
                .queue_present(self.queue, &present_info)
                .map_err(|e| match e {
                    vk::Result::ERROR_OUT_OF_DATE_KHR => RhiError::SwapchainOutOfDate,
                    _ => RhiError::PresentFailed(e.to_string()),
                })?;
        }
        Ok(())
    }

    pub fn submit_frame(
        &self,
        cmd: super::command::VulkanCommandBuffer,
        sc: &super::swapchain::VulkanSwapchain,
        frame_index: usize,
        image_index: u32,
    ) -> RhiResult<()> {
        // Seed with the swapchain's acquire→render→present semaphores, then append the
        // command buffer's pending value sync. Value waits use ALL_COMMANDS; the acquire
        // wait only needs to gate the color attachment write.
        let mut waits = vec![(sc.present_complete_semaphores[frame_index], 0u64)];
        let mut wait_stages = vec![vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        for (sem, val) in self.collect_value_waits(&cmd.pending_value_waits)? {
            waits.push((sem, val));
            wait_stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
        }
        let mut signals = vec![(sc.rendering_complete_semaphores[image_index as usize], 0u64)];
        signals.extend(self.collect_value_signals(&cmd.pending_value_signals)?);
        let fence = sc.in_flight_fences[frame_index];
        let raw_cmd = cmd.command_buffer;
        self.submit_timeline(raw_cmd, &waits, &wait_stages, &signals, fence)?;

        // Track the command buffer so it can be freed after the fence signals.
        if let Some(slot) = sc.in_flight_cmd_buffers.borrow_mut().get_mut(frame_index) {
            *slot = raw_cmd;
        }
        Ok(())
    }

    pub fn wait_idle(&self) {
        unsafe {
            let _ = self.device.queue_wait_idle(self.queue);
        }
    }
}

/// Unwrap a slice of `(TimelineSemaphore, u64)` into `(vk::Semaphore, u64)` pairs,
/// erroring out if any handle is from a non-Vulkan backend.
fn timeline_pairs(
    pairs: &[(TimelineSemaphore, u64)],
    kind: &'static str,
) -> RhiResult<Vec<(vk::Semaphore, u64)>> {
    pairs
        .iter()
        .map(|(sem, value)| match &sem.inner {
            TimelineSemaphoreInner::Vulkan(vk_semaphore) => Ok((vk_semaphore.semaphore, *value)),
            #[allow(unreachable_patterns)]
            _ => Err(RhiError::SyncError(format!(
                "Timeline {kind} semaphore backend mismatch on Vulkan queue submit"
            ))),
        })
        .collect()
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
fn find_memorytype_index(
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
        Format::R32G32B32Float => vk::Format::R32G32B32_SFLOAT,
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

const MAX_BINDLESS_STORAGE_IMAGES: u32 = 1_000_000;

impl VulkanDevice {
    pub fn new(desc: &DeviceDesc) -> RhiResult<Self> {
        let entry = unsafe { Entry::load() }
            .map_err(|e| RhiError::DeviceCreation(format!("Failed to load Vulkan: {e}")))?;

        let app_name = CString::new("Spectradio").unwrap();

        // Validation layers
        let mut layer_names_raw: Vec<*const c_char> = Vec::new();
        let layer_name_validation = c"VK_LAYER_KHRONOS_validation";
        if desc.validation {
            layer_names_raw.push(layer_name_validation.as_ptr());
        }

        // Instance extensions
        // Note: We don't have a window handle here, so surface extensions
        // will be added when creating the surface. For now, add debug + portability.
        let mut extension_names: Vec<*const c_char> = Vec::new();
        if desc.validation {
            extension_names.push(debug_utils::NAME.as_ptr());
        }

        // macOS portability
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            extension_names.push(ash::khr::portability_enumeration::NAME.as_ptr());
            extension_names.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
        }

        // We also need surface extensions for swapchain -- add the common ones
        extension_names.push(ash::khr::surface::NAME.as_ptr());

        #[cfg(target_os = "macos")]
        {
            // macOS Vulkan surface extension.
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
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(&app_name)
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

        // Debug callback
        let (debug_utils_loader, debug_callback) = if desc.validation {
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

            let loader = debug_utils::Instance::new(&entry, &instance);
            let callback = unsafe {
                loader
                    .create_debug_utils_messenger(&debug_info, None)
                    .map_err(|e| RhiError::DeviceCreation(format!("Debug callback: {e}")))?
            };
            (Some(loader), callback)
        } else {
            (None, vk::DebugUtilsMessengerEXT::null())
        };

        // Physical device selection
        let physical_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .map_err(|e| RhiError::DeviceCreation(format!("Enumerate devices: {e}")))?
        };

        let (physical_device, queue_family_index) = physical_devices
            .iter()
            .find_map(|pdevice| {
                let props = unsafe { instance.get_physical_device_properties(*pdevice) };
                let api_version = props.api_version;
                let supports_vulkan_13 = vk::api_version_major(api_version) > 1
                    || (vk::api_version_major(api_version) == 1
                        && vk::api_version_minor(api_version) >= 3);
                if !supports_vulkan_13 {
                    return None;
                }

                unsafe { instance.get_physical_device_queue_family_properties(*pdevice) }
                    .iter()
                    .enumerate()
                    .find_map(|(index, info)| {
                        if info
                            .queue_flags
                            .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                        {
                            Some((*pdevice, index as u32))
                        } else {
                            None
                        }
                    })
            })
            .ok_or(RhiError::NoSuitableGpu)?;

        // Log selected device
        let device_props = unsafe { instance.get_physical_device_properties(physical_device) };
        let device_name = unsafe {
            std::ffi::CStr::from_ptr(device_props.device_name.as_ptr())
                .to_string_lossy()
                .to_string()
        };
        log::info!("RHI: Selected GPU: {}", device_name);

        // Device extension support
        let device_extension_props = unsafe {
            instance
                .enumerate_device_extension_properties(physical_device)
                .map_err(|e| {
                    RhiError::DeviceCreation(format!("Enumerate device extensions: {e}"))
                })?
        };
        let has_ext = |needle: &[u8]| {
            device_extension_props.iter().any(|ext| {
                let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                name.to_bytes() == needle
            })
        };
        let supports_descriptor_buffer = has_ext(b"VK_EXT_descriptor_buffer");
        let supports_mesh_shader = has_ext(b"VK_EXT_mesh_shader");
        // RT requires all three extensions together.
        let supports_rt = has_ext(b"VK_KHR_acceleration_structure")
            && has_ext(b"VK_KHR_ray_tracing_pipeline")
            && has_ext(b"VK_KHR_deferred_host_operations");
        log::info!(
            "RHI: Optional extensions — mesh_shader={supports_mesh_shader} ray_tracing={supports_rt}"
        );

        if desc.bindless_mode == Some(BindlessMode::ArgumentTable) {
            return Err(RhiError::Unsupported(
                "Vulkan does not support Metal argument tables".into(),
            ));
        }
        if !supports_descriptor_buffer {
            return Err(RhiError::Unsupported(
                "Vulkan descriptor buffer is required but not supported".into(),
            ));
        }
        let bindless_mode = BindlessMode::DescriptorBuffer;

        // Device extensions
        let mut device_extension_names: Vec<*const c_char> = vec![swapchain::NAME.as_ptr()];

        device_extension_names.push(descriptor_buffer::NAME.as_ptr());
        if supports_mesh_shader {
            device_extension_names.push(vk_mesh_shader::NAME.as_ptr());
        }
        if supports_rt {
            device_extension_names.push(vk_accel_structure::NAME.as_ptr());
            device_extension_names.push(vk_rt_pipeline::NAME.as_ptr());
            device_extension_names.push(ash::khr::deferred_host_operations::NAME.as_ptr());
        }

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            device_extension_names.push(ash::khr::portability_subset::NAME.as_ptr());
        }

        // All required features that were promoted to Vulkan 1.2/1.3 core go through the
        // consolidated PhysicalDeviceVulkan1{2,3}Features structs — no separate per-feature
        // structs needed since we require Vulkan 1.3.
        let mut vulkan12_features = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .timeline_semaphore(true)
            .draw_indirect_count(true);
        let mut vulkan13_features = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);

        let mut descriptor_buffer_features =
            vk::PhysicalDeviceDescriptorBufferFeaturesEXT::default().descriptor_buffer(true);

        // Optional: mesh shader and ray tracing feature structs (declared unconditionally
        // so they outlive `features2`, then conditionally chained below).
        let mut mesh_shader_features = vk::PhysicalDeviceMeshShaderFeaturesEXT::default()
            .mesh_shader(true)
            .task_shader(true);
        let mut accel_structure_features =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
                .acceleration_structure(true);
        let mut rt_pipeline_features =
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default().ray_tracing_pipeline(true);

        let features = vk::PhysicalDeviceFeatures {
            shader_clip_distance: 1,
            fill_mode_non_solid: 1,
            multi_draw_indirect: 1,
            ..Default::default()
        };

        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .features(features)
            .push_next(&mut vulkan12_features)
            .push_next(&mut vulkan13_features)
            .push_next(&mut descriptor_buffer_features);

        if supports_mesh_shader {
            let _ = features2.push_next(&mut mesh_shader_features);
        }
        if supports_rt {
            let _ = features2.push_next(&mut accel_structure_features);
            let _ = features2.push_next(&mut rt_pipeline_features);
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

        let present_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // Extension loaders
        let surface_loader = surface::Instance::new(&entry, &instance);
        let swapchain_loader = swapchain::Device::new(&instance, &device);
        // Note: dynamic_rendering, buffer_device_address, and synchronization2 are Vulkan 1.3 core
        // — their functionality is available directly on `ash::Device` without a loader.

        let device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        // Command pool
        let pool_create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_index);
        let command_pool = unsafe {
            device
                .create_command_pool(&pool_create_info, None)
                .map_err(|e| RhiError::DeviceCreation(format!("Command pool: {e}")))?
        };

        // Setup command buffer
        let cmd_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY);
        let setup_command_buffer = unsafe {
            device
                .allocate_command_buffers(&cmd_alloc_info)
                .map_err(|e| RhiError::DeviceCreation(format!("Setup cmd buffer: {e}")))?
        }[0];

        let descriptor_buffer_loader = Some(descriptor_buffer::Device::new(&instance, &device));

        // Acceleration structure + RT pipeline loaders
        let acceleration_structure_opt = if supports_rt {
            Some(vk_accel_structure::Device::new(&instance, &device))
        } else {
            None
        };
        let ray_tracing_opt = if supports_rt {
            let rt_loader = vk_rt_pipeline::Device::new(&instance, &device);
            // Query RT pipeline properties for shader group handle size etc.
            let mut rt_props = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut rt_props);
            unsafe { instance.get_physical_device_properties2(physical_device, &mut props2) };
            Some((
                rt_loader,
                RayTracingProperties {
                    shader_group_handle_size: rt_props.shader_group_handle_size,
                    shader_group_handle_alignment: rt_props.shader_group_handle_alignment,
                    max_ray_recursion_depth: rt_props.max_ray_recursion_depth,
                },
            ))
        } else {
            None
        };
        // Create bindless heap layout + storage
        let (texture_descriptor_set_layout, descriptor_buffer_heap) = {
            let loader = descriptor_buffer_loader
                .as_ref()
                .expect("descriptor buffer loader missing");
            let heap = create_descriptor_buffer_heap(
                &instance,
                &device,
                physical_device,
                &device_memory_properties,
                loader,
            )?;
            (heap.layout, Some(heap))
        };

        let queue = Queue {
            inner: QueueInner::Vulkan(Box::new(VulkanQueue {
                queue: present_queue,
                device: device.clone(),
                swapchain_loader: swapchain::Device::new(&instance, &device),
                command_pool,
                value_sync: Mutex::new(HashMap::new()),
            })),
        };

        Ok(Self {
            entry,
            instance,
            device,
            physical_device,
            queue_family_index,
            queue,
            present_queue,
            command_pool,
            device_memory_properties,
            bindless_mode,
            max_draw_indirect_count: device_props.limits.max_draw_indirect_count,
            surface_loader,
            swapchain_loader,
            descriptor_buffer_loader,
            debug_utils_loader,
            debug_callback,
            texture_descriptor_set_layout,
            descriptor_buffer_heap,
            textures: Arc::new(Mutex::new(Vec::new())),
            next_texture_id: RefCell::new(0),
            allocations: Arc::new(Mutex::new(BTreeMap::new())),
            samplers: RefCell::new(Vec::new()),
            next_sampler_id: RefCell::new(0),
            shader_modules: RefCell::new(Vec::new()),
            setup_command_buffer,
            mesh_shader_supported: supports_mesh_shader,
            ray_tracing: ray_tracing_opt,
            acceleration_structure: acceleration_structure_opt,
            accel_counter: RefCell::new(0),
        })
    }

    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    pub fn bindless_mode(&self) -> BindlessMode {
        self.bindless_mode
    }

    pub fn wait_idle(&self) {
        unsafe {
            let _ = self.device.device_wait_idle();
        }
    }

    pub fn wait_for_frame(&self, _frame_index: usize) {
        // Frame fence waiting is handled in acquire_image
    }

    /// Get raw Vulkan handles for escape-hatch scenarios (e.g. ImGui).
    pub fn vulkan_handles(&self) -> VulkanHandles {
        VulkanHandles {
            instance: self.instance.clone(),
            physical_device: self.physical_device,
            device: self.device.clone(),
            queue: self.present_queue,
            queue_family_index: self.queue_family_index,
            command_pool: self.command_pool,
        }
    }

    // -- Surface --

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
            inner: SurfaceInner::Vulkan(VulkanSurface { surface }),
        })
    }

    // -- Swapchain --

    pub fn create_swapchain(
        &self,
        surface: &Surface,
        desc: &SwapchainDesc,
    ) -> RhiResult<Swapchain> {
        let vk_surface = match &surface.inner {
            SurfaceInner::Vulkan(s) => s.surface,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        };

        let surface_formats = unsafe {
            self.surface_loader
                .get_physical_device_surface_formats(self.physical_device, vk_surface)
                .map_err(|e| RhiError::SwapchainCreation(e.to_string()))?
        };
        let desired_vk_format = format_to_vk(desc.format);
        let surface_format = surface_formats
            .iter()
            .find(|f| f.format == desired_vk_format)
            .cloned()
            .unwrap_or(surface_formats[0]);

        let SwapchainContents {
            swapchain,
            images,
            image_views,
            extent,
            depth_image,
            depth_image_view,
            depth_image_memory,
            present_complete_semaphores,
            rendering_complete_semaphores,
            in_flight_fences,
            in_flight_cmd_buffers,
        } = self.build_swapchain_contents(vk_surface, surface_format, desc, vk::SwapchainKHR::null())?;

        Ok(Swapchain {
            inner: SwapchainInner::Vulkan(VulkanSwapchain {
                swapchain,
                surface: vk_surface,
                images,
                image_views,
                format: vk_to_format(surface_format.format),
                surface_format,
                extent,
                depth_image,
                depth_image_view,
                depth_image_memory,
                present_complete_semaphores,
                rendering_complete_semaphores,
                in_flight_fences,
                in_flight_cmd_buffers: RefCell::new(in_flight_cmd_buffers),
            }),
        })
    }

    pub fn recreate_swapchain(
        &self,
        swapchain: &mut Swapchain,
        desc: &SwapchainDesc,
    ) -> RhiResult<()> {
        unsafe {
            self.device
                .device_wait_idle()
                .map_err(|e| RhiError::Backend(e.to_string()))?
        };

        let sc = match &mut swapchain.inner {
            SwapchainInner::Vulkan(sc) => sc,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        };

        let old_swapchain = sc.swapchain;
        let surface = sc.surface;
        let surface_format = sc.surface_format;

        // Tear down old image views, depth buffer, in-flight command buffers, and sync objects.
        unsafe {
            self.device.free_memory(sc.depth_image_memory, None);
            self.device.destroy_image_view(sc.depth_image_view, None);
            self.device.destroy_image(sc.depth_image, None);
            for &view in &sc.image_views {
                self.device.destroy_image_view(view, None);
            }
        }
        {
            let mut cmd_buffers = sc.in_flight_cmd_buffers.borrow_mut();
            let to_free: Vec<_> = cmd_buffers
                .iter()
                .copied()
                .filter(|c| *c != vk::CommandBuffer::null())
                .collect();
            if !to_free.is_empty() {
                unsafe {
                    self.device
                        .free_command_buffers(self.command_pool, &to_free);
                }
            }
            cmd_buffers.clear();
        }
        unsafe {
            for &sem in &sc.present_complete_semaphores {
                self.device.destroy_semaphore(sem, None);
            }
            for &sem in &sc.rendering_complete_semaphores {
                self.device.destroy_semaphore(sem, None);
            }
            for &fence in &sc.in_flight_fences {
                self.device.destroy_fence(fence, None);
            }
        }

        let contents =
            self.build_swapchain_contents(surface, surface_format, desc, old_swapchain)?;

        unsafe {
            self.swapchain_loader.destroy_swapchain(old_swapchain, None);
        }

        sc.swapchain = contents.swapchain;
        sc.images = contents.images;
        sc.image_views = contents.image_views;
        sc.extent = contents.extent;
        sc.depth_image = contents.depth_image;
        sc.depth_image_view = contents.depth_image_view;
        sc.depth_image_memory = contents.depth_image_memory;
        sc.present_complete_semaphores = contents.present_complete_semaphores;
        sc.rendering_complete_semaphores = contents.rendering_complete_semaphores;
        sc.in_flight_fences = contents.in_flight_fences;
        sc.in_flight_cmd_buffers = RefCell::new(contents.in_flight_cmd_buffers);

        Ok(())
    }

    fn build_swapchain_contents(
        &self,
        surface: vk::SurfaceKHR,
        surface_format: vk::SurfaceFormatKHR,
        desc: &SwapchainDesc,
        old_swapchain: vk::SwapchainKHR,
    ) -> RhiResult<SwapchainContents> {
        let caps = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, surface)
                .map_err(|e| RhiError::SwapchainCreation(e.to_string()))?
        };

        let mut image_count = desc.image_count.max(caps.min_image_count);
        if caps.max_image_count > 0 {
            image_count = image_count.min(caps.max_image_count);
        }

        let extent = if caps.current_extent.width == u32::MAX {
            vk::Extent2D {
                width: desc.width,
                height: desc.height,
            }
        } else {
            caps.current_extent
        };

        let pre_transform = if caps
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            caps.current_transform
        };

        let present_mode = if desc.vsync {
            vk::PresentModeKHR::FIFO
        } else {
            unsafe {
                self.surface_loader
                    .get_physical_device_surface_present_modes(self.physical_device, surface)
                    .map_err(|e| RhiError::SwapchainCreation(e.to_string()))?
            }
            .into_iter()
            .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO)
        };

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_color_space(surface_format.color_space)
            .image_format(surface_format.format)
            .image_extent(extent)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(pre_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .image_array_layers(1)
            .old_swapchain(old_swapchain);

        let swapchain = unsafe {
            self.swapchain_loader
                .create_swapchain(&create_info, None)
                .map_err(|e| RhiError::SwapchainCreation(e.to_string()))?
        };
        let images = unsafe {
            self.swapchain_loader
                .get_swapchain_images(swapchain)
                .map_err(|e| RhiError::SwapchainCreation(e.to_string()))?
        };
        let image_views = self.create_swapchain_image_views(&images, surface_format.format)?;
        let (depth_image, depth_image_view, depth_image_memory) =
            self.create_depth_buffer(extent.width, extent.height)?;

        let sem_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let mk_sem = || unsafe {
            self.device
                .create_semaphore(&sem_info, None)
                .map_err(|e| RhiError::SwapchainCreation(e.to_string()))
        };
        let mk_fence = || unsafe {
            self.device
                .create_fence(&fence_info, None)
                .map_err(|e| RhiError::SwapchainCreation(e.to_string()))
        };

        let mut present_complete_semaphores = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut in_flight_fences = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            present_complete_semaphores.push(mk_sem()?);
            in_flight_fences.push(mk_fence()?);
        }
        let mut rendering_complete_semaphores = Vec::with_capacity(images.len());
        for _ in 0..images.len() {
            rendering_complete_semaphores.push(mk_sem()?);
        }

        Ok(SwapchainContents {
            swapchain,
            images,
            image_views,
            extent,
            depth_image,
            depth_image_view,
            depth_image_memory,
            present_complete_semaphores,
            rendering_complete_semaphores,
            in_flight_fences,
            in_flight_cmd_buffers: vec![vk::CommandBuffer::null(); MAX_FRAMES_IN_FLIGHT],
        })
    }

    // -- Buffer --

    pub fn create_buffer(&self, desc: &BufferDesc) -> RhiResult<GpuBuffer> {
        let usage_flags = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::INDIRECT_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;

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

        let mem_flags = match desc.memory {
            MemoryType::GpuOnly => vk::MemoryPropertyFlags::DEVICE_LOCAL,
            MemoryType::Default => {
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
            }
            MemoryType::Readback => {
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_CACHED
            }
        };

        let mem_type_index =
            find_memorytype_index(&mem_requirements, &self.device_memory_properties, mem_flags)
                .ok_or_else(|| RhiError::AllocationFailed("No suitable memory type".into()))?;

        let mut alloc_flags_info =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index)
            .push_next(&mut alloc_flags_info);

        let memory = unsafe {
            self.device
                .allocate_memory(&alloc_info, None)
                .map_err(|e| RhiError::AllocationFailed(e.to_string()))?
        };

        unsafe {
            self.device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(|e| RhiError::BufferCreation(e.to_string()))?;
        }

        // Get GPU address
        let addr_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        let gpu_addr = unsafe { self.device.get_buffer_device_address(&addr_info) };

        // Map if host-visible
        let mapped_ptr = match desc.memory {
            MemoryType::Default | MemoryType::Readback => {
                let ptr = unsafe {
                    self.device
                        .map_memory(memory, 0, desc.size, vk::MemoryMapFlags::empty())
                        .map_err(|e| RhiError::AllocationFailed(e.to_string()))?
                };
                Some(ptr as *mut u8)
            }
            MemoryType::GpuOnly => None,
        };

        let vk_buffer = VulkanBuffer {
            buffer,
            memory,
            size: desc.size,
            mapped_ptr,
            gpu_address: GpuAddress(gpu_addr),
        };

        {
            let mut allocations = self.allocations.lock().expect("allocations lock poisoned");
            allocations.insert(
                vk_buffer.gpu_address.0,
                BufferAllocation {
                    base: vk_buffer.gpu_address,
                    size: vk_buffer.size,
                    buffer: vk_buffer.buffer,
                    memory: vk_buffer.memory,
                    memory_type_index: mem_type_index,
                    mapped_ptr: vk_buffer.mapped_ptr,
                },
            );
        }

        Ok(GpuBuffer {
            inner: GpuBufferInner::Vulkan(vk_buffer),
        })
    }

    pub fn host_to_device_pointer(&self, cpu_ptr: *const u8) -> Option<GpuAddress> {
        if cpu_ptr.is_null() {
            return None;
        }
        let ptr = cpu_ptr as usize;
        let allocations = self.allocations.lock().expect("allocations lock poisoned");
        for alloc in allocations.values() {
            if let Some(mapped) = alloc.mapped_ptr {
                let base = mapped as usize;
                let end = base + alloc.size as usize;
                if ptr >= base && ptr < end {
                    let offset = (ptr - base) as u64;
                    return Some(GpuAddress(alloc.base.0 + offset));
                }
            }
        }
        None
    }

    // -- Texture --

    pub fn texture_size_align(&self, desc: &TextureDesc) -> RhiResult<TextureSizeAlign> {
        // Create a throwaway image just to query the driver's size/alignment requirements.
        let (image, _, _) = self.create_image_for_desc(desc)?;
        let mem_reqs = unsafe { self.device.get_image_memory_requirements(image) };
        unsafe {
            self.device.destroy_image(image, None);
        }
        Ok(TextureSizeAlign {
            size: mem_reqs.size,
            align: mem_reqs.alignment,
        })
    }

    /// Build the `vk::ImageCreateInfo` for `desc` and create the image.
    /// Returns the image plus the effective `array_layers` (cube → ×6) and the format.
    fn create_image_for_desc(
        &self,
        desc: &TextureDesc,
    ) -> RhiResult<(vk::Image, u32, vk::Format)> {
        let vk_format = format_to_vk(desc.format);

        let mut usage = vk::ImageUsageFlags::empty();
        use crate::texture::TextureUsage;
        let pairs = [
            (TextureUsage::SAMPLED, vk::ImageUsageFlags::SAMPLED),
            (TextureUsage::STORAGE, vk::ImageUsageFlags::STORAGE),
            (
                TextureUsage::COLOR_ATTACHMENT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ),
            (
                TextureUsage::DEPTH_STENCIL_ATTACHMENT,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ),
            (TextureUsage::TRANSFER_SRC, vk::ImageUsageFlags::TRANSFER_SRC),
            (TextureUsage::TRANSFER_DST, vk::ImageUsageFlags::TRANSFER_DST),
        ];
        for (flag, vk_flag) in pairs {
            if desc.usage.contains(flag) {
                usage |= vk_flag;
            }
        }

        let image_type = match desc.dimension {
            TextureDimension::D1 => vk::ImageType::TYPE_1D,
            TextureDimension::D2
            | TextureDimension::D2Array
            | TextureDimension::Cube
            | TextureDimension::CubeArray => vk::ImageType::TYPE_2D,
            TextureDimension::D3 => vk::ImageType::TYPE_3D,
        };
        let samples = match desc.sample_count {
            SampleCount::S1 => vk::SampleCountFlags::TYPE_1,
            SampleCount::S2 => vk::SampleCountFlags::TYPE_2,
            SampleCount::S4 => vk::SampleCountFlags::TYPE_4,
            SampleCount::S8 => vk::SampleCountFlags::TYPE_8,
            SampleCount::S16 => vk::SampleCountFlags::TYPE_16,
        };
        // A single cubemap = 6 faces; n cubes = n × 6. CubeArray callers supply the total.
        let array_layers = match desc.dimension {
            TextureDimension::Cube => desc.array_layers * 6,
            _ => desc.array_layers,
        };
        let image_flags = match desc.dimension {
            TextureDimension::Cube | TextureDimension::CubeArray => {
                vk::ImageCreateFlags::CUBE_COMPATIBLE
            }
            _ => vk::ImageCreateFlags::empty(),
        };

        let image_info = vk::ImageCreateInfo::default()
            .flags(image_flags)
            .image_type(image_type)
            .format(vk_format)
            .extent(vk::Extent3D {
                width: desc.width,
                height: desc.height,
                depth: desc.depth,
            })
            .mip_levels(desc.mip_levels)
            .array_layers(array_layers)
            .samples(samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe {
            self.device
                .create_image(&image_info, None)
                .map_err(|e| RhiError::TextureCreation(e.to_string()))?
        };
        Ok((image, array_layers, vk_format))
    }

    pub fn create_texture(
        &self,
        desc: &TextureDesc,
        texture_gpu: GpuAddress,
    ) -> RhiResult<Texture> {
        if texture_gpu.is_null() {
            return Err(RhiError::TextureCreation(
                "create_texture requires a non-null texture allocation address".into(),
            ));
        }

        let (image, array_layers, vk_format) = self.create_image_for_desc(desc)?;
        let mem_reqs = unsafe { self.device.get_image_memory_requirements(image) };

        // Resolve the caller-supplied GPU address against the allocation registry.
        // Any validation failure here is fatal — destroy the partially-built image first.
        let resolve = || -> RhiResult<(vk::DeviceMemory, u64)> {
            let allocations = self.allocations.lock().expect("allocations lock poisoned");
            let alloc = allocations
                .range(..=texture_gpu.0)
                .next_back()
                .map(|(_, alloc)| alloc)
                .filter(|alloc| texture_gpu.0 < alloc.base.0 + alloc.size)
                .ok_or_else(|| {
                    RhiError::TextureCreation(format!(
                        "texture allocation address 0x{:x} was not returned by gpuMalloc",
                        texture_gpu.0
                    ))
                })?;
            let offset = texture_gpu.0 - alloc.base.0;
            if !offset.is_multiple_of(mem_reqs.alignment) {
                return Err(RhiError::TextureCreation(format!(
                    "texture allocation address 0x{:x} has memory offset {offset}, expected alignment {}",
                    texture_gpu.0, mem_reqs.alignment
                )));
            }
            if offset + mem_reqs.size > alloc.size {
                return Err(RhiError::TextureCreation(format!(
                    "texture allocation address 0x{:x} has {} bytes available, needs {}",
                    texture_gpu.0,
                    alloc.size - offset,
                    mem_reqs.size
                )));
            }
            if mem_reqs.memory_type_bits & (1 << alloc.memory_type_index) == 0 {
                return Err(RhiError::TextureCreation(
                    "texture allocation memory type is not compatible with this image".into(),
                ));
            }
            Ok((alloc.memory, offset))
        };
        let (memory, memory_offset) = match resolve() {
            Ok(v) => v,
            Err(e) => {
                unsafe { self.device.destroy_image(image, None) };
                return Err(e);
            }
        };

        unsafe {
            self.device
                .bind_image_memory(image, memory, memory_offset)
                .map_err(|e| RhiError::TextureCreation(e.to_string()))?;
        }

        // Create image view
        let view_type = match desc.dimension {
            TextureDimension::D1 => vk::ImageViewType::TYPE_1D,
            TextureDimension::D2 => vk::ImageViewType::TYPE_2D,
            TextureDimension::D2Array => vk::ImageViewType::TYPE_2D_ARRAY,
            TextureDimension::D3 => vk::ImageViewType::TYPE_3D,
            TextureDimension::Cube => vk::ImageViewType::CUBE,
            TextureDimension::CubeArray => vk::ImageViewType::CUBE_ARRAY,
        };

        let aspect = if is_depth_format(desc.format) {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(view_type)
            .format(vk_format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: desc.mip_levels,
                base_array_layer: 0,
                layer_count: array_layers,
            });

        let image_view = unsafe {
            self.device
                .create_image_view(&view_info, None)
                .map_err(|e| RhiError::TextureCreation(e.to_string()))?
        };

        // Allocate texture ID from RefCell
        let texture_id = {
            let mut id = self.next_texture_id.borrow_mut();
            let tid = TextureId(*id);
            *id += 1;
            tid
        };

        // Transition to unified GENERAL layout before first use.
        self.transition_image_to_general(image, aspect, desc.mip_levels, array_layers)?;

        if desc.usage.contains(crate::texture::TextureUsage::SAMPLED) {
            self.write_image_descriptor(texture_id, image_view, vk::ImageLayout::GENERAL, false)?;
        }
        if desc.usage.contains(crate::texture::TextureUsage::STORAGE) {
            self.write_image_descriptor(texture_id, image_view, vk::ImageLayout::GENERAL, true)?;
        }

        // Store texture data
        let vk_texture = VulkanTexture {
            image,
            image_view,
            is_view: false,
        };

        {
            let mut textures = self.textures.lock().expect("textures lock poisoned");
            if textures.len() <= texture_id.0 as usize {
                textures.resize_with(texture_id.0 as usize + 1, || None);
            }
            textures[texture_id.0 as usize] = Some(vk_texture);
        }

        Ok(Texture {
            id: texture_id,
            gpu_address: texture_gpu,
            desc: desc.clone(),
        })
    }

    // -- Sampler --

    pub fn create_sampler(&self, desc: &SamplerDesc) -> RhiResult<Sampler> {
        let mag_filter = match desc.mag_filter {
            FilterMode::Nearest => vk::Filter::NEAREST,
            FilterMode::Linear => vk::Filter::LINEAR,
        };
        let min_filter = match desc.min_filter {
            FilterMode::Nearest => vk::Filter::NEAREST,
            FilterMode::Linear => vk::Filter::LINEAR,
        };
        let mip_mode = match desc.mip_filter {
            FilterMode::Nearest => vk::SamplerMipmapMode::NEAREST,
            FilterMode::Linear => vk::SamplerMipmapMode::LINEAR,
        };
        let address_u = address_mode_to_vk(desc.address_u);
        let address_v = address_mode_to_vk(desc.address_v);
        let address_w = address_mode_to_vk(desc.address_w);

        let mut sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(mag_filter)
            .min_filter(min_filter)
            .mipmap_mode(mip_mode)
            .address_mode_u(address_u)
            .address_mode_v(address_v)
            .address_mode_w(address_w)
            .mip_lod_bias(desc.mip_lod_bias)
            .min_lod(desc.min_lod)
            .max_lod(desc.max_lod);

        if let Some(max_aniso) = desc.max_anisotropy {
            sampler_info = sampler_info
                .anisotropy_enable(true)
                .max_anisotropy(max_aniso);
        }

        if let Some(compare) = desc.compare {
            sampler_info = sampler_info
                .compare_enable(true)
                .compare_op(compare_op_to_vk(compare));
        }

        let sampler = unsafe {
            self.device
                .create_sampler(&sampler_info, None)
                .map_err(|e| RhiError::Backend(format!("Sampler creation: {e}")))?
        };

        let id = {
            let mut sid = self.next_sampler_id.borrow_mut();
            let id = SamplerId(*sid);
            *sid += 1;
            id
        };

        self.write_sampler_descriptor(id, sampler)?;

        self.samplers.borrow_mut().push(sampler);

        Ok(Sampler { id })
    }

    // -- Shader --

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

        let vk_module = VulkanShaderModule {
            module,
            entry_point,
        };

        self.shader_modules.borrow_mut().push(vk_module);

        Ok(ShaderModule {
            inner: ShaderModuleInner::Vulkan(VulkanShaderModule {
                module,
                entry_point: CString::new(desc.entry_point).unwrap(),
            }),
            stage: desc.stage,
        })
    }

    // -- Pipeline --

    pub fn create_graphics_pso(&self, desc: &GraphicsPsoDesc) -> RhiResult<GraphicsPso> {
        // Get shader modules
        let modules = self.shader_modules.borrow();
        let vert_module = &modules[desc.vertex_shader];
        let frag_module = &modules[desc.pixel_shader];

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
            stencil_format: desc
                .stencil_format
                .map(format_to_vk)
                .unwrap_or(vk::Format::UNDEFINED),
        };

        // Push constants for root data
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(desc.root_constant_size);

        // Pipeline layout with bindless texture set
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
            root_constant_size: desc.root_constant_size,
            device: self.device.clone(),
            desc: pso_desc,
            blend_pipelines: RefCell::new(std::collections::HashMap::new()),
        };
        // Pre-bake the embedded blend state if provided, otherwise bake the default.
        let initial_blend = desc.blendstate.as_ref().cloned().unwrap_or_default();
        let pipeline = vk_pso.pipeline_for_blend(&initial_blend);
        vk_pso.pipeline = pipeline;

        Ok(GraphicsPso {
            inner: GraphicsPsoInner::Vulkan(Box::new(vk_pso)),
        })
    }

    pub fn create_compute_pso(&self, desc: &ComputePsoDesc) -> RhiResult<ComputePso> {
        let modules = self.shader_modules.borrow();
        let shader = &modules[desc.compute_shader];

        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader.module)
            .name(&shader.entry_point);

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(desc.root_constant_size);

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
            .stage(stage)
            .layout(pipeline_layout);

        let pipelines = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|e| RhiError::PipelineCreation(format!("{e:?}")))?
        };

        Ok(ComputePso {
            inner: ComputePsoInner::Vulkan(VulkanComputePso {
                pipeline: pipelines[0],
                pipeline_layout,
                root_constant_size: desc.root_constant_size,
                threads_per_threadgroup: desc.threads_per_threadgroup,
            }),
        })
    }

    // -- Mesh-shader pipeline (VK_EXT_mesh_shader) --

    pub fn create_meshlet_pso(&self, desc: &MeshletPsoDesc) -> RhiResult<MeshletPso> {
        // Require mesh-shader extension support.
        if !self.mesh_shader_supported {
            return Err(RhiError::Unsupported(
                "VK_EXT_mesh_shader not available on this device".into(),
            ));
        }

        let modules = self.shader_modules.borrow();
        let mesh_module = &modules[desc.mesh_shader];
        let frag_module = &modules[desc.pixel_shader];

        let pso_desc = VulkanMeshletPsoDesc {
            mesh_module: mesh_module.module,
            frag_module: frag_module.module,
            mesh_entry: mesh_module.entry_point.clone(),
            frag_entry: frag_module.entry_point.clone(),
            color_targets: desc.color_targets.clone(),
            depth_format: desc.depth_format.map(format_to_vk),
            stencil_format: desc
                .stencil_format
                .map(format_to_vk)
                .unwrap_or(vk::Format::UNDEFINED),
            sample_count: desc.sample_count,
            alpha_to_coverage: desc.alpha_to_coverage,
            cull: desc.cull,
        };

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(desc.root_constant_size);

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
            root_constant_size: desc.root_constant_size,
            device: self.device.clone(),
            desc: pso_desc,
            blend_pipelines: RefCell::new(std::collections::HashMap::new()),
        };
        let initial_blend = desc.blendstate.as_ref().cloned().unwrap_or_default();
        let pipeline = vk_pso.pipeline_for_blend(&initial_blend);
        vk_pso.pipeline = pipeline;

        Ok(MeshletPso {
            inner: MeshletPsoInner::Vulkan(Box::new(vk_pso)),
        })
    }

    // -- Ray tracing pipeline (VK_KHR_ray_tracing_pipeline) --

    pub fn create_ray_tracing_pso(&self, desc: &RayTracingPsoDesc) -> RhiResult<RayTracingPso> {
        let (rt_loader, rt_props) = match &self.ray_tracing {
            Some(rt) => rt,
            None => {
                return Err(RhiError::Unsupported(
                    "VK_KHR_ray_tracing_pipeline not available on this device".into(),
                ));
            }
        };
        if desc.max_recursion_depth > rt_props.max_ray_recursion_depth {
            return Err(RhiError::Unsupported(format!(
                "ray recursion depth {} exceeds device limit {}",
                desc.max_recursion_depth, rt_props.max_ray_recursion_depth
            )));
        }

        let modules = self.shader_modules.borrow();
        let module_for_group_shader = |shader: usize| {
            let module_idx = *desc.shaders.get(shader).ok_or_else(|| {
                RhiError::PipelineCreation(format!(
                    "ray tracing shader group index {shader} is out of range"
                ))
            })?;
            modules.get(module_idx).ok_or_else(|| {
                RhiError::PipelineCreation(format!(
                    "ray tracing shader module index {module_idx} is out of range"
                ))
            })
        };
        let mut stage_infos: Vec<vk::PipelineShaderStageCreateInfo> = Vec::new();
        let mut groups: Vec<vk::RayTracingShaderGroupCreateInfoKHR> = Vec::new();

        for group in &desc.groups {
            match group {
                RayTracingShaderGroup::RayGen { shader } => {
                    let module = module_for_group_shader(*shader)?;
                    let idx = stage_infos.len() as u32;
                    stage_infos.push(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::RAYGEN_KHR)
                            .module(module.module)
                            .name(&module.entry_point),
                    );
                    groups.push(
                        vk::RayTracingShaderGroupCreateInfoKHR::default()
                            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                            .general_shader(idx)
                            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                            .any_hit_shader(vk::SHADER_UNUSED_KHR)
                            .intersection_shader(vk::SHADER_UNUSED_KHR),
                    );
                }
                RayTracingShaderGroup::Miss { shader } => {
                    let module = module_for_group_shader(*shader)?;
                    let idx = stage_infos.len() as u32;
                    stage_infos.push(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::MISS_KHR)
                            .module(module.module)
                            .name(&module.entry_point),
                    );
                    groups.push(
                        vk::RayTracingShaderGroupCreateInfoKHR::default()
                            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                            .general_shader(idx)
                            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                            .any_hit_shader(vk::SHADER_UNUSED_KHR)
                            .intersection_shader(vk::SHADER_UNUSED_KHR),
                    );
                }
                RayTracingShaderGroup::TriangleHit {
                    closest_hit,
                    any_hit,
                } => {
                    let chit_module = module_for_group_shader(*closest_hit)?;
                    let chit_idx = stage_infos.len() as u32;
                    stage_infos.push(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                            .module(chit_module.module)
                            .name(&chit_module.entry_point),
                    );
                    let ahit_idx = if let Some(ah) = any_hit {
                        let ahit_module = module_for_group_shader(*ah)?;
                        let i = stage_infos.len() as u32;
                        stage_infos.push(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::ANY_HIT_KHR)
                                .module(ahit_module.module)
                                .name(&ahit_module.entry_point),
                        );
                        i
                    } else {
                        vk::SHADER_UNUSED_KHR
                    };
                    groups.push(
                        vk::RayTracingShaderGroupCreateInfoKHR::default()
                            .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                            .general_shader(vk::SHADER_UNUSED_KHR)
                            .closest_hit_shader(chit_idx)
                            .any_hit_shader(ahit_idx)
                            .intersection_shader(vk::SHADER_UNUSED_KHR),
                    );
                }
                RayTracingShaderGroup::ProceduralHit {
                    intersection,
                    closest_hit,
                    any_hit,
                } => {
                    let isect_module = module_for_group_shader(*intersection)?;
                    let isect_idx = stage_infos.len() as u32;
                    stage_infos.push(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::INTERSECTION_KHR)
                            .module(isect_module.module)
                            .name(&isect_module.entry_point),
                    );
                    let chit_idx = if let Some(ch) = closest_hit {
                        let chit_module = module_for_group_shader(*ch)?;
                        let i = stage_infos.len() as u32;
                        stage_infos.push(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                                .module(chit_module.module)
                                .name(&chit_module.entry_point),
                        );
                        i
                    } else {
                        vk::SHADER_UNUSED_KHR
                    };
                    let ahit_idx = if let Some(ah) = any_hit {
                        let ahit_module = module_for_group_shader(*ah)?;
                        let i = stage_infos.len() as u32;
                        stage_infos.push(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::ANY_HIT_KHR)
                                .module(ahit_module.module)
                                .name(&ahit_module.entry_point),
                        );
                        i
                    } else {
                        vk::SHADER_UNUSED_KHR
                    };
                    groups.push(
                        vk::RayTracingShaderGroupCreateInfoKHR::default()
                            .ty(vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP)
                            .general_shader(vk::SHADER_UNUSED_KHR)
                            .closest_hit_shader(chit_idx)
                            .any_hit_shader(ahit_idx)
                            .intersection_shader(isect_idx),
                    );
                }
                RayTracingShaderGroup::Callable { shader } => {
                    let module = module_for_group_shader(*shader)?;
                    let idx = stage_infos.len() as u32;
                    stage_infos.push(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::CALLABLE_KHR)
                            .module(module.module)
                            .name(&module.entry_point),
                    );
                    groups.push(
                        vk::RayTracingShaderGroupCreateInfoKHR::default()
                            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                            .general_shader(idx)
                            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                            .any_hit_shader(vk::SHADER_UNUSED_KHR)
                            .intersection_shader(vk::SHADER_UNUSED_KHR),
                    );
                }
            }
        }

        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::ALL)
            .offset(0)
            .size(64); // Standard: two GpuAddress root pointers
        let set_layouts = [self.texture_descriptor_set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(std::slice::from_ref(&push_range));
        let pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&layout_info, None)
                .map_err(|e| RhiError::PipelineCreation(e.to_string()))?
        };

        let pipeline_info = vk::RayTracingPipelineCreateInfoKHR::default()
            .stages(&stage_infos)
            .groups(&groups)
            .max_pipeline_ray_recursion_depth(desc.max_recursion_depth)
            .layout(pipeline_layout);

        let pipelines = unsafe {
            rt_loader
                .create_ray_tracing_pipelines(
                    vk::DeferredOperationKHR::null(),
                    vk::PipelineCache::null(),
                    &[pipeline_info],
                    None,
                )
                .map_err(|e| RhiError::PipelineCreation(format!("{e:?}")))?
        };
        let pipeline = pipelines[0];

        // Fetch opaque shader group handles for SBT construction.
        let handle_size = rt_props.shader_group_handle_size;
        let total = handle_size as usize * groups.len();
        let group_handles = unsafe {
            rt_loader
                .get_ray_tracing_shader_group_handles(pipeline, 0, groups.len() as u32, total)
                .map_err(|e| RhiError::PipelineCreation(e.to_string()))?
        };

        Ok(RayTracingPso {
            inner: RayTracingPsoInner::Vulkan(Box::new(VulkanRayTracingPso {
                pipeline,
                pipeline_layout,
                device: self.device.clone(),
                group_handles,
                handle_size,
                handle_alignment: rt_props.shader_group_handle_alignment,
            })),
        })
    }

    // -- Acceleration structures (VK_KHR_acceleration_structure) --

    fn build_flags_to_vk(flags: BuildAccelFlags) -> vk::BuildAccelerationStructureFlagsKHR {
        let mut out = vk::BuildAccelerationStructureFlagsKHR::empty();
        if flags.contains(BuildAccelFlags::ALLOW_UPDATE) {
            out |= vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE;
        }
        if flags.contains(BuildAccelFlags::ALLOW_COMPACTION) {
            out |= vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION;
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

    pub fn create_blas(&self, desc: &BlasDesc) -> RhiResult<AccelerationStructure> {
        let accel_loader = self.require_accel_loader()?;

        // Build geometry descriptors.
        let geometries: Vec<vk::AccelerationStructureGeometryKHR> = desc
            .meshes
            .iter()
            .map(|m| match m.geometry_type {
                GeometryType::Triangles => {
                    let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                        .vertex_format(vk::Format::R32G32B32_SFLOAT)
                        .vertex_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: m.vertex_buffer.0,
                        })
                        .vertex_stride(m.vertex_stride)
                        .max_vertex(m.vertex_count.saturating_sub(1))
                        .index_type(if m.index_count > 0 {
                            vk::IndexType::UINT32
                        } else {
                            vk::IndexType::NONE_KHR
                        })
                        .index_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: m.index_buffer.0,
                        });
                    let geo_data = vk::AccelerationStructureGeometryDataKHR { triangles };
                    let flags = if m.flags.contains(GeometryFlags::OPAQUE) {
                        vk::GeometryFlagsKHR::OPAQUE
                    } else {
                        vk::GeometryFlagsKHR::empty()
                    };
                    vk::AccelerationStructureGeometryKHR::default()
                        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                        .geometry(geo_data)
                        .flags(flags)
                }
                GeometryType::Aabbs => {
                    let aabbs = vk::AccelerationStructureGeometryAabbsDataKHR::default()
                        .data(vk::DeviceOrHostAddressConstKHR {
                            device_address: m.aabb_buffer.0,
                        })
                        .stride(std::mem::size_of::<vk::AabbPositionsKHR>() as u64);
                    let geo_data = vk::AccelerationStructureGeometryDataKHR { aabbs };
                    let flags = if m.flags.contains(GeometryFlags::OPAQUE) {
                        vk::GeometryFlagsKHR::OPAQUE
                    } else {
                        vk::GeometryFlagsKHR::empty()
                    };
                    vk::AccelerationStructureGeometryKHR::default()
                        .geometry_type(vk::GeometryTypeKHR::AABBS)
                        .geometry(geo_data)
                        .flags(flags)
                }
            })
            .collect();

        let primitive_counts: Vec<u32> = desc
            .meshes
            .iter()
            .map(|m| match m.geometry_type {
                GeometryType::Triangles => {
                    if m.index_count > 0 {
                        m.index_count / 3
                    } else {
                        m.vertex_count / 3
                    }
                }
                GeometryType::Aabbs => m.aabb_count,
            })
            .collect();

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(Self::build_flags_to_vk(desc.flags))
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometries);

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            accel_loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &primitive_counts,
                &mut size_info,
            );
        }

        self.finalize_accel_structure(
            accel_loader,
            size_info.acceleration_structure_size,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        )
    }

    /// Vulkan's TLAS instance layout is `VkAccelerationStructureInstanceKHR`, which is
    /// exactly the RHI's `TlasInstance` layout (BLAS referenced by device address).
    pub fn tlas_instance_stride(&self) -> usize {
        std::mem::size_of::<crate::types::TlasInstance>()
    }

    pub fn write_tlas_instance(&self, dst: *mut u8, inst: &crate::types::TlasInstance) {
        unsafe {
            std::ptr::write_unaligned(dst as *mut crate::types::TlasInstance, *inst);
        }
    }

    pub fn create_tlas(&self, desc: &TlasDesc) -> RhiResult<AccelerationStructure> {
        let accel_loader = self.require_accel_loader()?;

        let instances_data = vk::AccelerationStructureGeometryInstancesDataKHR::default()
            .array_of_pointers(false)
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: desc.instance_buffer.0,
            });
        let geo_data = vk::AccelerationStructureGeometryDataKHR {
            instances: instances_data,
        };
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(geo_data);
        let geometries = [geometry];

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(Self::build_flags_to_vk(desc.flags))
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometries);

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            accel_loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[desc.instance_count],
                &mut size_info,
            );
        }

        self.finalize_accel_structure(
            accel_loader,
            size_info.acceleration_structure_size,
            vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        )
    }

    fn require_accel_loader(&self) -> RhiResult<&vk_accel_structure::Device> {
        self.acceleration_structure.as_ref().ok_or_else(|| {
            RhiError::Unsupported(
                "VK_KHR_acceleration_structure not available on this device".into(),
            )
        })
    }

    /// Allocate the backing buffer, create the acceleration structure, query its
    /// device address, and wrap everything into the public `AccelerationStructure`.
    /// Shared by `create_blas` / `create_tlas` — both only differ in the geometry
    /// build info (which feeds size_info before this is called).
    fn finalize_accel_structure(
        &self,
        accel_loader: &vk_accel_structure::Device,
        size: u64,
        ty: vk::AccelerationStructureTypeKHR,
    ) -> RhiResult<AccelerationStructure> {
        let (buffer, memory) = self.allocate_accel_buffer(size)?;
        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(buffer)
            .size(size)
            .ty(ty);
        let acceleration_structure = unsafe {
            accel_loader
                .create_acceleration_structure(&create_info, None)
                .map_err(|e| RhiError::AllocationFailed(e.to_string()))?
        };
        let device_address = unsafe {
            accel_loader.get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                    .acceleration_structure(acceleration_structure),
            )
        };

        let id = {
            let mut next = self.accel_counter.borrow_mut();
            let id = *next;
            *next += 1;
            AccelerationStructureId(id)
        };

        Ok(AccelerationStructure {
            id,
            inner: AccelInner::Vulkan(Box::new(VulkanAccelerationStructure {
                acceleration_structure,
                buffer,
                buffer_memory: memory,
                device_address,
                device: self.device.clone(),
                accel_loader: accel_loader.clone(),
            })),
        })
    }

    /// Allocate a device-local buffer for an acceleration structure.
    fn allocate_accel_buffer(&self, size: u64) -> RhiResult<(vk::Buffer, vk::DeviceMemory)> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe {
            self.device
                .create_buffer(&buffer_info, None)
                .map_err(|e| RhiError::AllocationFailed(e.to_string()))?
        };
        let reqs = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let mem_index = find_memorytype_index(
            &reqs,
            &self.device_memory_properties,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or_else(|| RhiError::AllocationFailed("No device-local memory for AS".into()))?;
        let mut flags_info =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(mem_index)
            .push_next(&mut flags_info);
        let memory = unsafe {
            self.device
                .allocate_memory(&alloc_info, None)
                .map_err(|e| RhiError::AllocationFailed(e.to_string()))?
        };
        unsafe {
            self.device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(|e| RhiError::AllocationFailed(e.to_string()))?;
        }
        Ok((buffer, memory))
    }

    // -- Command Buffer --

    pub fn create_command_buffer(&self) -> RhiResult<CommandBuffer> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY);

        let cmd = unsafe {
            self.device
                .allocate_command_buffers(&alloc_info)
                .map_err(|e| RhiError::CommandBuffer(e.to_string()))?
        }[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin_info)
                .map_err(|e| RhiError::CommandBuffer(e.to_string()))?;
        }

        let heap = self
            .descriptor_buffer_heap
            .as_ref()
            .expect("descriptor buffer heap missing");
        let descriptor_buffer_binding = vk::DescriptorBufferBindingInfoEXT::default()
            .address(heap.gpu_address.0)
            .usage(
                vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT
                    | vk::BufferUsageFlags::SAMPLER_DESCRIPTOR_BUFFER_EXT,
            );
        let descriptor_buffer_loader =
            Some(descriptor_buffer::Device::new(&self.instance, &self.device));
        let descriptor_buffer_binding = Some(descriptor_buffer_binding);
        let mesh_shader = if self.mesh_shader_supported {
            Some(vk_mesh_shader::Device::new(&self.instance, &self.device))
        } else {
            None
        };
        let accel_loader_cmd = self.acceleration_structure.clone();
        let rt_loader_cmd = self.ray_tracing.as_ref().map(|(l, _)| l.clone());

        Ok(CommandBuffer {
            inner: CommandBufferInner::Vulkan(Box::new(VulkanCommandBuffer {
                command_buffer: cmd,
                device: self.device.clone(),
                swapchain_image_views: Vec::new(),
                swapchain_images: Vec::new(),
                depth_image_view: vk::ImageView::null(),
                pipeline_layout: vk::PipelineLayout::null(),
                descriptor_buffer_loader,
                descriptor_buffer_binding,
                active_descriptor_buffer_offset: 0,
                root_constant_size: (std::mem::size_of::<GpuAddress>() * 4) as u32,
                push_constant_stages: vk::ShaderStageFlags::empty(),
                current_blend_state: BlendState::default(),
                pending_split_barrier: None,
                pending_value_waits: Vec::new(),
                pending_value_signals: Vec::new(),
                allocations: self.allocations.clone(),
                textures: self.textures.clone(),
                mesh_shader,
                acceleration_structure: accel_loader_cmd,
                ray_tracing: rt_loader_cmd,
                max_draw_indirect_count: self.max_draw_indirect_count,
            })),
        })
    }

    /// Create a command buffer pre-configured with swapchain image views for rendering.
    pub fn create_command_buffer_for_swapchain(
        &self,
        swapchain: &Swapchain,
    ) -> RhiResult<CommandBuffer> {
        let sc = match &swapchain.inner {
            SwapchainInner::Vulkan(s) => s,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        };

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY);

        let cmd = unsafe {
            self.device
                .allocate_command_buffers(&alloc_info)
                .map_err(|e| RhiError::CommandBuffer(e.to_string()))?
        }[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin_info)
                .map_err(|e| RhiError::CommandBuffer(e.to_string()))?;
        }

        let heap = self
            .descriptor_buffer_heap
            .as_ref()
            .expect("descriptor buffer heap missing");
        let descriptor_buffer_binding = vk::DescriptorBufferBindingInfoEXT::default()
            .address(heap.gpu_address.0)
            .usage(
                vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT
                    | vk::BufferUsageFlags::SAMPLER_DESCRIPTOR_BUFFER_EXT,
            );
        let descriptor_buffer_loader =
            Some(descriptor_buffer::Device::new(&self.instance, &self.device));
        let descriptor_buffer_binding = Some(descriptor_buffer_binding);
        let mesh_shader = if self.mesh_shader_supported {
            Some(vk_mesh_shader::Device::new(&self.instance, &self.device))
        } else {
            None
        };
        let accel_loader_cmd = self.acceleration_structure.clone();
        let rt_loader_cmd = self.ray_tracing.as_ref().map(|(l, _)| l.clone());

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
                active_descriptor_buffer_offset: 0,
                root_constant_size: (std::mem::size_of::<GpuAddress>() * 4) as u32,
                push_constant_stages: vk::ShaderStageFlags::empty(),
                current_blend_state: BlendState::default(),
                pending_split_barrier: None,
                pending_value_waits: Vec::new(),
                pending_value_signals: Vec::new(),
                allocations: self.allocations.clone(),
                textures: self.textures.clone(),
                mesh_shader,
                acceleration_structure: accel_loader_cmd,
                ray_tracing: rt_loader_cmd,
                max_draw_indirect_count: self.max_draw_indirect_count,
            })),
        })
    }

    // -- Timeline Semaphore --

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

        Ok(TimelineSemaphore {
            inner: TimelineSemaphoreInner::Vulkan(Box::new(VulkanTimelineSemaphore {
                semaphore,
                device: self.device.clone(),
            })),
        })
    }

    // -- Destroy --

    pub fn destroy_buffer(&self, buffer: GpuBuffer) {
        match buffer.inner {
            GpuBufferInner::Vulkan(b) => unsafe {
                {
                    let mut allocations =
                        self.allocations.lock().expect("allocations lock poisoned");
                    allocations.remove(&b.gpu_address.0);
                }
                if b.mapped_ptr.is_some() {
                    self.device.unmap_memory(b.memory);
                }
                self.device.destroy_buffer(b.buffer, None);
                self.device.free_memory(b.memory, None);
            },
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    pub fn destroy_texture(&self, texture: Texture) {
        let idx = texture.id.0 as usize;
        let mut textures = self.textures.lock().expect("textures lock poisoned");
        if let Some(Some(vk_tex)) = textures.get(idx) {
            unsafe {
                self.device.destroy_image_view(vk_tex.image_view, None);
                // View-only entries don't own the underlying image.
                if !vk_tex.is_view {
                    self.device.destroy_image(vk_tex.image, None);
                }
            }
        }
        if idx < textures.len() {
            textures[idx] = None;
        }
    }

    pub fn texture_view_descriptor(
        &self,
        source: &crate::texture::Texture,
        view: &crate::texture::GpuViewDesc,
    ) -> RhiResult<TextureId> {
        self.create_view_internal(source, view, false)
    }

    pub fn rw_texture_view_descriptor(
        &self,
        source: &crate::texture::Texture,
        view: &crate::texture::GpuViewDesc,
    ) -> RhiResult<TextureId> {
        self.create_view_internal(source, view, true)
    }

    fn create_view_internal(
        &self,
        source: &crate::texture::Texture,
        view: &crate::texture::GpuViewDesc,
        storage: bool,
    ) -> RhiResult<TextureId> {
        use crate::texture::{ALL_LAYERS, ALL_MIPS};

        // Look up the source VulkanTexture to get the vk::Image and its aspect/format.
        let (src_image, src_format, src_aspect, src_mip_levels, src_array_layers, src_view_type) = {
            let textures = self.textures.lock().expect("textures lock poisoned");
            let src = textures
                .get(source.id.0 as usize)
                .and_then(|t| t.as_ref())
                .ok_or_else(|| {
                    RhiError::Backend("texture_view_descriptor: invalid source TextureId".into())
                })?;

            let fmt = format_to_vk(source.desc().format);
            let aspect = if is_depth_format(source.desc().format) {
                vk::ImageAspectFlags::DEPTH
            } else {
                vk::ImageAspectFlags::COLOR
            };
            let mips = source.desc().mip_levels;
            let layers = match source.desc().dimension {
                TextureDimension::Cube => source.desc().array_layers * 6,
                _ => source.desc().array_layers,
            };
            let vt = match source.desc().dimension {
                TextureDimension::D1 => vk::ImageViewType::TYPE_1D,
                TextureDimension::D2 => vk::ImageViewType::TYPE_2D,
                TextureDimension::D2Array => vk::ImageViewType::TYPE_2D_ARRAY,
                TextureDimension::D3 => vk::ImageViewType::TYPE_3D,
                TextureDimension::Cube => vk::ImageViewType::CUBE,
                TextureDimension::CubeArray => vk::ImageViewType::CUBE_ARRAY,
            };
            (src.image, fmt, aspect, mips, layers, vt)
        };

        let vk_format = view.format.map(format_to_vk).unwrap_or(src_format);

        let level_count = if view.mip_count == ALL_MIPS {
            src_mip_levels.saturating_sub(view.base_mip as u32)
        } else {
            view.mip_count as u32
        };
        let layer_count = if view.layer_count == ALL_LAYERS {
            src_array_layers.saturating_sub(view.base_layer as u32)
        } else {
            view.layer_count as u32
        };

        let view_info = vk::ImageViewCreateInfo::default()
            .image(src_image)
            .view_type(src_view_type)
            .format(vk_format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: src_aspect,
                base_mip_level: view.base_mip as u32,
                level_count,
                base_array_layer: view.base_layer as u32,
                layer_count,
            });

        let image_view = unsafe {
            self.device
                .create_image_view(&view_info, None)
                .map_err(|e| RhiError::TextureCreation(format!("texture_view_descriptor: {e}")))?
        };

        let texture_id = {
            let mut id = self.next_texture_id.borrow_mut();
            let tid = TextureId(*id);
            *id += 1;
            tid
        };

        self.write_image_descriptor(texture_id, image_view, vk::ImageLayout::GENERAL, storage)?;

        let vk_texture = VulkanTexture {
            image: vk::Image::null(),
            image_view,
            is_view: true,
        };

        {
            let mut textures = self.textures.lock().expect("textures lock poisoned");
            if textures.len() <= texture_id.0 as usize {
                textures.resize_with(texture_id.0 as usize + 1, || None);
            }
            textures[texture_id.0 as usize] = Some(vk_texture);
        }

        Ok(texture_id)
    }

    // -- Private helpers --

    fn write_image_descriptor(
        &self,
        id: TextureId,
        image_view: vk::ImageView,
        layout: vk::ImageLayout,
        storage: bool,
    ) -> RhiResult<()> {
        let heap = self
            .descriptor_buffer_heap
            .as_ref()
            .ok_or_else(|| RhiError::Backend("Descriptor buffer heap missing".into()))?;
        let loader = self
            .descriptor_buffer_loader
            .as_ref()
            .ok_or_else(|| RhiError::Backend("Descriptor buffer loader missing".into()))?;

        let (base, stride, ty, kind) = if storage {
            (
                heap.storage_image_offset,
                heap.storage_image_stride,
                vk::DescriptorType::STORAGE_IMAGE,
                "Storage",
            )
        } else {
            (
                heap.sampled_image_offset,
                heap.sampled_image_stride,
                vk::DescriptorType::SAMPLED_IMAGE,
                "Sampled",
            )
        };

        let offset = base + (id.0 as u64) * stride;
        if offset + stride > heap.size {
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
                stride as usize,
            );
            loader.get_descriptor(&get_info, dst);
        }
        Ok(())
    }

    fn write_sampler_descriptor(&self, id: SamplerId, sampler: vk::Sampler) -> RhiResult<()> {
        let heap = self
            .descriptor_buffer_heap
            .as_ref()
            .ok_or_else(|| RhiError::Backend("Descriptor buffer heap missing".into()))?;
        let loader = self
            .descriptor_buffer_loader
            .as_ref()
            .ok_or_else(|| RhiError::Backend("Descriptor buffer loader missing".into()))?;

        let offset = heap.sampler_offset + (id.0 as u64) * heap.sampler_stride;
        if offset + heap.sampler_stride > heap.size {
            return Err(RhiError::Backend("Sampler descriptor heap overflow".into()));
        }

        let get_info = vk::DescriptorGetInfoEXT::default()
            .ty(vk::DescriptorType::SAMPLER)
            .data(vk::DescriptorDataEXT {
                p_sampler: &sampler,
            });

        unsafe {
            let dst = std::slice::from_raw_parts_mut(
                heap.mapped_ptr.add(offset as usize),
                heap.sampler_stride as usize,
            );
            loader.get_descriptor(&get_info, dst);
        }

        Ok(())
    }

    fn create_swapchain_image_views(
        &self,
        images: &[vk::Image],
        format: vk::Format,
    ) -> RhiResult<Vec<vk::ImageView>> {
        images
            .iter()
            .map(|&image| {
                let view_info = vk::ImageViewCreateInfo::default()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image(image);
                unsafe {
                    self.device
                        .create_image_view(&view_info, None)
                        .map_err(|e| RhiError::SwapchainCreation(e.to_string()))
                }
            })
            .collect()
    }

    fn create_depth_buffer(
        &self,
        width: u32,
        height: u32,
    ) -> RhiResult<(vk::Image, vk::ImageView, vk::DeviceMemory)> {
        let depth_image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let depth_image = unsafe {
            self.device
                .create_image(&depth_image_info, None)
                .map_err(|e| RhiError::SwapchainCreation(format!("Depth image: {e}")))?
        };

        let mem_reqs = unsafe { self.device.get_image_memory_requirements(depth_image) };
        let mem_index = find_memorytype_index(
            &mem_reqs,
            &self.device_memory_properties,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or_else(|| RhiError::AllocationFailed("No memory for depth".into()))?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_index);

        let depth_memory = unsafe {
            self.device
                .allocate_memory(&alloc_info, None)
                .map_err(|e| RhiError::AllocationFailed(e.to_string()))?
        };

        unsafe {
            self.device
                .bind_image_memory(depth_image, depth_memory, 0)
                .map_err(|e| RhiError::SwapchainCreation(e.to_string()))?;
        }

        // Transition depth image to optimal layout
        self.transition_depth_image(depth_image)?;

        let view_info = vk::ImageViewCreateInfo::default()
            .image(depth_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let depth_view = unsafe {
            self.device
                .create_image_view(&view_info, None)
                .map_err(|e| RhiError::SwapchainCreation(format!("Depth view: {e}")))?
        };

        Ok((depth_image, depth_view, depth_memory))
    }

    fn transition_depth_image(&self, depth_image: vk::Image) -> RhiResult<()> {
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

    fn transition_image_to_general(
        &self,
        image: vk::Image,
        aspect: vk::ImageAspectFlags,
        mip_levels: u32,
        layer_count: u32,
    ) -> RhiResult<()> {
        let barrier = vk::ImageMemoryBarrier::default()
            .image(image)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
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

    /// Record a single image barrier on the setup command buffer, submit it, and
    /// block on the queue. Used for one-shot UNDEFINED → initial-layout transitions
    /// that happen outside the normal command-buffer flow.
    fn submit_setup_barrier(
        &self,
        barrier: vk::ImageMemoryBarrier<'_>,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
    ) -> RhiResult<()> {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .reset_command_buffer(
                    self.setup_command_buffer,
                    vk::CommandBufferResetFlags::empty(),
                )
                .map_err(|e| RhiError::CommandBuffer(e.to_string()))?;
            self.device
                .begin_command_buffer(self.setup_command_buffer, &begin_info)
                .map_err(|e| RhiError::CommandBuffer(e.to_string()))?;
            self.device.cmd_pipeline_barrier(
                self.setup_command_buffer,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
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
        unsafe {
            let _ = self.device.device_wait_idle();

            let q = match &self.queue.inner {
                QueueInner::Vulkan(q) => q,
                #[allow(unreachable_patterns)]
                _ => unreachable!(),
            };
            let mut value_sync = q.value_sync.lock().expect("value sync lock poisoned");
            for (_ptr, state) in value_sync.drain() {
                self.device.destroy_semaphore(state.semaphore, None);
            }

            // Destroy textures
            for t in self
                .textures
                .lock()
                .expect("textures lock poisoned")
                .drain(..)
                .flatten()
            {
                self.device.destroy_image_view(t.image_view, None);
                if !t.is_view {
                    self.device.destroy_image(t.image, None);
                }
            }

            // Destroy samplers
            for sampler in self.samplers.borrow_mut().drain(..) {
                self.device.destroy_sampler(sampler, None);
            }

            // Destroy shader modules
            for shader in self.shader_modules.borrow_mut().drain(..) {
                self.device.destroy_shader_module(shader.module, None);
            }

            if let Some(heap) = self.descriptor_buffer_heap.as_ref() {
                self.device.unmap_memory(heap.memory);
                self.device.destroy_buffer(heap.buffer, None);
                self.device.free_memory(heap.memory, None);
                self.device.destroy_descriptor_set_layout(heap.layout, None);
            }

            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);

            if let Some(ref debug_loader) = self.debug_utils_loader {
                debug_loader.destroy_debug_utils_messenger(self.debug_callback, None);
            }

            self.surface_loader
                .destroy_surface(vk::SurfaceKHR::null(), None);
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
        vk::DescriptorBindingFlags::PARTIALLY_BOUND,
    ];
    let mut binding_flags_info =
        vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);

    let bindings = [
        vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
            descriptor_count: MAX_BINDLESS_TEXTURES,
            stage_flags: vk::ShaderStageFlags::ALL,
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            binding: 1,
            descriptor_type: vk::DescriptorType::SAMPLER,
            descriptor_count: MAX_BINDLESS_SAMPLERS,
            stage_flags: vk::ShaderStageFlags::ALL,
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            binding: 2,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: MAX_BINDLESS_STORAGE_IMAGES,
            stage_flags: vk::ShaderStageFlags::ALL,
            ..Default::default()
        },
    ];

    let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
        .bindings(&bindings)
        .flags(vk::DescriptorSetLayoutCreateFlags::DESCRIPTOR_BUFFER_EXT)
        .push_next(&mut binding_flags_info);

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
    let sampled_image_offset =
        unsafe { loader.get_descriptor_set_layout_binding_offset(layout, 0) };
    let sampler_offset = unsafe { loader.get_descriptor_set_layout_binding_offset(layout, 1) };
    let storage_image_offset =
        unsafe { loader.get_descriptor_set_layout_binding_offset(layout, 2) };

    let sampled_image_stride = props.sampled_image_descriptor_size as u64;
    let sampler_stride = props.sampler_descriptor_size as u64;
    let storage_image_stride = props.storage_image_descriptor_size as u64;

    let align = props.descriptor_buffer_offset_alignment.max(1);
    let aligned_size = (layout_size + align - 1) & !(align - 1);

    let usage = vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT
        | vk::BufferUsageFlags::SAMPLER_DESCRIPTOR_BUFFER_EXT
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;

    let buffer_info = vk::BufferCreateInfo::default()
        .size(aligned_size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe {
        device
            .create_buffer(&buffer_info, None)
            .map_err(|e| RhiError::DeviceCreation(format!("Descriptor buffer: {e}")))?
    };

    let mem_reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mem_type_index = find_memorytype_index(
        &mem_reqs,
        mem_props,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .ok_or_else(|| {
        RhiError::AllocationFailed("No host visible memory for descriptor buffer".into())
    })?;

    let mut alloc_flags_info =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_reqs.size)
        .memory_type_index(mem_type_index)
        .push_next(&mut alloc_flags_info);

    let memory = unsafe {
        device
            .allocate_memory(&alloc_info, None)
            .map_err(|e| RhiError::AllocationFailed(e.to_string()))?
    };

    unsafe {
        device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(|e| RhiError::DeviceCreation(format!("Bind descriptor buffer: {e}")))?;
    }

    let mapped_ptr = unsafe {
        device
            .map_memory(memory, 0, aligned_size, vk::MemoryMapFlags::empty())
            .map_err(|e| RhiError::AllocationFailed(e.to_string()))?
    } as *mut u8;

    let addr_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
    let gpu_addr = unsafe { device.get_buffer_device_address(&addr_info) };

    Ok(DescriptorBufferHeap {
        buffer,
        memory,
        mapped_ptr,
        size: aligned_size,
        gpu_address: GpuAddress(gpu_addr),
        layout,
        sampled_image_offset,
        sampler_offset,
        storage_image_offset,
        sampled_image_stride,
        sampler_stride,
        storage_image_stride,
    })
}

fn address_mode_to_vk(mode: AddressMode) -> vk::SamplerAddressMode {
    match mode {
        AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
        AddressMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        AddressMode::ClampToBorder => vk::SamplerAddressMode::CLAMP_TO_BORDER,
    }
}

fn compare_op_to_vk(op: CompareOp) -> vk::CompareOp {
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

fn is_depth_format(format: Format) -> bool {
    matches!(
        format,
        Format::D16Unorm | Format::D32Float | Format::D24UnormS8Uint | Format::D32FloatS8Uint
    )
}
