//! Shader-visible descriptor heaps. Slots come from [`SlotTable`](crate::backend::slots).

use ash::{Device, Instance, vk, vk::TaggedStructure as _};

use crate::backend::suballoc::align_up;
use crate::error::{RhiError, RhiResult};
use crate::types::{BindlessCapacity, GpuPtr};

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
        debug_assert!(
            end <= self.reserved_offset,
            "descriptor slot {index} out of range"
        );
        start as usize..end as usize
    }

    pub(crate) fn bind_info(&self) -> vk::BindHeapInfoEXT<'static> {
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

/// Sizing inputs for one heap, all sourced from `VkPhysicalDeviceDescriptorHeapPropertiesEXT`.
struct HeapLayout {
    slots: u32,
    descriptor_size: u64,
    alignment: u64,
    reserved_size: u64,
    max_size: u64,
}

/// Allocate the resource and sampler heaps. Both are plain mapped allocations; under
/// `VK_EXT_descriptor_heap` there is no set layout, no binding list and no mutable-type dance.
pub(crate) fn create_descriptor_heaps(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    bindless: BindlessCapacity,
) -> RhiResult<DescriptorHeaps> {
    let mut props = vk::PhysicalDeviceDescriptorHeapPropertiesEXT::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push(&mut props);
    unsafe {
        instance.get_physical_device_properties2(physical_device, &mut props2);
    }

    // The resource heap holds only images: buffers reach the shader as device addresses, and an
    // acceleration structure is its own device address, so the stride is `imageDescriptorSize`
    // and never the image/buffer maximum.
    let resource = create_descriptor_heap(
        device,
        mem_props,
        HeapLayout {
            slots: bindless.textures,
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
            slots: bindless.samplers,
            descriptor_size: props.sampler_descriptor_size,
            alignment: props.sampler_heap_alignment,
            reserved_size: props.min_sampler_heap_reserved_range,
            max_size: props.max_sampler_heap_size,
        },
        "sampler descriptor heap",
    )?;

    Ok(DescriptorHeaps { resource, sampler })
}

fn create_descriptor_heap(
    device: &Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    layout: HeapLayout,
    what: &str,
) -> RhiResult<DescriptorHeap> {
    let align = layout.alignment.max(1);
    let overflow = || RhiError::AllocationFailed(format!("{what} size overflows").into());
    let descriptors = (layout.slots as u64)
        .checked_mul(layout.descriptor_size)
        .ok_or_else(overflow)?;
    // Park the driver's reserved range past the last descriptor so slot N stays at N * stride.
    let reserved_offset = align_up(descriptors, align).ok_or_else(overflow)?;
    let size = reserved_offset
        .checked_add(layout.reserved_size)
        .and_then(|total| align_up(total, align))
        .ok_or_else(overflow)?;

    if size > layout.max_size {
        return Err(RhiError::Unsupported(
            format!(
                "{what} needs {size} bytes but the device caps it at {}",
                layout.max_size
            )
            .into(),
        ));
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
        return Err(RhiError::AllocationFailed(
            format!("{what} address {gpu_address:#x} is not {align}-byte aligned").into(),
        ));
    }

    let mapped_ptr =
        match unsafe { device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty()) } {
            Ok(ptr) => ptr.cast::<u8>(),
            Err(error) => {
                unsafe {
                    device.destroy_buffer(buffer, None);
                    device.free_memory(memory, None);
                }
                return Err(RhiError::AllocationFailed(error.into()));
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
