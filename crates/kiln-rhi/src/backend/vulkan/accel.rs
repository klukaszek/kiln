//! Vulkan acceleration structures: creation and TLAS instance encoding.

use ash::khr::acceleration_structure as vk_accel_structure;
use ash::vk;

use super::device::{
    VulkanDevice, build_accel_flags_to_vk, find_memorytype_index, geometry_flags_to_vk,
};
use super::memory::SharedBufferPool;

use crate::accel::{AccelInner, AccelerationStructure};
use crate::error::{RhiError, RhiResult};
use crate::types::{BlasDesc, GeometryType, TlasDesc};

/// Vulkan acceleration structure entry (BLAS or TLAS).
///
/// `acceleration_structure` is an opaque `VkAccelerationStructureKHR`. Its storage and its build
/// scratch are ordinary pool suballocations, addressed rather than bound, so neither owns a
/// `VkBuffer` of its own.
pub struct VulkanAccelerationStructure {
    pub(crate) acceleration_structure: vk::AccelerationStructureKHR,
    pub(crate) backing: BlockRange,
    /// Resource-heap slot holding this structure's device address, released on drop. The slot is
    /// the only copy: a TLAS build reads it back through `accel_device_address`.
    pub(crate) accel_slot: u32,
    /// The value a shader's `DescriptorHandle<RaytracingAccelerationStructure>` carries.
    pub(crate) heap_index: u64,
    pub(crate) free_accel_slots: super::device::SharedAccelFreeSlots,
    /// Build scratch storage, owned by the structure so it outlives the GPU build:
    /// `build_blas`/`build_tlas` only *record* the build into a command buffer that
    /// executes later, so the scratch must stay alive until then. `scratch_address`
    /// is aligned to the device's `minAccelerationStructureScratchOffsetAlignment`.
    pub(crate) scratch: BlockRange,
    pub(crate) scratch_address: u64,
    pub(crate) buffer_pool: SharedBufferPool,
    /// Extension loader for `VK_KHR_acceleration_structure`.
    pub(crate) accel_loader: ash::khr::acceleration_structure::Device,
}

/// A range handed back to the pool when its owner dies.
#[derive(Clone, Copy)]
pub(crate) struct BlockRange {
    pub(crate) block_index: usize,
    pub(crate) offset: u64,
    pub(crate) range: u64,
}

impl Drop for VulkanAccelerationStructure {
    /// Destroys immediately; the caller guarantees the GPU is done with it.
    fn drop(&mut self) {
        unsafe {
            self.accel_loader
                .destroy_acceleration_structure(self.acceleration_structure, None);
        }
        self.free_accel_slots.borrow_mut().push(self.accel_slot);
        let mut pool = self.buffer_pool.borrow_mut();
        for r in [self.backing, self.scratch] {
            pool.release(r.block_index, r.offset, r.range);
        }
    }
}

impl VulkanDevice {
    pub fn create_blas(&self, desc: &BlasDesc) -> RhiResult<AccelerationStructure> {
        let accel_loader = &self.acceleration_structure;

        let geometries: Vec<vk::AccelerationStructureGeometryKHR> = desc
            .meshes
            .iter()
            .map(|m| match m.geometry_type {
                GeometryType::Triangles => {
                    let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                        .vertex_format(vk::Format::R32G32B32_SFLOAT)
                        .vertex_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: m.vertex_buffer.address,
                        })
                        .vertex_stride(m.vertex_stride)
                        .max_vertex(m.vertex_count.saturating_sub(1))
                        .index_type(if m.index_count > 0 {
                            vk::IndexType::UINT32
                        } else {
                            vk::IndexType::NONE_KHR
                        })
                        .index_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: m.index_buffer.address,
                        });
                    let geo_data = vk::AccelerationStructureGeometryDataKHR { triangles };
                    vk::AccelerationStructureGeometryKHR::default()
                        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                        .geometry(geo_data)
                        .flags(geometry_flags_to_vk(m.flags))
                }
                GeometryType::Aabbs => {
                    let aabbs = vk::AccelerationStructureGeometryAabbsDataKHR::default()
                        .data(vk::DeviceOrHostAddressConstKHR {
                            device_address: m.aabb_buffer.address,
                        })
                        .stride(std::mem::size_of::<vk::AabbPositionsKHR>() as u64);
                    let geo_data = vk::AccelerationStructureGeometryDataKHR { aabbs };
                    vk::AccelerationStructureGeometryKHR::default()
                        .geometry_type(vk::GeometryTypeKHR::AABBS)
                        .geometry(geo_data)
                        .flags(geometry_flags_to_vk(m.flags))
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
            .flags(build_accel_flags_to_vk(desc.flags))
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometries);

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            accel_loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                Some(&primitive_counts),
                &mut size_info,
            );
        }

        self.finalize_accel_structure(
            accel_loader,
            size_info.acceleration_structure_size,
            size_info.build_scratch_size,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        )
    }

    /// Vulkan's native TLAS instance layout matches the public fields byte-for-byte.
    pub fn tlas_instance_stride(&self) -> usize {
        std::mem::size_of::<crate::types::TlasInstance>()
    }

    /// The layouts agree on every field but the structure reference, where a build needs the
    /// device address that the public handle's heap slot holds.
    pub fn write_tlas_instance(&self, dst: *mut u8, inst: &crate::types::TlasInstance) {
        let mut native = *inst;
        native.acceleration_structure_reference = crate::types::AccelHandle::from_raw(
            self.accel_device_address(inst.acceleration_structure_reference),
        );
        unsafe {
            std::ptr::write_unaligned(dst as *mut crate::types::TlasInstance, native);
        }
    }

    pub fn create_tlas(&self, desc: &TlasDesc) -> RhiResult<AccelerationStructure> {
        let accel_loader = &self.acceleration_structure;

        let instances_data = vk::AccelerationStructureGeometryInstancesDataKHR::default()
            .array_of_pointers(false)
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: desc.instance_buffer.address,
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
            .flags(build_accel_flags_to_vk(desc.flags))
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometries);

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            accel_loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                Some(&[desc.instance_count]),
                &mut size_info,
            );
        }

        self.finalize_accel_structure(
            accel_loader,
            size_info.acceleration_structure_size,
            size_info.build_scratch_size,
            vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        )
    }

    /// Allocate the backing buffer, create the acceleration structure, query its
    /// device address, and wrap everything into the public `AccelerationStructure`.
    /// Shared by `create_blas` / `create_tlas` — both only differ in the geometry
    /// build info (which feeds size_info before this is called).
    pub(crate) fn finalize_accel_structure(
        &self,
        accel_loader: &vk_accel_structure::Device,
        size: u64,
        scratch_size: u64,
        ty: vk::AccelerationStructureTypeKHR,
    ) -> RhiResult<AccelerationStructure> {
        // `create_acceleration_structure2` takes an address range, so the storage is an ordinary
        // suballocation rather than a dedicated buffer.
        let (backing, backing_address) =
            self.allocate_accel_range(size, super::memory::BLOCK_BUFFER_USAGE)?;
        let create_info = vk::AccelerationStructureCreateInfo2KHR::default()
            .address_range(
                vk::DeviceAddressRangeKHR::default()
                    .address(backing_address)
                    .size(size),
            )
            .ty(ty);
        let acceleration_structure = unsafe {
            self.address_commands_loader
                .create_acceleration_structure2(&create_info, None)
                .map_err(|e| RhiError::AllocationFailed(e.to_string()))?
        };
        let device_address = unsafe {
            accel_loader.get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                    .acceleration_structure(acceleration_structure),
            )
        };

        // Leave enough headroom to align the scratch address as required by Vulkan.
        let scratch_align = self.accel_scratch_alignment();
        let (scratch, scratch_base) = self.allocate_accel_range(
            scratch_size + scratch_align,
            super::memory::SCRATCH_BUFFER_USAGE,
        )?;
        let scratch_address = scratch_base.next_multiple_of(scratch_align);

        // Publish the address into the heap so shaders can reach the structure bindlessly.
        let (accel_slot, heap_index) = self.write_accel_address(device_address)?;

        Ok(AccelerationStructure {
            inner: AccelInner::Vulkan(Box::new(VulkanAccelerationStructure {
                acceleration_structure,
                backing,
                accel_slot,
                heap_index,
                free_accel_slots: self.free_accel_slots.clone(),
                scratch,
                scratch_address,
                buffer_pool: self.buffer_pool.clone(),
                accel_loader: accel_loader.clone(),
            })),
            _owner: None,
        })
    }

    /// Device-local storage for an acceleration structure or its build scratch, carved from the
    /// ordinary buffer pool.
    fn allocate_accel_range(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> RhiResult<(BlockRange, u64)> {
        let requirements = self.buffer_requirements(size, usage)?;
        let memory_type = find_memorytype_index(
            &requirements,
            &self.device_memory_properties,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or_else(|| RhiError::AllocationFailed("no device-local memory type".into()))?;
        let sub = self.buffer_pool.borrow_mut().allocate(
            &self.device,
            &requirements,
            memory_type,
            false,
            usage,
        )?;
        Ok((
            BlockRange {
                block_index: sub.block_index,
                offset: sub.offset,
                range: sub.range,
            },
            sub.address,
        ))
    }
}
