//! Vulkan acceleration structures: creation and TLAS instance encoding.

use ash::khr::acceleration_structure as vk_accel_structure;
use ash::vk;

use super::device::{VulkanDevice, build_accel_flags_to_vk, geometry_flags_to_vk};

use crate::accel::{AccelInner, AccelerationStructure};
use crate::error::{RhiError, RhiResult};
use crate::types::{BlasDesc, GeometryType, TlasDesc};

/// Vulkan acceleration structure entry (BLAS or TLAS).
///
/// `acceleration_structure` is an opaque `VkAccelerationStructureKHR`.
/// `buffer` / `buffer_memory` hold the backing storage for the AS data.
pub struct VulkanAccelerationStructure {
    pub(crate) acceleration_structure: vk::AccelerationStructureKHR,
    pub(crate) buffer: vk::Buffer,
    pub(crate) buffer_memory: vk::DeviceMemory,
    /// The GPU-visible address of this acceleration structure.
    /// Use this value in `TlasInstance::acceleration_structure_reference`
    /// and in root structs where the shader accesses it via `TraceRayInline`.
    pub(crate) device_address: u64,
    /// Build scratch storage, owned by the structure so it outlives the GPU build:
    /// `build_blas`/`build_tlas` only *record* the build into a command buffer that
    /// executes later, so the scratch must stay alive until then. `scratch_address`
    /// is aligned to the device's `minAccelerationStructureScratchOffsetAlignment`.
    pub(crate) scratch_buffer: vk::Buffer,
    pub(crate) scratch_memory: vk::DeviceMemory,
    pub(crate) scratch_address: u64,
    /// Extension loader for `VK_KHR_acceleration_structure`.
    pub(crate) accel_loader: ash::khr::acceleration_structure::Device,
    pub(crate) device: ash::Device,
}

impl Drop for VulkanAccelerationStructure {
    /// Destroys immediately; the caller guarantees the GPU is done with it.
    fn drop(&mut self) {
        unsafe {
            self.accel_loader
                .destroy_acceleration_structure(self.acceleration_structure, None);
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.buffer_memory, None);
            self.device.destroy_buffer(self.scratch_buffer, None);
            self.device.free_memory(self.scratch_memory, None);
        }
    }
}

impl VulkanDevice {
    pub fn create_blas(&self, desc: &BlasDesc) -> RhiResult<AccelerationStructure> {
        let accel_loader = self.require_accel_loader()?;

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
                &primitive_counts,
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
                &[desc.instance_count],
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

        // Leave enough headroom to align the scratch address as required by Vulkan.
        let scratch_align = self.accel_scratch_alignment();
        let (scratch_buffer, scratch_memory, scratch_base) =
            self.allocate_scratch_buffer(scratch_size + scratch_align)?;
        let scratch_address = scratch_base.next_multiple_of(scratch_align);

        Ok(AccelerationStructure {
            inner: AccelInner::Vulkan(Box::new(VulkanAccelerationStructure {
                acceleration_structure,
                buffer,
                buffer_memory: memory,
                device_address,
                scratch_buffer,
                scratch_memory,
                scratch_address,
                accel_loader: accel_loader.clone(),
                device: self.device.clone(),
            })),
            _owner: None,
        })
    }

    pub(crate) fn allocate_scratch_buffer(
        &self,
        size: u64,
    ) -> RhiResult<(vk::Buffer, vk::DeviceMemory, u64)> {
        super::memory::allocate_bound_buffer(
            &self.device,
            &self.device_memory_properties,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            "AS scratch",
        )
    }

    pub(crate) fn allocate_accel_buffer(
        &self,
        size: u64,
    ) -> RhiResult<(vk::Buffer, vk::DeviceMemory)> {
        let (buffer, memory, _) = super::memory::allocate_bound_buffer(
            &self.device,
            &self.device_memory_properties,
            size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            "AS storage",
        )?;
        Ok((buffer, memory))
    }
}
