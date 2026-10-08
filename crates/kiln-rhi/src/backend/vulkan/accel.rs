//! Vulkan acceleration structures: creation and TLAS instance encoding.

use ash::vk;
use smallvec::SmallVec;
use zerocopy::IntoBytes as _;

use super::device::{
    VulkanDevice, build_accel_flags_to_vk, find_memorytype_index, geometry_flags_to_vk,
};
use super::memory::{BLOCK_BUFFER_USAGE, SCRATCH_BUFFER_USAGE, SharedBufferPool};
use super::queue::VulkanRetiredResource;
use crate::accel::AccelerationStructure;
use crate::error::{RhiError, RhiResult};
use crate::types::{BlasDesc, BlasGeometry, BuildAccelFlags, TlasDesc, TlasInstance};

/// The build info `create_*` sizes a structure with and `build_*` records it with, minus the
/// destination and scratch only a build has. Sharing it keeps the two from disagreeing.
pub(crate) fn build_geometry_info<'a>(
    ty: vk::AccelerationStructureTypeKHR,
    flags: BuildAccelFlags,
    geometries: &'a [vk::AccelerationStructureGeometryKHR<'a>],
) -> vk::AccelerationStructureBuildGeometryInfoKHR<'a> {
    vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(ty)
        .flags(build_accel_flags_to_vk(flags))
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(geometries)
}

/// The single instances geometry of a TLAS.
pub(crate) fn tlas_geometry(desc: &TlasDesc) -> vk::AccelerationStructureGeometryKHR<'static> {
    vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                .array_of_pointers(false)
                .data(vk::DeviceOrHostAddressConstKHR {
                    device_address: desc.instance_buffer.address,
                }),
        })
}

/// Geometry descriptors and primitive counts for a BLAS.
pub(crate) fn blas_geometries(
    desc: &BlasDesc<'_>,
) -> (
    SmallVec<[vk::AccelerationStructureGeometryKHR<'static>; 4]>,
    SmallVec<[u32; 4]>,
) {
    let geometries = desc
        .meshes
        .iter()
        .map(|mesh| {
            let (ty, data) = match mesh.geometry {
                BlasGeometry::Triangles {
                    vertices,
                    stride,
                    count,
                    indices,
                } => (
                    vk::GeometryTypeKHR::TRIANGLES,
                    vk::AccelerationStructureGeometryDataKHR {
                        triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                            .vertex_format(vk::Format::R32G32B32_SFLOAT)
                            .vertex_data(vk::DeviceOrHostAddressConstKHR {
                                device_address: vertices.address,
                            })
                            .vertex_stride(stride)
                            .max_vertex(count.saturating_sub(1))
                            .index_type(match indices {
                                Some(_) => vk::IndexType::UINT32,
                                None => vk::IndexType::NONE_KHR,
                            })
                            .index_data(vk::DeviceOrHostAddressConstKHR {
                                device_address: indices.map_or(0, |i| i.buffer.address),
                            }),
                    },
                ),
                BlasGeometry::Aabbs { buffer, .. } => (
                    vk::GeometryTypeKHR::AABBS,
                    vk::AccelerationStructureGeometryDataKHR {
                        aabbs: vk::AccelerationStructureGeometryAabbsDataKHR::default()
                            .data(vk::DeviceOrHostAddressConstKHR {
                                device_address: buffer.address,
                            })
                            .stride(std::mem::size_of::<vk::AabbPositionsKHR>() as u64),
                    },
                ),
            };
            vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(ty)
                .geometry(data)
                .flags(geometry_flags_to_vk(mesh.flags))
        })
        .collect();
    let primitive_counts = desc
        .meshes
        .iter()
        .map(|mesh| mesh.geometry.primitive_count())
        .collect();
    (geometries, primitive_counts)
}

/// Vulkan acceleration structure entry (BLAS or TLAS).
///
/// `acceleration_structure` is an opaque `VkAccelerationStructureKHR`. Its storage and its build
/// scratch are ordinary pool suballocations, addressed rather than bound, so neither owns a
/// `VkBuffer` of its own.
pub struct VulkanAccelerationStructure {
    pub(crate) acceleration_structure: vk::AccelerationStructureKHR,
    pub(crate) backing: BlockRange,
    /// The GPU-visible address of this acceleration structure, and the whole of its public
    /// [`AccelHandle`](crate::AccelHandle): a shader converts it back with
    /// `OpConvertUToAccelerationStructureKHR`, and a TLAS instance descriptor stores it verbatim.
    pub(crate) device_address: u64,
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

impl VulkanAccelerationStructure {
    /// Destroy the structure and return its backing and scratch ranges to the pool. Called from
    /// the queue's retirement path, never from `Drop`.
    ///
    /// # Safety
    /// No submission referencing this structure may still be executing.
    pub(crate) unsafe fn destroy(&self) {
        unsafe {
            self.accel_loader
                .destroy_acceleration_structure(self.acceleration_structure, None);
        }
        let mut pool = self.buffer_pool.borrow_mut();
        for r in [self.backing, self.scratch] {
            pool.release(r.block_index, r.offset, r.range);
        }
    }
}

impl VulkanDevice {
    pub fn create_blas(&self, desc: &BlasDesc<'_>) -> RhiResult<AccelerationStructure> {
        let (geometries, primitive_counts) = blas_geometries(desc);
        self.create_accel(
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            desc.flags,
            &geometries,
            &primitive_counts,
        )
    }

    pub fn create_tlas(&self, desc: &TlasDesc) -> RhiResult<AccelerationStructure> {
        self.create_accel(
            vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            desc.flags,
            &[tlas_geometry(desc)],
            &[desc.instance_count],
        )
    }

    pub fn tlas_instance_stride(&self) -> usize {
        TLAS_INSTANCE_STRIDE
    }

    /// Size a structure, allocate its storage and scratch, and create it.
    fn create_accel(
        &self,
        ty: vk::AccelerationStructureTypeKHR,
        flags: BuildAccelFlags,
        geometries: &[vk::AccelerationStructureGeometryKHR<'_>],
        primitive_counts: &[u32],
    ) -> RhiResult<AccelerationStructure> {
        let accel_loader = &self.loaders.acceleration_structure;
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            accel_loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_geometry_info(ty, flags, geometries),
                Some(primitive_counts),
                &mut sizes,
            );
        }
        let size = sizes.acceleration_structure_size;

        // `create_acceleration_structure2` takes an address range, so the storage is an ordinary
        // suballocation rather than a dedicated buffer.
        let (backing, backing_address) = self.allocate_accel_range(size, 1, BLOCK_BUFFER_USAGE)?;
        let create_info = vk::AccelerationStructureCreateInfo2KHR::default()
            .address_range(
                vk::DeviceAddressRangeKHR::default()
                    .address(backing_address)
                    .size(size),
            )
            .ty(ty);
        let acceleration_structure = unsafe {
            self.loaders
                .address_commands
                .create_acceleration_structure2(&create_info, None)
                .map_err(|e| RhiError::AllocationFailed(e.into()))?
        };
        let device_address = unsafe {
            accel_loader.get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                    .acceleration_structure(acceleration_structure),
            )
        };

        // The pool carves at the alignment asked for, so the range is exactly the scratch size.
        let (scratch, scratch_address) = self.allocate_accel_range(
            sizes.build_scratch_size,
            self.accel_scratch_alignment,
            SCRATCH_BUFFER_USAGE,
        )?;
        debug_assert_eq!(
            scratch_address % self.accel_scratch_alignment,
            0,
            "scratch must satisfy minAccelerationStructureScratchOffsetAlignment"
        );

        Ok(AccelerationStructure {
            inner: Box::new(VulkanAccelerationStructure {
                acceleration_structure,
                backing,
                device_address,
                scratch,
                scratch_address,
                buffer_pool: self.buffer_pool.clone(),
                accel_loader: accel_loader.clone(),
            }),
            _owner: None,
        })
    }

    /// Device-local storage for an acceleration structure or its build scratch, carved from the
    /// ordinary buffer pool.
    fn allocate_accel_range(
        &self,
        size: u64,
        align: u64,
        usage: vk::BufferUsageFlags,
    ) -> RhiResult<(BlockRange, u64)> {
        let mut requirements = self.buffer_requirements(size, usage)?;
        requirements.alignment = requirements.alignment.max(align).max(1);
        let memory_type = find_memorytype_index(
            &requirements,
            &self.device_memory_properties,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or_else(|| RhiError::AllocationFailed("no device-local memory type".into()))?;
        let sub = self.buffer_pool.borrow_mut().allocate(
            &self.loaders.device,
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

    pub fn destroy_accel(&self, accel: Box<VulkanAccelerationStructure>) {
        self.queue
            .release_resource(VulkanRetiredResource::Accel(accel));
    }
}

/// Vulkan's native instance layout matches [`TlasInstance`] byte-for-byte.
pub(crate) const TLAS_INSTANCE_STRIDE: usize = size_of::<TlasInstance>();

/// The public layout is already Vulkan's: an [`AccelHandle`](crate::AccelHandle) is the
/// structure's device address, which is what the instance descriptor wants.
pub(crate) fn encode_tlas_instance(dst: &mut [u8], inst: &TlasInstance) {
    dst[..TLAS_INSTANCE_STRIDE].copy_from_slice(inst.as_bytes());
}
