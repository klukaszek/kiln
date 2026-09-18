use std::rc::Rc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSArray;

use super::as_allocation;
use super::device::MetalShared;
use objc2_metal::{
    MTL4AccelerationStructureBoundingBoxGeometryDescriptor,
    MTL4AccelerationStructureGeometryDescriptor,
    MTL4AccelerationStructureTriangleGeometryDescriptor, MTL4BufferRange, MTLAccelerationStructure,
    MTLAccelerationStructureDescriptor, MTLAccelerationStructureUsage, MTLBuffer, MTLIndexType,
    MTLResidencySet,
};

use smallvec::SmallVec;

use super::device::{MetalDevice, MetalRetiredResource};
use crate::accel::AccelerationStructure;
use crate::error::{RhiError, RhiResult};
use crate::types::{Aabb, BlasDesc, BlasGeometry, BuildAccelFlags, GeometryFlags};
use crate::types::{InstanceFlags, TlasDesc};
use objc2_metal::MTLDevice;

pub(crate) fn set_accel_usage(
    descriptor: &MTLAccelerationStructureDescriptor,
    flags: BuildAccelFlags,
) {
    let mut usage = MTLAccelerationStructureUsage::None;
    if flags.contains(BuildAccelFlags::PREFER_FAST_TRACE) {
        usage |= MTLAccelerationStructureUsage::PreferFastIntersection;
    }
    if flags.contains(BuildAccelFlags::PREFER_FAST_BUILD) {
        usage |= MTLAccelerationStructureUsage::PreferFastBuild;
    }
    if flags.contains(BuildAccelFlags::MINIMIZE_MEMORY) {
        usage |= MTLAccelerationStructureUsage::MinimizeMemory;
    }
    descriptor.setUsage(usage);
}

pub struct MetalAccelerationStructure {
    pub(crate) acceleration_structure: Retained<ProtocolObject<dyn MTLAccelerationStructure>>,
    pub(crate) gpu_resource_id: u64,
    pub(crate) scratch_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) shared: Rc<MetalShared>,
}

impl MetalAccelerationStructure {
    /// Drop the structure's residency entries. Called from the queue's retirement path once every
    /// submission that could be tracing against it has completed — never from `Drop`, which would
    /// run while an in-flight frame still references the structure.
    pub(crate) fn release_residency(&self) {
        self.shared
            .residency_set
            .removeAllocation(as_allocation(&self.acceleration_structure));
        self.shared
            .residency_set
            .removeAllocation(as_allocation(&self.scratch_buffer));
        self.shared.residency_dirty.set(true);
    }
}

enum MetalBlasGeometryDescriptor {
    Triangle(Retained<MTL4AccelerationStructureTriangleGeometryDescriptor>),
    Aabb(Retained<MTL4AccelerationStructureBoundingBoxGeometryDescriptor>),
}

impl MetalBlasGeometryDescriptor {
    /// The common base every geometry descriptor inherits, for the flags that are not
    /// triangle- or AABB-specific.
    fn as_base(&self) -> &MTL4AccelerationStructureGeometryDescriptor {
        // SAFETY: both descriptor classes derive from
        // MTL4AccelerationStructureGeometryDescriptor.
        match self {
            Self::Triangle(desc) => unsafe {
                downcast_base::<MTL4AccelerationStructureTriangleGeometryDescriptor, _>(desc)
            },
            Self::Aabb(desc) => unsafe {
                downcast_base::<MTL4AccelerationStructureBoundingBoxGeometryDescriptor, _>(desc)
            },
        }
    }
}

/// Reinterpret an acceleration-structure descriptor as one of its base classes.
///
/// objc2 exposes these as plain structs rather than `ProtocolObject`s, so `cast_protocol` does not
/// apply and the cast is spelled out here once instead of at each of the five call sites.
///
/// # Safety
/// `Base` must actually be a base class of `Derived`.
pub(crate) unsafe fn downcast_base<Derived, Base>(derived: &Derived) -> &Base {
    unsafe { &*std::ptr::from_ref(derived).cast::<Base>() }
}

/// `NSArray` retains its elements, so the descriptors need no separate owner.
pub(crate) struct MetalBlasGeometryDescriptors {
    pub(crate) array: Retained<NSArray<MTL4AccelerationStructureGeometryDescriptor>>,
}

pub(crate) fn make_blas_geometry_descriptors(desc: &BlasDesc<'_>) -> MetalBlasGeometryDescriptors {
    let mut descriptors = Vec::with_capacity(desc.meshes.len());

    for (mesh_index, mesh) in desc.meshes.iter().enumerate() {
        let descriptor = match mesh.geometry {
            BlasGeometry::Triangles {
                vertices,
                stride,
                count,
                indices,
            } => {
                let geo = MTL4AccelerationStructureTriangleGeometryDescriptor::new();
                unsafe {
                    geo.setVertexBuffer(MTL4BufferRange {
                        bufferAddress: vertices.address,
                        length: (count as u64) * stride,
                    });
                    geo.setVertexStride(stride as usize);
                    geo.setTriangleCount(mesh.geometry.primitive_count() as usize);
                    if let Some(indices) = indices {
                        geo.setIndexBuffer(MTL4BufferRange {
                            bufferAddress: indices.buffer.address,
                            length: (indices.count as u64) * size_of::<u32>() as u64,
                        });
                        geo.setIndexType(MTLIndexType::UInt32);
                    }
                }
                MetalBlasGeometryDescriptor::Triangle(geo)
            }
            BlasGeometry::Aabbs { buffer, count } => {
                // `Aabb` is the layout Metal expects, so its size is the stride.
                const STRIDE: u64 = size_of::<Aabb>() as u64;
                let geo = MTL4AccelerationStructureBoundingBoxGeometryDescriptor::new();
                geo.setBoundingBoxBuffer(MTL4BufferRange {
                    bufferAddress: buffer.address,
                    length: (count as u64) * STRIDE,
                });
                unsafe {
                    geo.setBoundingBoxStride(STRIDE as usize);
                    geo.setBoundingBoxCount(count as usize);
                }
                MetalBlasGeometryDescriptor::Aabb(geo)
            }
        };

        let base = descriptor.as_base();
        base.setOpaque(mesh.flags.contains(GeometryFlags::OPAQUE));
        base.setAllowDuplicateIntersectionFunctionInvocation(
            !mesh.flags.contains(GeometryFlags::NO_DUPLICATE_ANYHIT),
        );
        unsafe {
            base.setIntersectionFunctionTableOffset(mesh_index);
        }
        descriptors.push(descriptor);
    }

    let bases: SmallVec<[&MTL4AccelerationStructureGeometryDescriptor; 4]> = descriptors
        .iter()
        .map(MetalBlasGeometryDescriptor::as_base)
        .collect();
    MetalBlasGeometryDescriptors {
        array: NSArray::from_slice(&bases),
    }
}

impl MetalDevice {
    pub fn create_blas(&self, desc: &BlasDesc<'_>) -> RhiResult<AccelerationStructure> {
        use super::accel::make_blas_geometry_descriptors;
        use objc2_metal::MTL4PrimitiveAccelerationStructureDescriptor;

        let geometries = make_blas_geometry_descriptors(desc);
        let primitive_desc = MTL4PrimitiveAccelerationStructureDescriptor::new();
        primitive_desc.setGeometryDescriptors(Some(&geometries.array));
        // SAFETY: MTL4PrimitiveAccelerationStructureDescriptor derives from
        // MTLAccelerationStructureDescriptor.
        let primitive_base: &objc2_metal::MTLAccelerationStructureDescriptor =
            unsafe { super::accel::downcast_base(&*primitive_desc) };
        super::accel::set_accel_usage(primitive_base, desc.flags);

        let sizes = self
            .shared
            .device
            .accelerationStructureSizesWithDescriptor(primitive_base);
        self.finalize_accel_structure(sizes, "BLAS")
    }

    pub fn create_tlas(&self, desc: &TlasDesc) -> RhiResult<AccelerationStructure> {
        use objc2_metal::MTL4InstanceAccelerationStructureDescriptor;

        let instance_desc = MTL4InstanceAccelerationStructureDescriptor::new();
        unsafe {
            instance_desc.setInstanceDescriptorBuffer(objc2_metal::MTL4BufferRange {
                bufferAddress: desc.instance_buffer.address,
                // The descriptor uses Metal's indirect instance layout; callers write it with
                // `Device::write_tlas_instance`.
                length: (desc.instance_count as u64) * self.tlas_instance_stride() as u64,
            });
            instance_desc.setInstanceCount(desc.instance_count as usize);
        }
        // SAFETY: MTL4InstanceAccelerationStructureDescriptor derives from
        // MTLAccelerationStructureDescriptor.
        let instance_base: &objc2_metal::MTLAccelerationStructureDescriptor =
            unsafe { super::accel::downcast_base(&*instance_desc) };
        super::accel::set_accel_usage(instance_base, desc.flags);

        let sizes = self
            .shared
            .device
            .accelerationStructureSizesWithDescriptor(instance_base);
        self.finalize_accel_structure(sizes, "TLAS")
    }

    /// Native size of one TLAS instance descriptor. Metal's instance acceleration structure
    /// uses the *indirect* descriptor layout (references the BLAS by `gpuResourceID`).
    pub fn tlas_instance_stride(&self) -> usize {
        std::mem::size_of::<objc2_metal::MTLIndirectAccelerationStructureInstanceDescriptor>()
    }

    /// Encode `inst` into `dst` in Metal's native indirect instance-descriptor layout.
    ///
    /// `inst.acceleration_structure_reference` must be the BLAS handle (`blas.gpu()`).
    pub fn write_tlas_instance(&self, dst: &mut [u8], inst: &crate::types::TlasInstance) {
        use objc2_metal::{
            MTLAccelerationStructureInstanceOptions,
            MTLIndirectAccelerationStructureInstanceDescriptor, MTLPackedFloat3, MTLPackedFloat4x3,
            MTLResourceID,
        };

        // TlasInstance.transform is row-major 3x4 (transform[row][col]); Metal's packed 4x3
        // is column-major (columns[c] = (m[0][c], m[1][c], m[2][c])).
        let t = &inst.transform;
        let col = |c: usize| MTLPackedFloat3 {
            x: t[0][c],
            y: t[1][c],
            z: t[2][c],
        };
        // SAFETY: the handle came from `MTLAccelerationStructure::gpuResourceID().to_raw()`
        // in `create_blas` / `create_tlas`, so it is a live resource ID for this device.
        let resource_id =
            unsafe { MTLResourceID::from_raw(inst.acceleration_structure_reference.0) };

        let instance_flags =
            InstanceFlags::from_bits_retain((inst.instance_sbt_offset_and_flags >> 24) as u8);
        let mut options = MTLAccelerationStructureInstanceOptions::empty();
        if instance_flags.contains(InstanceFlags::TRIANGLE_FACING_CULL_DISABLE) {
            options |= MTLAccelerationStructureInstanceOptions::DisableTriangleCulling;
        }
        if instance_flags.contains(InstanceFlags::TRIANGLE_FLIP_FACING) {
            options |=
                MTLAccelerationStructureInstanceOptions::TriangleFrontFacingWindingCounterClockwise;
        }
        if instance_flags.contains(InstanceFlags::FORCE_OPAQUE) {
            options |= MTLAccelerationStructureInstanceOptions::Opaque;
        }
        if instance_flags.contains(InstanceFlags::FORCE_NO_OPAQUE) {
            options |= MTLAccelerationStructureInstanceOptions::NonOpaque;
        }

        let desc = MTLIndirectAccelerationStructureInstanceDescriptor {
            transformationMatrix: MTLPackedFloat4x3 {
                columns: [col(0), col(1), col(2), col(3)],
            },
            options,
            mask: (inst.instance_custom_index_and_mask >> 24) & 0xFF,
            intersectionFunctionTableOffset: inst.instance_sbt_offset_and_flags & 0x00FF_FFFF,
            userID: inst.instance_custom_index_and_mask & 0x00FF_FFFF,
            accelerationStructureID: resource_id,
        };
        assert!(dst.len() >= size_of::<MTLIndirectAccelerationStructureInstanceDescriptor>());
        // SAFETY: `dst` is at least one descriptor long, checked above, and `&mut` rules out
        // aliasing. Unaligned because the caller's buffer is only stride-aligned.
        unsafe {
            std::ptr::write_unaligned(
                dst.as_mut_ptr()
                    .cast::<MTLIndirectAccelerationStructureInstanceDescriptor>(),
                desc,
            );
        }
    }

    /// Allocate the acceleration structure + scratch buffer for `sizes`, register both
    /// with the residency set and query the GPU resource ID. Shared by `create_blas`/`create_tlas`.
    fn finalize_accel_structure(
        &self,
        sizes: objc2_metal::MTLAccelerationStructureSizes,
        label: &'static str,
    ) -> RhiResult<AccelerationStructure> {
        use super::accel::MetalAccelerationStructure;
        use objc2_metal::{MTLAccelerationStructure as _, MTLDevice, MTLResourceOptions};

        let accel = self
            .shared
            .device
            .newAccelerationStructureWithSize(sizes.accelerationStructureSize)
            .ok_or_else(|| {
                RhiError::AllocationFailed(format!("Failed to allocate Metal {label}").into())
            })?;
        let scratch = self
            .shared
            .device
            .newBufferWithLength_options(
                sizes.buildScratchBufferSize,
                MTLResourceOptions::StorageModePrivate,
            )
            .ok_or_else(|| {
                RhiError::AllocationFailed(
                    format!("Failed to allocate {label} scratch buffer").into(),
                )
            })?;

        self.shared
            .residency_set
            .addAllocation(as_allocation(&accel));
        self.shared
            .residency_set
            .addAllocation(as_allocation(&scratch));
        self.shared.residency_dirty.set(true);

        let gpu_resource_id = accel.gpuResourceID().to_raw();

        Ok(AccelerationStructure {
            inner: Box::new(MetalAccelerationStructure {
                acceleration_structure: accel,
                gpu_resource_id,
                scratch_buffer: scratch,
                shared: self.shared.clone(),
            }),
            _owner: None,
        })
    }

    pub fn destroy_accel(&self, accel: Box<super::accel::MetalAccelerationStructure>) {
        self.queue
            .release_resource(MetalRetiredResource::Accel(accel));
    }
}
