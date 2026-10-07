use std::rc::Rc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSArray;
use objc2_metal::{
    MTL4AccelerationStructureBoundingBoxGeometryDescriptor,
    MTL4AccelerationStructureGeometryDescriptor,
    MTL4AccelerationStructureTriangleGeometryDescriptor, MTL4BufferRange,
    MTL4InstanceAccelerationStructureDescriptor, MTL4PrimitiveAccelerationStructureDescriptor,
    MTLAccelerationStructure, MTLAccelerationStructureDescriptor,
    MTLAccelerationStructureInstanceOptions, MTLAccelerationStructureSizes,
    MTLAccelerationStructureUsage, MTLBuffer, MTLDevice, MTLIndexType,
    MTLIndirectAccelerationStructureInstanceDescriptor, MTLPackedFloat3, MTLPackedFloat4x3,
    MTLResidencySet, MTLResourceID, MTLResourceOptions,
};

use super::as_allocation;
use super::device::{MetalDevice, MetalRetiredResource, MetalShared};
use crate::accel::AccelerationStructure;
use crate::error::{RhiError, RhiResult};
use crate::types::{
    Aabb, BlasDesc, BlasGeometry, BuildAccelFlags, GeometryFlags, InstanceFlags, TlasDesc,
    TlasInstance,
};

pub struct MetalAccelerationStructure {
    pub(crate) acceleration_structure: Retained<ProtocolObject<dyn MTLAccelerationStructure>>,
    pub(crate) gpu_resource_id: u64,
    pub(crate) scratch_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) shared: Rc<MetalShared>,
}

impl MetalAccelerationStructure {
    /// Called by the queue once no submission can still be tracing against the structure.
    pub(crate) fn release_residency(&self) {
        let residency = &self.shared.residency_set;
        residency.removeAllocation(as_allocation(&self.acceleration_structure));
        residency.removeAllocation(as_allocation(&self.scratch_buffer));
        self.shared.residency_dirty.set(true);
    }
}

fn set_usage(descriptor: &MTLAccelerationStructureDescriptor, flags: BuildAccelFlags) {
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

fn geometry_descriptor(
    desc: &BlasDesc<'_>,
    mesh_index: usize,
) -> Retained<MTL4AccelerationStructureGeometryDescriptor> {
    let mesh = &desc.meshes[mesh_index];
    let descriptor: Retained<MTL4AccelerationStructureGeometryDescriptor> = match mesh.geometry {
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
                    length: u64::from(count) * stride,
                });
                geo.setVertexStride(stride as usize);
                geo.setTriangleCount(mesh.geometry.primitive_count() as usize);
                if let Some(indices) = indices {
                    geo.setIndexBuffer(MTL4BufferRange {
                        bufferAddress: indices.buffer.address,
                        length: u64::from(indices.count) * size_of::<u32>() as u64,
                    });
                    geo.setIndexType(MTLIndexType::UInt32);
                }
            }
            Retained::into_super(geo)
        }
        BlasGeometry::Aabbs { buffer, count } => {
            // `Aabb` matches Metal's bounding-box layout.
            const STRIDE: u64 = size_of::<Aabb>() as u64;
            let geo = MTL4AccelerationStructureBoundingBoxGeometryDescriptor::new();
            geo.setBoundingBoxBuffer(MTL4BufferRange {
                bufferAddress: buffer.address,
                length: u64::from(count) * STRIDE,
            });
            unsafe {
                geo.setBoundingBoxStride(STRIDE as usize);
                geo.setBoundingBoxCount(count as usize);
            }
            Retained::into_super(geo)
        }
    };
    descriptor.setOpaque(mesh.flags.contains(GeometryFlags::OPAQUE));
    descriptor.setAllowDuplicateIntersectionFunctionInvocation(
        !mesh.flags.contains(GeometryFlags::NO_DUPLICATE_ANYHIT),
    );
    unsafe { descriptor.setIntersectionFunctionTableOffset(mesh_index) };
    descriptor
}

/// The descriptor used both to size a BLAS and to build it.
pub(crate) fn blas_descriptor(
    desc: &BlasDesc<'_>,
) -> Retained<MTL4PrimitiveAccelerationStructureDescriptor> {
    let geometries: Vec<_> = (0..desc.meshes.len())
        .map(|i| geometry_descriptor(desc, i))
        .collect();
    let descriptor = MTL4PrimitiveAccelerationStructureDescriptor::new();
    descriptor.setGeometryDescriptors(Some(&NSArray::from_retained_slice(&geometries)));
    set_usage(&descriptor, desc.flags);
    descriptor
}

/// The descriptor used both to size a TLAS and to build it.
pub(crate) fn tlas_descriptor(
    desc: &TlasDesc,
) -> Retained<MTL4InstanceAccelerationStructureDescriptor> {
    let descriptor = MTL4InstanceAccelerationStructureDescriptor::new();
    unsafe {
        descriptor.setInstanceDescriptorBuffer(MTL4BufferRange {
            bufferAddress: desc.instance_buffer.address,
            length: u64::from(desc.instance_count) * TLAS_INSTANCE_STRIDE as u64,
        });
        descriptor.setInstanceCount(desc.instance_count as usize);
    }
    set_usage(&descriptor, desc.flags);
    descriptor
}

/// Metal's instance acceleration structure uses the indirect layout, which names each BLAS by
/// `gpuResourceID`.
const TLAS_INSTANCE_STRIDE: usize = size_of::<MTLIndirectAccelerationStructureInstanceDescriptor>();

impl MetalDevice {
    pub fn create_blas(&self, desc: &BlasDesc<'_>) -> RhiResult<AccelerationStructure> {
        let sizes = self
            .shared
            .device
            .accelerationStructureSizesWithDescriptor(&blas_descriptor(desc));
        self.allocate_accel(sizes, "BLAS")
    }

    pub fn create_tlas(&self, desc: &TlasDesc) -> RhiResult<AccelerationStructure> {
        let sizes = self
            .shared
            .device
            .accelerationStructureSizesWithDescriptor(&tlas_descriptor(desc));
        self.allocate_accel(sizes, "TLAS")
    }

    pub fn tlas_instance_stride(&self) -> usize {
        TLAS_INSTANCE_STRIDE
    }

    /// Encode `inst` into `dst` in Metal's indirect instance layout.
    /// `inst.acceleration_structure_reference` must be a BLAS handle (`blas.gpu()`).
    pub fn write_tlas_instance(&self, dst: &mut [u8], inst: &TlasInstance) {
        // `transform` is row-major 3x4; Metal's packed 4x3 is column-major.
        let t = &inst.transform;
        let col = |c: usize| MTLPackedFloat3 {
            x: t[0][c],
            y: t[1][c],
            z: t[2][c],
        };
        // SAFETY: the handle came from `gpuResourceID().to_raw()` on a live structure.
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
        assert!(dst.len() >= TLAS_INSTANCE_STRIDE);
        // SAFETY: `dst` holds at least one descriptor, checked above. Unaligned because the
        // caller's buffer is only stride-aligned.
        unsafe {
            std::ptr::write_unaligned(
                dst.as_mut_ptr()
                    .cast::<MTLIndirectAccelerationStructureInstanceDescriptor>(),
                desc,
            );
        }
    }

    /// Allocate a structure and its scratch buffer and make both resident.
    fn allocate_accel(
        &self,
        sizes: MTLAccelerationStructureSizes,
        label: &'static str,
    ) -> RhiResult<AccelerationStructure> {
        let device = &self.shared.device;
        let accel = device
            .newAccelerationStructureWithSize(sizes.accelerationStructureSize)
            .ok_or_else(|| {
                RhiError::AllocationFailed(format!("Failed to allocate Metal {label}").into())
            })?;
        let scratch = device
            .newBufferWithLength_options(
                sizes.buildScratchBufferSize,
                MTLResourceOptions::StorageModePrivate,
            )
            .ok_or_else(|| {
                RhiError::AllocationFailed(
                    format!("Failed to allocate {label} scratch buffer").into(),
                )
            })?;

        let residency = &self.shared.residency_set;
        residency.addAllocation(as_allocation(&accel));
        residency.addAllocation(as_allocation(&scratch));
        self.shared.residency_dirty.set(true);

        Ok(AccelerationStructure {
            inner: Box::new(MetalAccelerationStructure {
                gpu_resource_id: accel.gpuResourceID().to_raw(),
                acceleration_structure: accel,
                scratch_buffer: scratch,
                shared: self.shared.clone(),
            }),
            _owner: None,
        })
    }

    pub fn destroy_accel(&self, accel: Box<MetalAccelerationStructure>) {
        self.queue
            .release_resource(MetalRetiredResource::Accel(accel));
    }
}
