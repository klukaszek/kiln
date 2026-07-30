use std::cell::Cell;
use std::rc::Rc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSArray;
use objc2_metal::{
    MTL4AccelerationStructureBoundingBoxGeometryDescriptor,
    MTL4AccelerationStructureGeometryDescriptor,
    MTL4AccelerationStructureTriangleGeometryDescriptor, MTL4BufferRange, MTLAccelerationStructure,
    MTLAccelerationStructureDescriptor, MTLAccelerationStructureUsage, MTLAllocation, MTLBuffer,
    MTLIndexType, MTLResidencySet,
};

use crate::types::{BlasDesc, BuildAccelFlags, GeometryFlags, GeometryType};

pub(crate) fn set_accel_usage(
    descriptor: &MTLAccelerationStructureDescriptor,
    flags: BuildAccelFlags,
) {
    let mut usage = MTLAccelerationStructureUsage::None;
    if flags.contains(BuildAccelFlags::ALLOW_UPDATE) {
        usage |= MTLAccelerationStructureUsage::Refit;
    }
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
    pub(crate) residency_set: Retained<ProtocolObject<dyn MTLResidencySet>>,
    pub(crate) residency_dirty: Rc<Cell<bool>>,
}

impl Drop for MetalAccelerationStructure {
    fn drop(&mut self) {
        let accel = unsafe {
            &*(self.acceleration_structure.as_ref()
                as *const ProtocolObject<dyn MTLAccelerationStructure>
                as *const ProtocolObject<dyn MTLAllocation>)
        };
        self.residency_set.removeAllocation(accel);
        let scratch = unsafe {
            &*(self.scratch_buffer.as_ref() as *const ProtocolObject<dyn MTLBuffer>
                as *const ProtocolObject<dyn MTLAllocation>)
        };
        self.residency_set.removeAllocation(scratch);
        self.residency_dirty.set(true);
    }
}

enum MetalBlasGeometryDescriptor {
    Triangle(Retained<MTL4AccelerationStructureTriangleGeometryDescriptor>),
    Aabb(Retained<MTL4AccelerationStructureBoundingBoxGeometryDescriptor>),
}

impl MetalBlasGeometryDescriptor {
    fn as_base(&self) -> &MTL4AccelerationStructureGeometryDescriptor {
        unsafe {
            match self {
                Self::Triangle(desc) => {
                    &*(desc.as_ref() as *const MTL4AccelerationStructureTriangleGeometryDescriptor
                        as *const MTL4AccelerationStructureGeometryDescriptor)
                }
                Self::Aabb(desc) => {
                    &*(desc.as_ref()
                        as *const MTL4AccelerationStructureBoundingBoxGeometryDescriptor
                        as *const MTL4AccelerationStructureGeometryDescriptor)
                }
            }
        }
    }
}

pub(crate) struct MetalBlasGeometryDescriptors {
    #[allow(dead_code)]
    descriptors: Vec<MetalBlasGeometryDescriptor>,
    pub(crate) array: Retained<NSArray<MTL4AccelerationStructureGeometryDescriptor>>,
}

pub(crate) fn make_blas_geometry_descriptors(desc: &BlasDesc) -> MetalBlasGeometryDescriptors {
    let mut descriptors = Vec::with_capacity(desc.meshes.len());

    for (mesh_index, mesh) in desc.meshes.iter().enumerate() {
        let descriptor = match mesh.geometry_type {
            GeometryType::Triangles => {
                let geo = MTL4AccelerationStructureTriangleGeometryDescriptor::new();
                unsafe {
                    geo.setVertexBuffer(MTL4BufferRange {
                        bufferAddress: mesh.vertex_buffer.0,
                        length: (mesh.vertex_count as u64) * mesh.vertex_stride,
                    });
                    geo.setVertexStride(mesh.vertex_stride as usize);
                    geo.setTriangleCount(triangle_primitive_count(mesh) as usize);
                    if mesh.index_count > 0 {
                        geo.setIndexBuffer(MTL4BufferRange {
                            bufferAddress: mesh.index_buffer.0,
                            length: (mesh.index_count as u64) * 4,
                        });
                        geo.setIndexType(MTLIndexType::UInt32);
                    }
                }
                MetalBlasGeometryDescriptor::Triangle(geo)
            }
            GeometryType::Aabbs => {
                let geo = MTL4AccelerationStructureBoundingBoxGeometryDescriptor::new();
                geo.setBoundingBoxBuffer(MTL4BufferRange {
                    bufferAddress: mesh.aabb_buffer.0,
                    length: (mesh.aabb_count as u64) * 24,
                });
                unsafe {
                    geo.setBoundingBoxStride(24);
                    geo.setBoundingBoxCount(mesh.aabb_count as usize);
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

    let geo_base_refs: Vec<&MTL4AccelerationStructureGeometryDescriptor> =
        descriptors.iter().map(|g| g.as_base()).collect();
    let array = NSArray::from_slice(&geo_base_refs);

    MetalBlasGeometryDescriptors { descriptors, array }
}

fn triangle_primitive_count(mesh: &crate::types::BlasMeshDesc) -> u32 {
    if mesh.index_count > 0 {
        mesh.index_count / 3
    } else {
        mesh.vertex_count / 3
    }
}
