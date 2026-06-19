use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSArray;
use objc2_metal::{
    MTL4AccelerationStructureBoundingBoxGeometryDescriptor,
    MTL4AccelerationStructureGeometryDescriptor,
    MTL4AccelerationStructureTriangleGeometryDescriptor, MTL4BufferRange, MTLAccelerationStructure,
    MTLBuffer, MTLIndexType,
};

use crate::error::{RhiError, RhiResult};
use crate::types::{BlasDesc, GeometryFlags, GeometryType};

/// Metal acceleration structure entry (BLAS or TLAS).
///
/// Metal acceleration structures are MTLAccelerationStructure objects that live
/// in GPU memory.  Their GPU address (stored in `gpu_resource_id`) is placed into
/// root structs and accessed in intersection shaders / ray generation kernels via the
/// Metal `intersect(ray, accelerationStructure, ...)` intrinsic.
///
/// The raw `u64` GPU resource ID is cached at build time by extracting it via
/// `unsafe { std::mem::transmute(accelerationStructure.gpuResourceID()) }` —
/// the `MTLResourceID` struct is guaranteed to be a single `uint64_t` by the Metal spec.
pub struct MetalAccelerationStructure {
    /// The underlying Metal acceleration structure.
    pub(crate) acceleration_structure: Retained<ProtocolObject<dyn MTLAccelerationStructure>>,
    /// Opaque 64-bit resource ID (raw `MTLResourceID._impl`).
    /// Stored at build time so callers don't need objc2 access to read it.
    pub(crate) gpu_resource_id: u64,
    /// Scratch buffer used during build (freed after build completes in a one-shot command).
    /// Kept alive here until the next build or Drop to avoid UAF.
    #[allow(dead_code)]
    pub(crate) scratch_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
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

pub(crate) fn make_blas_geometry_descriptors(
    desc: &BlasDesc,
) -> RhiResult<MetalBlasGeometryDescriptors> {
    let mut descriptors = Vec::with_capacity(desc.meshes.len());

    for (mesh_index, mesh) in desc.meshes.iter().enumerate() {
        let descriptor = match mesh.geometry_type {
            GeometryType::Triangles => {
                if mesh.vertex_buffer.0 == 0 || mesh.vertex_count == 0 {
                    return Err(RhiError::Backend(
                        "triangle BLAS geometry requires a non-null vertex_buffer and vertex_count"
                            .into(),
                    ));
                }
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
                if mesh.aabb_buffer.0 == 0 || mesh.aabb_count == 0 {
                    return Err(RhiError::Backend(
                        "AABB BLAS geometry requires a non-null aabb_buffer and aabb_count".into(),
                    ));
                }
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

    Ok(MetalBlasGeometryDescriptors { descriptors, array })
}

fn triangle_primitive_count(mesh: &crate::types::BlasMeshDesc) -> u32 {
    if mesh.index_count > 0 {
        mesh.index_count / 3
    } else {
        mesh.vertex_count / 3
    }
}
