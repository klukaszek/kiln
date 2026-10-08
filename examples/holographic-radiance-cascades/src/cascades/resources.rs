//! GPU-side scene state: the packed primitives, the tessellated BVH geometry, the broadphase grid,
//! and the acceleration structures over them.
//!
//! Everything here is sized once for [`MAX_PRIMS`] and rebuilt in place each frame. The original
//! sized its acceleration structures once for the same reason: releasing one and creating a
//! replacement every frame is a lot of churn for geometry whose topology never changes.

use glam::Vec4;
use kiln_rhi::{
    AccelerationStructure, Allocation, BlasDesc, BlasGeometry, BlasIndices, BlasMeshDesc,
    BuildAccelFlags, Device, GeometryFlags, GpuPtr, MemoryType, RhiResult, TlasDesc, TlasInstance,
    TlasInstances,
};

use crate::scene::{MAX_PRIMS, ROWS_PER_PRIM};

use super::program::{EDGES_PER_PRIM, GRID_CAPACITY, GRID_CELLS};

/// Vertices per tessellated outline edge: the edge extruded into a wall quad.
const VERTS_PER_EDGE: u32 = 4;
/// Indices per wall quad, as two triangles.
const INDICES_PER_EDGE: u32 = 6;
/// Bytes per BVH vertex: three tightly packed floats.
const VERTEX_STRIDE: u64 = 12;

/// Outline edges across a full-capacity scene.
const MAX_EDGES: u32 = MAX_PRIMS * EDGES_PER_PRIM;

pub(super) struct SceneResources {
    /// Packed primitives, `ROWS_PER_PRIM` rows each.
    pub(super) scene: Allocation<Vec4>,
    pub(super) vertices: Allocation<[f32; 3]>,
    pub(super) indices: Allocation<u32>,
    /// Primitive that produced each BVH triangle, so a hit can be resolved to its material.
    pub(super) tri_prim: Allocation<u32>,
    pub(super) cell_count: Allocation<u32>,
    pub(super) cell_prims: Allocation<u32>,
    /// Per-cell distance to the nearest primitive, from the cell's bounding circle outwards.
    pub(super) cell_clear: Allocation<f32>,
    /// Sized by the backend's instance stride, which is not `size_of::<TlasInstance>()` on Metal.
    instances: TlasInstances,
    blas: AccelerationStructure,
    pub(super) tlas: AccelerationStructure,
    /// Primitives in the scene currently uploaded.
    pub(super) prim_count: u32,
}

impl SceneResources {
    pub(super) fn new(device: &Device) -> RhiResult<Self> {
        fn device_buffer<T>(device: &Device, count: u32, label: &str) -> RhiResult<Allocation<T>> {
            Ok(device
                .allocate_array::<T>(count as usize, MemoryType::GpuOnly)?
                .labeled(label))
        }

        let scene = device_buffer(device, MAX_PRIMS * ROWS_PER_PRIM, "hrc-scene-primitives")?;
        let vertices = device_buffer(device, MAX_EDGES * VERTS_PER_EDGE, "hrc-bvh-vertices")?;
        let indices = device_buffer(device, MAX_EDGES * INDICES_PER_EDGE, "hrc-bvh-indices")?;
        let tri_prim = device_buffer(device, MAX_EDGES * 2, "hrc-bvh-triangle-primitive")?;
        let cell_count = device_buffer(device, GRID_CELLS * GRID_CELLS, "hrc-grid-count")?;
        let cell_prims = device_buffer(
            device,
            GRID_CELLS * GRID_CELLS * GRID_CAPACITY,
            "hrc-grid-primitives",
        )?;
        let cell_clear = device_buffer(device, GRID_CELLS * GRID_CELLS, "hrc-grid-clearance")?;

        // Sized for the largest scene that will ever be traced; each frame builds whatever the
        // current one actually uses.
        let meshes = [blas_mesh(vertices.gpu(), indices.gpu(), MAX_PRIMS)];
        let blas = device.create_blas(&BlasDesc {
            meshes: &meshes,
            flags: BLAS_FLAGS,
        })?;

        let mut instances = device
            .create_tlas_instances(1)?
            .labeled("hrc-tlas-instances");
        // One instance at the identity, written once: the BLAS is rebuilt in place, so neither its
        // handle nor its transform ever changes.
        instances.write(
            0,
            &TlasInstance {
                transform: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                instance_custom_index_and_mask: 0xFF << 24,
                instance_sbt_offset_and_flags: 0,
                acceleration_structure_reference: blas.gpu(),
            },
        )?;
        let tlas = device.create_tlas(&tlas_desc(&instances))?;

        Ok(Self {
            scene,
            vertices,
            indices,
            tri_prim,
            cell_count,
            cell_prims,
            cell_clear,
            instances,
            blas,
            tlas,
            prim_count: 0,
        })
    }

    pub(super) fn blas_mesh(&self) -> BlasMeshDesc {
        blas_mesh(self.vertices.gpu(), self.indices.gpu(), self.prim_count)
    }

    pub(super) fn tlas_desc(&self) -> TlasDesc {
        tlas_desc(&self.instances)
    }

    pub(super) fn blas(&self) -> &AccelerationStructure {
        &self.blas
    }

    pub(super) fn destroy(self, device: &Device) {
        device.destroy(self.scene);
        device.destroy(self.vertices);
        device.destroy(self.indices);
        device.destroy(self.tri_prim);
        device.destroy(self.cell_count);
        device.destroy(self.cell_prims);
        device.destroy(self.cell_clear);
        device.destroy(self.instances);
        device.destroy(self.blas);
        device.destroy(self.tlas);
    }
}

/// The geometry is rewritten every frame, so build time is the cost that matters.
pub(super) const BLAS_FLAGS: BuildAccelFlags = BuildAccelFlags::PREFER_FAST_BUILD;

fn blas_mesh(vertices: GpuPtr<[f32; 3]>, indices: GpuPtr<u32>, prims: u32) -> BlasMeshDesc {
    let edges = prims * EDGES_PER_PRIM;
    BlasMeshDesc {
        flags: GeometryFlags::OPAQUE,
        geometry: BlasGeometry::Triangles {
            vertices,
            stride: VERTEX_STRIDE,
            count: edges * VERTS_PER_EDGE,
            indices: Some(BlasIndices {
                buffer: indices,
                count: edges * INDICES_PER_EDGE,
            }),
        },
    }
}

fn tlas_desc(instances: &TlasInstances) -> TlasDesc {
    instances.tlas_desc(BuildAccelFlags::PREFER_FAST_BUILD)
}
