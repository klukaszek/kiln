//! GPU-side scene state: the packed primitives, the tessellated BVH geometry, the broadphase grid,
//! and the acceleration structures over them.
//!
//! Everything here is sized once for [`MAX_PRIMS`] and rebuilt in place each frame. The original
//! sized its acceleration structures once for the same reason: releasing one and creating a
//! replacement every frame is a lot of churn for geometry whose topology never changes.

use kiln_rhi::{
    AccelerationStructure, Allocation, AllocationDesc, BlasDesc, BlasGeometry, BlasIndices,
    BlasMeshDesc, BuildAccelFlags, Device, GeometryFlags, GpuPtr, MemoryType, RhiResult, TlasDesc,
    TlasInstance,
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
    /// Packed primitives, `ROWS_PER_PRIM` `float4` rows each.
    pub(super) scene: Allocation,
    pub(super) vertices: Allocation,
    pub(super) indices: Allocation,
    /// Primitive that produced each BVH triangle, so a hit can be resolved to its material.
    pub(super) tri_prim: Allocation,
    pub(super) cell_count: Allocation,
    pub(super) cell_prims: Allocation,
    /// Per-cell distance to the nearest primitive, from the cell's bounding circle outwards.
    pub(super) cell_clear: Allocation,
    instances: Allocation,
    blas: AccelerationStructure,
    pub(super) tlas: AccelerationStructure,
    /// Primitives in the scene currently uploaded.
    pub(super) prim_count: u32,
}

impl SceneResources {
    pub(super) fn new(device: &Device) -> RhiResult<Self> {
        let device_buffer = |size: u64, label: &'static str| {
            device.create_allocation(&AllocationDesc {
                size,
                memory: MemoryType::GpuOnly,
                label: Some(label),
                ..Default::default()
            })
        };

        let scene = device_buffer(
            u64::from(MAX_PRIMS * ROWS_PER_PRIM) * 16,
            "hrc-scene-primitives",
        )?;
        let vertices = device_buffer(
            u64::from(MAX_EDGES * VERTS_PER_EDGE) * VERTEX_STRIDE,
            "hrc-bvh-vertices",
        )?;
        let indices = device_buffer(
            u64::from(MAX_EDGES * INDICES_PER_EDGE) * 4,
            "hrc-bvh-indices",
        )?;
        let tri_prim = device_buffer(u64::from(MAX_EDGES * 2) * 4, "hrc-bvh-triangle-primitive")?;
        let cell_count = device_buffer(u64::from(GRID_CELLS * GRID_CELLS) * 4, "hrc-grid-count")?;
        let cell_prims = device_buffer(
            u64::from(GRID_CELLS * GRID_CELLS * GRID_CAPACITY) * 4,
            "hrc-grid-primitives",
        )?;
        let cell_clear =
            device_buffer(u64::from(GRID_CELLS * GRID_CELLS) * 4, "hrc-grid-clearance")?;

        // Sized for the largest scene that will ever be traced; each frame builds whatever the
        // current one actually uses.
        let meshes = [blas_mesh(
            vertices.gpu().cast(),
            indices.gpu().cast(),
            MAX_PRIMS,
        )];
        let blas = device.create_blas(&BlasDesc {
            meshes: &meshes,
            flags: BLAS_FLAGS,
        })?;

        let mut instances = device.create_allocation(&AllocationDesc {
            size: device.tlas_instance_stride() as u64,
            memory: MemoryType::Upload,
            label: Some("hrc-tlas-instances"),
            ..Default::default()
        })?;
        // One instance at the identity, written once: the BLAS is rebuilt in place, so neither its
        // handle nor its transform ever changes.
        device.write_tlas_instance(
            &mut instances,
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
        blas_mesh(
            self.vertices.gpu().cast(),
            self.indices.gpu().cast(),
            self.prim_count,
        )
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

fn tlas_desc(instances: &Allocation) -> TlasDesc {
    TlasDesc {
        instance_buffer: instances.gpu().cast(),
        instance_count: 1,
        flags: BuildAccelFlags::PREFER_FAST_BUILD,
    }
}
