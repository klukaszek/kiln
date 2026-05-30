//! Acceleration structure types (BLAS + TLAS) for ray tracing. An AS is just GPU memory: its
//! address goes into a root struct and the shader dereferences it via `TraceRayInline`.

use crate::types::{AccelerationStructureId, GpuAddress};

/// A built acceleration structure (BLAS or TLAS). Build it with `cmd.build_blas`/`build_tlas`,
/// then store [`gpu()`](Self::gpu) in a root `GpuAddress` field for the shader.
pub struct AccelerationStructure {
    pub id: AccelerationStructureId,
    pub(crate) inner: AccelInner,
}

impl AccelerationStructure {
    /// GPU address of this structure, for a root `GpuAddress` field (`TraceRayInline`).
    pub fn gpu(&self) -> GpuAddress {
        match &self.inner {
            #[cfg(feature = "vulkan")]
            AccelInner::Vulkan(a) => GpuAddress(a.device_address),
            #[cfg(feature = "metal")]
            AccelInner::Metal(a) => GpuAddress(a.gpu_resource_id),
        }
    }
}

pub(crate) enum AccelInner {
    #[cfg(feature = "vulkan")]
    Vulkan(Box<crate::backend::vulkan::accel::VulkanAccelerationStructure>),
    #[cfg(feature = "metal")]
    Metal(Box<crate::backend::metal::accel::MetalAccelerationStructure>),
}

pub use crate::types::{
    BlasDesc, BlasMeshDesc, BuildAccelFlags, GeometryFlags, GeometryType, InstanceFlags, TlasDesc,
    TlasInstance,
};
