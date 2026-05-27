//! Acceleration structure types (BLAS + TLAS) for ray tracing.
//!
//! Following Aaltonen's model: the AS is just another piece of GPU memory.
//! The GPU address is placed into root structs as a `u64` and the shader
//! dereferences it with a `TraceRayInline` / `intersect(ray, as, ...)` intrinsic.

use crate::types::AccelerationStructureId;

/// A built acceleration structure (BLAS or TLAS).
///
/// Created by `device.create_blas(desc)` or `device.create_tlas(desc)`.
/// The `id` is opaque — pass it to `cmd.build_blas()` / `cmd.build_tlas()`.
/// The GPU address is obtained via `device.accel_gpu_address(id)`.
pub struct AccelerationStructure {
    pub id: AccelerationStructureId,
    pub(crate) inner: AccelInner,
}

pub(crate) enum AccelInner {
    #[cfg(feature = "vulkan")]
    Vulkan(Box<crate::backend::vulkan::accel::VulkanAccelerationStructure>),
    #[cfg(feature = "metal")]
    Metal(Box<crate::backend::metal::accel::MetalAccelerationStructure>),
}

pub use crate::types::{
    BlasDesc, BlasMeshDesc, BuildAccelFlags, GeometryFlags, GeometryType, InstanceFlags, SbtRegion,
    TlasDesc, TlasInstance,
};
