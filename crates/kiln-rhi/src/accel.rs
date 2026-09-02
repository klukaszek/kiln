//! Acceleration structure types (BLAS + TLAS) for ray tracing. An AS is just GPU memory: its
//! address goes into a root struct and the shader dereferences it via `TraceRayInline`.

use crate::types::{AccelerationStructureId, GpuAddress};

/// A built acceleration structure (BLAS or TLAS). Build it with `cmd.build_blas`/`build_tlas`,
/// then store [`gpu()`](Self::gpu) in a root `GpuAddress` field for the shader.
pub struct AccelerationStructure {
    pub id: AccelerationStructureId,
    pub(crate) inner: AccelInner,
    pub(crate) _owner: Option<std::rc::Rc<crate::device::DeviceInner>>,
}

impl AccelerationStructure {
    /// GPU handle for this acceleration structure.
    ///
    /// Assign to an [`AccelHandle`](crate::AccelHandle) field in a root struct:
    ///
    /// ```ignore
    /// gpu_struct! {
    ///     pub struct TraceRoot {
    ///         tlas: AccelHandle,
    ///     }
    /// }
    /// root.tlas = tlas_accel.gpu();
    /// ```
    ///
    /// Vulkan: acceleration-structure device address
    /// (`vkGetAccelerationStructureDeviceAddressKHR`). Metal: `gpuResourceID`. Slang
    /// converts the stored 64-bit value to a `RaytracingAccelerationStructure` handle
    /// at the use site. No descriptor set or argument-table slot required.
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
