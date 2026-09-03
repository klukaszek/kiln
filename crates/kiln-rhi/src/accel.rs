//! Acceleration structure types (BLAS + TLAS) for ray tracing.

use crate::types::{AccelHandle, AccelerationStructureId};

/// A built acceleration structure (BLAS or TLAS). Build it with `cmd.build_blas`/`build_tlas`,
/// then store [`gpu()`](Self::gpu) in an `AccelHandle` field for the shader.
pub struct AccelerationStructure {
    pub id: AccelerationStructureId,
    pub(crate) inner: AccelInner,
    pub(crate) _owner: Option<std::rc::Rc<crate::device::DeviceInner>>,
}

impl AccelerationStructure {
    /// Opaque shader handle for this acceleration structure.
    ///
    /// Assign it directly to an [`AccelHandle`] field in root data:
    ///
    /// ```ignore
    /// gpu_struct! {
    ///     pub struct TraceRoot {
    ///         tlas: AccelHandle,
    ///     }
    /// }
    /// root.tlas = tlas.gpu();
    /// ```
    pub fn gpu(&self) -> AccelHandle {
        let value = match &self.inner {
            #[cfg(feature = "vulkan")]
            AccelInner::Vulkan(a) => a.device_address,
            #[cfg(feature = "metal")]
            AccelInner::Metal(a) => a.gpu_resource_id,
        };
        AccelHandle::from_raw(value)
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
