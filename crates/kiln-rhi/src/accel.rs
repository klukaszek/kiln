//! Acceleration structure types (BLAS + TLAS) for ray tracing.

use crate::types::{AccelHandle, AccelerationStructureId, GpuAddress};

/// A built acceleration structure (BLAS or TLAS). Build it with `cmd.build_blas`/`build_tlas`,
/// then store [`handle()`](Self::handle) in an `AccelHandle` field for the shader.
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
    /// root.tlas = tlas.handle();
    /// ```
    pub fn handle(&self) -> AccelHandle {
        AccelHandle::from_raw(self.address())
    }

    pub(crate) fn address(&self) -> GpuAddress {
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
