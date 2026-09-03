//! Acceleration structure types (BLAS + TLAS) for ray tracing.

use crate::types::AccelHandle;

/// A BLAS or TLAS. Build with `cmd.build_blas`/`build_tlas`, then pass [`gpu()`](Self::gpu) to
/// the shader.
pub struct AccelerationStructure {
    pub(crate) inner: AccelInner,
    pub(crate) _owner: Option<std::rc::Rc<crate::device::DeviceInner>>,
}

impl AccelerationStructure {
    /// Opaque shader handle; assign to an [`AccelHandle`] field in root data.
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
