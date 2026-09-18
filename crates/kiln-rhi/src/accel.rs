//! Acceleration structure types (BLAS + TLAS) for ray tracing.

use crate::types::AccelHandle;

/// A BLAS or TLAS. Build with `cmd.build_blas`/`build_tlas`, then pass [`gpu()`](Self::gpu) to
/// the shader.
///
/// A [`DeviceResource`](crate::DeviceResource): release it with
/// [`Device::destroy`](crate::Device::destroy), which holds the storage until the submissions
/// that trace against it have retired. Dropping one instead leaks it.
pub struct AccelerationStructure {
    pub(crate) inner: AccelInner,
    pub(crate) _owner: Option<std::rc::Rc<crate::device::DeviceInner>>,
}

impl AccelerationStructure {
    /// Opaque shader handle; assign to an [`AccelHandle`] field in root data or to a
    /// [`TlasInstance`](crate::TlasInstance)'s structure reference.
    ///
    /// Unlike [`Texture::gpu`](crate::Texture::gpu) this is not a bindless-heap index: each
    /// backend hands back what its own shaders and instance descriptors consume, a device address
    /// on Vulkan and a resource id on Metal. Shaders see neither, only the
    /// `RaytracingAccelerationStructure` the field's property returns.
    pub fn gpu(&self) -> AccelHandle {
        let value = match &self.inner {
            #[cfg(feature = "vulkan")]
            a => a.device_address,
            #[cfg(feature = "metal")]
            a => a.gpu_resource_id,
        };
        AccelHandle::from_raw(value)
    }
}

backend_enum!(AccelInner { vulkan: Box<crate::backend::vulkan::accel::VulkanAccelerationStructure>, metal: Box<crate::backend::metal::accel::MetalAccelerationStructure> });
