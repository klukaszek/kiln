//! Acceleration structure types (BLAS + TLAS) for ray tracing.

#[cfg(feature = "metal")]
use crate::backend::metal::accel::{TLAS_INSTANCE_STRIDE, encode_tlas_instance};
#[cfg(feature = "vulkan")]
use crate::backend::vulkan::accel::{TLAS_INSTANCE_STRIDE, encode_tlas_instance};
use crate::error::{RhiError, RhiResult};
use crate::memory::Allocation;
use crate::types::{AccelHandle, BuildAccelFlags, GpuPtr, TlasDesc, TlasInstance};

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

/// A TLAS's instances, stored in the active backend's native instance layout, which is not
/// [`TlasInstance`]'s layout on Metal. Create with
/// [`Device::create_tlas_instances`](crate::Device::create_tlas_instances).
///
/// A [`DeviceResource`](crate::DeviceResource): release it with
/// [`Device::destroy`](crate::Device::destroy).
pub struct TlasInstances {
    pub(crate) allocation: Allocation,
    pub(crate) count: u32,
}

impl TlasInstances {
    pub(crate) const STRIDE: usize = TLAS_INSTANCE_STRIDE;

    pub fn len(&self) -> u32 {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Encode `instance` into slot `index`. The caller orders the write before the submit that
    /// builds the TLAS.
    pub fn write(&mut self, index: u32, instance: &TlasInstance) -> RhiResult<()> {
        if index >= self.count {
            return Err(RhiError::AllocationFailed(
                format!(
                    "TLAS instance {index} is out of range ({} slots)",
                    self.count
                )
                .into(),
            ));
        }
        let start = index as usize * Self::STRIDE;
        let bytes = &mut self.allocation.as_mut_slice()?[start..start + Self::STRIDE];
        encode_tlas_instance(bytes, instance);
        Ok(())
    }

    /// Address of the first instance, for [`TlasDesc::instance_buffer`].
    pub fn gpu(&self) -> GpuPtr<TlasInstance> {
        self.allocation.gpu().cast()
    }

    /// A TLAS over every instance.
    pub fn tlas_desc(&self, flags: BuildAccelFlags) -> TlasDesc {
        TlasDesc {
            instance_buffer: self.gpu(),
            instance_count: self.count,
            flags,
        }
    }

    pub fn labeled(self, label: &str) -> Self {
        Self {
            allocation: self.allocation.labeled(label),
            count: self.count,
        }
    }
}
