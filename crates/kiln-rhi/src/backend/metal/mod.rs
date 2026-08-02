use objc2::runtime::ProtocolObject;
use objc2_metal::MTLAllocation;

pub mod accel;
pub mod barrier;
pub mod command;
pub mod device;
pub mod memory;
pub mod pipeline;
pub mod query;
pub mod shader;
pub mod surface;
pub mod swapchain;
pub mod sync;
pub mod texture;

/// `MTLBuffer`, `MTLTexture` and `MTLAccelerationStructure` all conform to `MTLAllocation`, but
/// objc2 models each protocol as its own type with no upcast between them, so reaching the
/// residency-set API needs a cast.
pub(crate) fn as_allocation<P: ?Sized>(
    object: &ProtocolObject<P>,
) -> &ProtocolObject<dyn MTLAllocation> {
    // SAFETY: every caller passes an object whose class conforms to MTLAllocation, and
    // ProtocolObject is a transparent wrapper around the object pointer.
    unsafe { &*(object as *const ProtocolObject<P> as *const ProtocolObject<dyn MTLAllocation>) }
}
