use objc2::runtime::ProtocolObject;
use objc2_metal::MTLAllocation;

pub mod accel;
pub mod barrier;
pub mod command;
pub mod device;
pub mod heap;
pub mod memory;
pub mod pipeline;
pub mod query;
pub mod queue;
pub mod sampler;
pub mod shader;
pub mod surface;
pub mod swapchain;
pub mod sync;
pub mod texture;

// Barriers have no module of their own: Metal 4 encodes them directly on the command encoder,
// so they live in `command.rs` beside the encoder state they act on.

/// Reinterpret a `ProtocolObject` as another protocol the object conforms to; objc2 has no upcast.
///
/// # Safety
/// The object's class must conform to `Q`.
pub(crate) unsafe fn cast_protocol<P: ?Sized, Q: ?Sized>(
    object: &ProtocolObject<P>,
) -> &ProtocolObject<Q> {
    unsafe { &*std::ptr::from_ref(object).cast::<ProtocolObject<Q>>() }
}

/// `MTLBuffer`, `MTLTexture` and `MTLAccelerationStructure` all conform to `MTLAllocation`, which
/// is what the residency-set API takes.
pub(crate) fn as_allocation<P: ?Sized>(
    object: &ProtocolObject<P>,
) -> &ProtocolObject<dyn MTLAllocation> {
    // SAFETY: every caller passes a Metal resource, all of which conform to MTLAllocation.
    unsafe { cast_protocol(object) }
}
