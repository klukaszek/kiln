//! Metal timestamp query pool, backed by an `MTL4CounterHeap` of type
//! [`MTL4CounterHeapType::Timestamp`](objc2_metal::MTL4CounterHeapType).

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTL4CounterHeap;

/// Backing for a [`crate::QueryPool`] on Metal. The heap is reference-counted by ARC, so it is
/// released when this struct drops; `Device::destroy_query_pool` simply drops it.
pub struct MetalQueryPool {
    pub(crate) heap: Retained<ProtocolObject<dyn MTL4CounterHeap>>,
}
