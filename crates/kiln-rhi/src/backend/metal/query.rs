//! Metal timestamp query pool, backed by an `MTL4CounterHeap` of type
//! [`MTL4CounterHeapType::Timestamp`](objc2_metal::MTL4CounterHeapType).

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTL4CounterHeap;

/// Backing for a [`crate::QueryPool`] on Metal. The heap is reference-counted by ARC, so it is
/// released when this struct drops; `Device::destroy` simply drops it.
pub struct MetalQueryPool {
    pub(crate) heap: Retained<ProtocolObject<dyn MTL4CounterHeap>>,
}

use objc2_metal::{MTL4CounterHeapDescriptor, MTL4CounterHeapType, MTLDevice};

use super::device::MetalDevice;
use crate::error::{RhiError, RhiResult};
use crate::query::{QueryPool, QueryPoolInner};

impl MetalDevice {
    pub fn create_query_pool(&self, count: u32) -> RhiResult<QueryPool> {
        use objc2_foundation::NSRange;

        let desc = MTL4CounterHeapDescriptor::new();
        desc.setType(MTL4CounterHeapType::Timestamp);
        // SAFETY: count is the heap entry count; not bounds-checked by the API.
        unsafe { desc.setCount(count as usize) };
        let heap = self
            .shared
            .device
            .newCounterHeapWithDescriptor_error(&desc)
            .map_err(|e| RhiError::Backend(format!("newCounterHeapWithDescriptor: {e}")))?;
        // Invalidate the new heap so unwritten slots resolve to zero.
        unsafe {
            heap.invalidateCounterRange(NSRange {
                location: 0,
                length: count as usize,
            })
        };
        Ok(QueryPool {
            inner: QueryPoolInner::Metal(MetalQueryPool { heap }),
            count,
            _owner: None,
        })
    }

    pub fn destroy_query_pool(&self, _pool: QueryPool) {}

    pub fn timestamp_period_ns(&self) -> f64 {
        let freq = self.shared.device.queryTimestampFrequency();
        if freq == 0 { 0.0 } else { 1.0e9 / freq as f64 }
    }

    pub fn read_timestamps(&self, pool: &QueryPool) -> RhiResult<Vec<u64>> {
        use objc2::rc::autoreleasepool;
        use objc2_foundation::NSRange;

        // MTL4CounterHeap::resolveCounterRange returns a newly allocated autoreleased NSData.
        // The window event loop does not guarantee a pool around every render callback, so keep
        // the resolve and byte copy inside an explicit pool; otherwise the HUD's App Memory grows
        // once per frame even though Metal residency remains flat.
        autoreleasepool(|_| {
            let heap = match &pool.inner {
                QueryPoolInner::Metal(p) => &p.heap,
                #[allow(unreachable_patterns)]
                _ => unreachable!("query pool backend does not match device backend"),
            };
            // The writing frame has completed, so resolve the timestamp heap directly.
            let range = NSRange {
                location: 0,
                length: pool.count as usize,
            };
            let data = unsafe { heap.resolveCounterRange(range) }
                .ok_or_else(|| RhiError::Backend("resolveCounterRange returned nil".into()))?;
            let mut out = vec![0u64; pool.count as usize];
            // `NSData::as_bytes_unchecked` is safe here: `data` remains alive and immutable for
            // the whole copy, and Metal returns exactly one packed u64 per counter slot.
            for (slot, chunk) in out
                .iter_mut()
                .zip(unsafe { data.as_bytes_unchecked() }.chunks_exact(8))
            {
                *slot =
                    u64::from_ne_bytes(chunk.try_into().expect("chunks_exact(8) yields 8 bytes"));
            }
            Ok(out)
        })
    }
}
