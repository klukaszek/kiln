//! Metal timestamp query pool, backed by an `MTL4CounterHeap` of type
//! [`MTL4CounterHeapType::Timestamp`](objc2_metal::MTL4CounterHeapType).

use std::borrow::Cow;

use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{MTL4CounterHeap, MTL4CounterHeapDescriptor, MTL4CounterHeapType, MTLDevice};
use zerocopy::FromBytes;

use super::command::MetalCommandBuffer;
use super::device::MetalDevice;
use crate::error::{RhiError, RhiResult};
use crate::query::QueryPool;

pub struct MetalQueryPool {
    pub(crate) heap: Retained<ProtocolObject<dyn MTL4CounterHeap>>,
}

impl MetalDevice {
    pub fn create_query_pool(&self, count: u32) -> RhiResult<QueryPool> {
        let desc = MTL4CounterHeapDescriptor::new();
        desc.setType(MTL4CounterHeapType::Timestamp);
        // SAFETY: a plain entry count.
        unsafe { desc.setCount(count as usize) };
        let heap = self
            .shared
            .device
            .newCounterHeapWithDescriptor_error(&desc)
            .map_err(|e| RhiError::Backend(format!("newCounterHeapWithDescriptor: {e}").into()))?;
        // Unwritten slots resolve to zero.
        // SAFETY: the range is the whole heap.
        unsafe { heap.invalidateCounterRange(NSRange::new(0, count as usize)) };
        Ok(QueryPool {
            inner: MetalQueryPool { heap },
            count,
            _owner: None,
        })
    }

    pub fn destroy_query_pool(&self, _pool: MetalQueryPool) {}

    pub fn timestamp_period_ns(&self) -> f64 {
        let freq = self.shared.device.queryTimestampFrequency();
        if freq == 0 { 0.0 } else { 1.0e9 / freq as f64 }
    }

    pub fn read_timestamps_into(
        &self,
        pool: &MetalQueryPool,
        count: u32,
        out: &mut [u64],
    ) -> RhiResult<()> {
        // `resolveCounterRange` returns autoreleased NSData and the event loop does not guarantee
        // a pool per render callback, so hold one here or App Memory grows every frame.
        autoreleasepool(|_| {
            // SAFETY: the caller has waited for the frame that wrote these slots.
            let data = unsafe {
                pool.heap
                    .resolveCounterRange(NSRange::new(0, count as usize))
            }
            .ok_or_else(|| RhiError::Backend("resolveCounterRange returned nil".into()))?;
            // SAFETY: `data` stays alive and immutable for the borrow below.
            let bytes = unsafe { data.as_bytes_unchecked() };
            // One packed `u64` per slot. Copied only if the data comes back unaligned.
            let resolved: Cow<'_, [u64]> = match <[u64]>::ref_from_bytes(bytes) {
                Ok(slice) => Cow::Borrowed(slice),
                Err(_) => Cow::Owned(
                    bytes
                        .chunks_exact(size_of::<u64>())
                        .map(|c| u64::from_ne_bytes(c.try_into().expect("chunk is 8 bytes")))
                        .collect(),
                ),
            };
            let n = out.len().min(resolved.len());
            out[..n].copy_from_slice(&resolved[..n]);
            Ok(())
        })
    }
}

impl MetalCommandBuffer {
    /// Invalidate `count` counter slots so unwritten ones resolve to zero.
    ///
    /// Unlike Vulkan's `vkCmdResetQueryPool` this is not recorded: `invalidateCounterRange` runs
    /// on the CPU the moment it is called, so the caller must already have waited for the
    /// previous GPU use of this pool. `CommandBuffer::reset_queries` documents that contract.
    pub(crate) fn reset_queries(&mut self, pool: &MetalQueryPool, count: u32) {
        // SAFETY: `count` is the pool's own slot count, so the range is inside the heap.
        unsafe {
            pool.heap
                .invalidateCounterRange(NSRange::new(0, count as usize))
        };
    }
}
