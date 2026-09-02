use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLSharedEvent;

use crate::RhiResult;

pub struct MetalTimelineSemaphore {
    pub(crate) event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
}

impl MetalTimelineSemaphore {
    pub fn value(&self) -> RhiResult<u64> {
        Ok(self.event.signaledValue())
    }

    pub fn wait(&self, value: u64, timeout_ns: u64) -> RhiResult<bool> {
        let timeout_ms = if timeout_ns == u64::MAX {
            u64::MAX
        } else {
            timeout_ns.saturating_add(999_999) / 1_000_000
        };
        Ok(self
            .event
            .waitUntilSignaledValue_timeoutMS(value, timeout_ms))
    }
}
