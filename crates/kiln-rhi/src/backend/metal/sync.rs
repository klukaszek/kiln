use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLSharedEvent};

use super::device::MetalDevice;
use crate::error::{RhiError, RhiResult};
use crate::sync::TimelineSemaphore;

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

impl MetalDevice {
    pub fn create_timeline_semaphore(&self, initial_value: u64) -> RhiResult<TimelineSemaphore> {
        let event = self
            .shared
            .device
            .newSharedEvent()
            .ok_or_else(|| RhiError::SyncError("Failed to create MTLSharedEvent".into()))?;
        event.setSignaledValue(initial_value);

        Ok(TimelineSemaphore {
            inner: Box::new(MetalTimelineSemaphore { event }),
            _owner: None,
        })
    }
}
