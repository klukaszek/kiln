use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLHeap};

use crate::types::GpuAddress;

pub struct MetalBuffer {
    pub(crate) buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) heap: Option<Retained<ProtocolObject<dyn MTLHeap>>>,
    pub(crate) size: u64,
    pub(crate) is_shared: bool,
}

impl MetalBuffer {
    pub fn mapped_ptr(&self) -> Option<*mut u8> {
        if self.is_shared {
            Some(self.buffer.contents().as_ptr() as *mut u8)
        } else {
            None
        }
    }

    pub fn gpu_address(&self) -> GpuAddress {
        GpuAddress(self.buffer.gpuAddress())
    }

    pub fn size(&self) -> u64 {
        self.size
    }
}
