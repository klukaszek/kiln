use kiln_rhi::{BumpAllocator, Device, GpuPod, GpuPtr, MAX_FRAMES_IN_FLIGHT, MemoryType};

use crate::render::Result;

/// Per-frame transient GPU roots, shared by renderer backends.
pub(crate) struct FrameArenas(Vec<BumpAllocator>);

impl FrameArenas {
    pub(crate) fn new(device: &Device, size: u64, label: &str) -> Result<Self> {
        let mut slots = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        for slot in 0..MAX_FRAMES_IN_FLIGHT {
            let slot_label = format!("{label}-{slot}");
            match device.allocate_bytes(size, MemoryType::Upload) {
                Ok(buffer) => slots.push(BumpAllocator::new(buffer.labeled(&slot_label))),
                Err(error) => {
                    destroy_slots(device, slots);
                    return Err(error.into());
                }
            }
        }

        Ok(Self(slots))
    }

    pub(crate) fn reset(&mut self, slot: usize) {
        self.0[slot].reset();
    }

    pub(crate) fn upload<T: GpuPod>(&mut self, slot: usize, value: &T) -> GpuPtr<T> {
        self.0[slot].upload(value).expect("frame arena exhausted")
    }

    pub(crate) fn destroy(self, device: &Device) {
        destroy_slots(device, self.0);
    }
}

fn destroy_slots(device: &Device, slots: impl IntoIterator<Item = BumpAllocator>) {
    for slot in slots {
        device.destroy(slot.into_allocation());
    }
}
