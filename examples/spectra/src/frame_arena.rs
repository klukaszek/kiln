use kiln_rhi::{
    BufferDesc, BumpAllocator, Device, GpuAddress, GpuPod, MAX_FRAMES_IN_FLIGHT, MemoryType,
};

const ROOT_ALIGNMENT: u64 = 16;

pub struct FrameArenas(Vec<BumpAllocator>);

impl FrameArenas {
    pub fn new(device: &Device, size: u64, label: &str) -> anyhow::Result<Self> {
        let mut slots = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        for slot in 0..MAX_FRAMES_IN_FLIGHT {
            match device.create_buffer(&BufferDesc {
                size,
                memory: MemoryType::Default,
                label: Some(format!("{label}-{slot}")),
            }) {
                Ok(buffer) => slots.push(BumpAllocator::new(buffer)),
                Err(error) => {
                    destroy_slots(device, slots);
                    return Err(error.into());
                }
            }
        }

        Ok(Self(slots))
    }

    pub fn reset(&mut self, slot: usize) {
        self.0[slot].reset();
    }

    pub fn upload<T: GpuPod>(&mut self, slot: usize, value: &T) -> GpuAddress {
        let allocation = self.0[slot]
            .alloc(std::mem::size_of::<T>() as u64, ROOT_ALIGNMENT)
            .expect("frame arena exhausted");
        allocation.upload(value).expect("upload frame data");
        allocation.gpu
    }

    pub fn destroy(self, device: &Device) {
        destroy_slots(device, self.0);
    }
}

fn destroy_slots(device: &Device, slots: impl IntoIterator<Item = BumpAllocator>) {
    for slot in slots {
        device.destroy_buffer(slot.into_buffer());
    }
}
