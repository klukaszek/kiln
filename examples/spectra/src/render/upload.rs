//! Batched uploads shared by renderer-owned device data.

use std::marker::PhantomData;

use kiln_rhi::{Device, GpuAddress, GpuAllocation, GpuPod, MemoryType};

use super::Result;

pub(crate) struct GpuArray<T> {
    allocation: GpuAllocation,
    len: u32,
    marker: PhantomData<fn() -> T>,
}

impl<T> GpuArray<T> {
    pub(crate) fn new(allocation: GpuAllocation, len: usize) -> Self {
        Self {
            allocation,
            len: len as u32,
            marker: PhantomData,
        }
    }

    pub(crate) fn gpu(&self) -> GpuAddress {
        self.allocation.gpu()
    }
    pub(crate) fn len(&self) -> u32 {
        self.len
    }
    pub(crate) fn destroy(self, device: &Device) {
        device.free(self.allocation);
    }
}

/// Stage immutable data into device-local allocations with one copy submission.
pub(crate) struct GpuUploadBatch<'a> {
    device: &'a Device,
    allocations: Vec<GpuAllocation>,
    staging: Vec<GpuAllocation>,
}

impl<'a> GpuUploadBatch<'a> {
    pub(crate) fn new(device: &'a Device) -> Self {
        Self {
            device,
            allocations: Vec::new(),
            staging: Vec::new(),
        }
    }

    pub(crate) fn upload<T: GpuPod>(&mut self, data: &[T]) -> Result<()> {
        let allocation = self.device.malloc(
            (std::mem::size_of_val(data) as u64).max(1),
            MemoryType::GpuOnly,
        )?;
        let staging = match self.device.upload_slice(data) {
            Ok(staging) => staging,
            Err(error) => {
                self.device.free(allocation);
                return Err(error.into());
            }
        };
        self.allocations.push(allocation);
        self.staging.push(staging);
        Ok(())
    }

    pub(crate) fn finish<const N: usize>(mut self) -> Result<[GpuAllocation; N]> {
        let mut cmd = self.device.create_command_buffer()?;
        for (destination, staging) in self.allocations.iter().zip(&self.staging) {
            cmd.memcpy(destination.gpu(), staging.gpu(), destination.size());
        }
        cmd.end();
        self.device.queue().submit(cmd)?;
        self.device.queue().wait_idle();
        for staging in self.staging.drain(..) {
            self.device.free(staging);
        }
        let allocations = std::mem::take(&mut self.allocations);
        match allocations.try_into() {
            Ok(allocations) => Ok(allocations),
            Err(_) => unreachable!("upload count is fixed at the call site"),
        }
    }
}

impl Drop for GpuUploadBatch<'_> {
    fn drop(&mut self) {
        for allocation in self.allocations.drain(..) {
            self.device.free(allocation);
        }
        for staging in self.staging.drain(..) {
            self.device.free(staging);
        }
    }
}
