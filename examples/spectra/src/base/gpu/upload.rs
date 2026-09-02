//! Batched uploads shared by renderer-owned device data.

use std::marker::PhantomData;

use kiln_rhi::{Allocation, CommandBuffer, Device, GpuPod, GpuPtr, MemoryType};

use crate::base::renderer::Result;

pub(crate) struct GpuArray<T> {
    allocation: Allocation,
    len: u32,
    marker: PhantomData<fn() -> T>,
}

impl<T> GpuArray<T> {
    pub(crate) fn new(allocation: Allocation, len: usize) -> Self {
        Self {
            allocation,
            len: len as u32,
            marker: PhantomData,
        }
    }

    pub(crate) fn gpu(&self) -> GpuPtr<T> {
        self.allocation.ptr()
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
    allocations: Vec<Allocation>,
    staging: Vec<Allocation>,
}

/// Batch range updates into existing device-local arrays. All patches are submitted together so
/// one editor transaction does not wait on the GPU once per field or mesh-light component.
pub(crate) struct GpuPatchBatch<'a> {
    device: &'a Device,
    command: Option<CommandBuffer>,
    staging: Vec<Allocation>,
}

impl<'a> GpuPatchBatch<'a> {
    pub(crate) fn new(device: &'a Device) -> Result<Self> {
        Ok(Self {
            device,
            command: Some(device.create_command_buffer()?),
            staging: Vec::new(),
        })
    }

    pub(crate) fn patch<T: GpuPod>(
        &mut self,
        array: &GpuArray<T>,
        start: usize,
        values: &[T],
    ) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }
        let byte_offset = start.checked_mul(std::mem::size_of::<T>()).ok_or(
            crate::base::renderer::Error::Capacity("GPU array patch offset exceeds usize capacity"),
        )?;
        let byte_size = std::mem::size_of_val(values);
        let byte_end =
            byte_offset
                .checked_add(byte_size)
                .ok_or(crate::base::renderer::Error::Capacity(
                    "GPU array patch exceeds usize capacity",
                ))?;
        if byte_end as u64 > array.allocation.size() {
            return Err(crate::base::renderer::Error::Capacity(
                "GPU array patch exceeds allocation bounds",
            ));
        }
        let staging = self.device.upload_slice(values)?;
        self.command
            .as_mut()
            .expect("patch batch already finished")
            .memcpy(
                array.allocation.gpu().offset(byte_offset as u64),
                staging.gpu(),
                byte_size as u64,
            );
        self.staging.push(staging);
        Ok(())
    }

    /// Submit the patch copies and return their staging allocations. The caller must retain them
    /// until the frame fence covering this submission signals.
    pub(crate) fn finish(mut self) -> Result<Vec<Allocation>> {
        let mut command = self.command.take().expect("patch batch already finished");
        command.end();
        self.device.queue().submit(command)?;
        Ok(std::mem::take(&mut self.staging))
    }
}

impl Drop for GpuPatchBatch<'_> {
    fn drop(&mut self) {
        for staging in self.staging.drain(..) {
            self.device.free(staging);
        }
    }
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
        let allocation = self.device.allocate(
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

    pub(crate) fn finish<const N: usize>(self) -> Result<[Allocation; N]> {
        let allocations = self.finish_vec()?;
        match allocations.try_into() {
            Ok(allocations) => Ok(allocations),
            Err(_) => unreachable!("upload count is fixed at the call site"),
        }
    }

    pub(crate) fn finish_vec(mut self) -> Result<Vec<Allocation>> {
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
        Ok(std::mem::take(&mut self.allocations))
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
