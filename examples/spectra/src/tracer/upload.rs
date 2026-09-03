//! Batched uploads shared by renderer-owned device data.

use std::marker::PhantomData;

use kiln_rhi::{
    Allocation, CommandBuffer, Device, GpuPod, GpuPtr, MemoryType, StageFlags, Texture,
};

use crate::render::Result;

/// A device-local array of `T`. The type parameter is what keeps sixteen same-shaped scene buffers
/// from being swapped for one another.
pub(crate) struct GpuArray<T> {
    allocation: Allocation,
    len: u32,
    marker: PhantomData<fn() -> T>,
}

impl<T> GpuArray<T> {
    pub(crate) fn gpu(&self) -> GpuPtr<T> {
        self.allocation.gpu().cast()
    }

    pub(crate) fn len(&self) -> u32 {
        self.len
    }

    /// Swap in a freshly built array and free the one it displaces.
    pub(crate) fn replace(&mut self, device: &Device, new: Self) {
        std::mem::replace(self, new).destroy(device);
    }

    pub(crate) fn destroy(self, device: &Device) {
        device.destroy(self.allocation);
    }
}

/// Cap on host-visible staging held between flushes.
///
/// Staging is a second copy of everything in flight, so holding a whole scene's worth doubles peak
/// device memory. A texture set alone can run to gigabytes; flushing on a budget bounds the
/// overhead at the cost of a few extra queue stalls during load.
const STAGING_BUDGET: u64 = 128 << 20;

/// Stage buffers and texture contents into device memory.
///
/// Copies are recorded as they are added and submitted in batches, so loading a scene costs a
/// handful of queue stalls rather than one per resource. An array is returned as soon as its
/// memory exists and holds its data once [`submit`](Self::submit) returns. A batch that fails
/// partway strands the arrays it already handed back; every caller abandons construction on error,
/// so device teardown reclaims them.
pub(crate) struct GpuUploadBatch<'a> {
    device: &'a Device,
    commands: Option<CommandBuffer>,
    staging: Vec<Allocation>,
    staged_bytes: u64,
}

impl<'a> GpuUploadBatch<'a> {
    pub(crate) fn new(device: &'a Device) -> Result<Self> {
        Ok(Self {
            device,
            commands: Some(device.create_command_buffer()?),
            staging: Vec::new(),
            staged_bytes: 0,
        })
    }

    pub(crate) fn upload<T: GpuPod>(&mut self, data: &[T]) -> Result<GpuArray<T>> {
        let allocation = self.device.allocate(
            (std::mem::size_of_val(data) as u64).max(1),
            MemoryType::GpuOnly,
        )?;
        let staging = match self.device.upload_slice(data) {
            Ok(staging) => staging,
            Err(error) => {
                self.device.destroy(allocation);
                return Err(error.into());
            }
        };
        self.commands()
            .memcpy(allocation.gpu(), staging.gpu(), allocation.size());
        self.stage(staging)?;
        Ok(GpuArray {
            allocation,
            len: data.len() as u32,
            marker: PhantomData,
        })
    }

    /// Stage `rgba8` into the base mip of `texture`, whose memory the caller already owns.
    pub(crate) fn upload_texture(&mut self, rgba8: &[u8], texture: &Texture) -> Result<()> {
        let staging = self.device.upload_slice(rgba8)?;
        self.commands()
            .copy_buffer_to_texture(staging.gpu(), texture);
        self.stage(staging)
    }

    /// Take ownership of a staging allocation, flushing first if too much is already in flight.
    fn stage(&mut self, staging: Allocation) -> Result<()> {
        self.staged_bytes += staging.size();
        self.staging.push(staging);
        if self.staged_bytes >= STAGING_BUDGET {
            self.flush()?;
            self.commands = Some(self.device.create_command_buffer()?);
        }
        Ok(())
    }

    /// Submit whatever is recorded, wait for it, and release its staging.
    fn flush(&mut self) -> Result<()> {
        let Some(mut commands) = self.commands.take() else {
            return Ok(());
        };
        // Texture copies leave the image in a transfer layout; open it to every later stage.
        commands.barrier(StageFlags::TRANSFER, StageFlags::ALL_COMMANDS);
        commands.end();
        self.device.queue().submit(commands)?;
        self.device.queue().wait_idle();
        for staging in self.staging.drain(..) {
            self.device.destroy(staging);
        }
        self.staged_bytes = 0;
        Ok(())
    }

    fn commands(&mut self) -> &mut CommandBuffer {
        self.commands
            .as_mut()
            .expect("upload batch is already submitted")
    }

    /// Run everything still recorded and wait for it, so every destination is readable on return.
    pub(crate) fn submit(&mut self) -> Result<()> {
        self.flush()
    }
}

impl Drop for GpuUploadBatch<'_> {
    fn drop(&mut self) {
        for staging in self.staging.drain(..) {
            self.device.destroy(staging);
        }
    }
}
