use glam::UVec2;
use kiln_rhi::{Device, GpuAllocation, MemoryType};

use super::roots::CameraGpu;

#[derive(Clone, Copy, PartialEq)]
pub struct FilmSignature {
    camera: CameraGpu,
    geometry_revision: u64,
    spectral_revision: u64,
    pixel_stride: u32,
    spectral_bins: u32,
}

impl FilmSignature {
    pub fn new(
        camera: CameraGpu,
        geometry_revision: u64,
        spectral_revision: u64,
        pixel_stride: u32,
        spectral_bins: u32,
    ) -> Self {
        Self {
            camera,
            geometry_revision,
            spectral_revision,
            pixel_stride,
            spectral_bins,
        }
    }
}

pub struct Film {
    accum: Option<GpuAllocation>,
    extent: UVec2,
    pass_count: u32,
    signature: Option<FilmSignature>,
    stride: u32,
}

impl Film {
    pub fn new(stride: u32) -> Self {
        Self {
            accum: None,
            extent: UVec2::ZERO,
            pass_count: 0,
            signature: None,
            stride,
        }
    }

    pub fn prepare(
        &mut self,
        device: &Device,
        extent: UVec2,
        signature: FilmSignature,
    ) -> anyhow::Result<bool> {
        if self.extent == extent && self.signature == Some(signature) && self.accum.is_some() {
            return Ok(false);
        }

        let bytes = self.byte_len(extent)?;
        let accum = match self.accum.take() {
            Some(existing) if self.extent == extent => existing,
            stale => {
                if let Some(stale) = stale {
                    device.free(stale);
                }
                device.malloc(bytes, MemoryType::GpuOnly)?
            }
        };

        self.accum = Some(accum);
        self.extent = extent;
        self.pass_count = 0;
        self.signature = Some(signature);
        Ok(true)
    }

    pub fn readback(&self, device: &Device) -> anyhow::Result<Vec<f32>> {
        let accum = self
            .accum
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("film has no accumulation buffer"))?;
        let readback = device.malloc(accum.size(), MemoryType::Readback)?;

        let result = (|| -> anyhow::Result<Vec<f32>> {
            device.wait_idle();
            let mut cmd = device.create_command_buffer()?;
            cmd.memcpy(readback.gpu(), accum.gpu(), accum.size());
            cmd.end();
            let queue = device.queue();
            queue.submit(cmd)?;
            queue.wait_idle();
            Ok(readback.as_slice::<f32>()?.to_vec())
        })();
        device.free(readback);
        result
    }

    pub fn destroy(mut self, device: &Device) {
        if let Some(accum) = self.accum.take() {
            device.free(accum);
        }
    }

    pub fn accum(&self) -> Option<&GpuAllocation> {
        self.accum.as_ref()
    }

    pub fn extent(&self) -> UVec2 {
        self.extent
    }

    pub fn pass_count(&self) -> u32 {
        self.pass_count
    }

    pub fn add_passes(&mut self, passes: u32) {
        self.pass_count += passes;
    }

    fn byte_len(&self, extent: UVec2) -> anyhow::Result<u64> {
        u64::from(extent.x)
            .checked_mul(u64::from(extent.y))
            .and_then(|pixels| pixels.checked_mul(u64::from(self.stride)))
            .and_then(|floats| floats.checked_mul(4))
            .ok_or_else(|| anyhow::anyhow!("spectral film size overflow for {extent:?}"))
    }
}
