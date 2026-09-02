use glam::UVec2;
use kiln_rhi::{Allocation, Device, MemoryType};

use crate::base::renderer::{self as render, Error};

use super::shader_types::CameraGpu;

#[derive(Clone, Copy, PartialEq)]
pub(super) struct FilmSignature {
    camera: CameraGpu,
    pixel_stride: u32,
}

impl FilmSignature {
    pub(super) fn new(camera: CameraGpu, pixel_stride: u32) -> Self {
        Self {
            camera,
            pixel_stride,
        }
    }
}

pub(super) struct Film {
    accum: Option<Allocation>,
    pub(super) extent: UVec2,
    pub(super) pass_count: u32,
    signature: Option<FilmSignature>,
    stride: u32,
    pub(super) element_count: u32,
}

impl Film {
    pub(super) fn new(stride: u32) -> Self {
        Self {
            accum: None,
            extent: UVec2::ZERO,
            pass_count: 0,
            signature: None,
            stride,
            element_count: 0,
        }
    }

    pub(super) fn prepare(
        &mut self,
        device: &Device,
        extent: UVec2,
        signature: FilmSignature,
    ) -> render::Result<bool> {
        if self.extent == extent && self.signature == Some(signature) {
            return Ok(false);
        }

        let mut element_count = self.element_count;
        let accum = match self.accum.take() {
            Some(existing) if self.extent == extent => existing,
            stale => {
                if let Some(stale) = stale {
                    device.wait_idle();
                    device.free(stale);
                }
                element_count = self.element_count_for(extent)?;
                device.allocate(
                    u64::from(element_count) * std::mem::size_of::<f32>() as u64,
                    MemoryType::GpuOnly,
                )?
            }
        };

        self.accum = Some(accum);
        self.extent = extent;
        self.pass_count = 0;
        self.signature = Some(signature);
        self.element_count = element_count;
        Ok(true)
    }

    pub(super) fn readback(&self, device: &Device) -> render::Result<Vec<f32>> {
        let accum = self.accum();
        let readback = device.allocate(accum.size(), MemoryType::Readback)?;
        let result = copy_to_readback(device, accum, &readback);
        device.free(readback);
        result
    }

    pub(super) fn destroy(mut self, device: &Device) {
        if let Some(accum) = self.accum.take() {
            device.free(accum);
        }
    }

    /// Keep the allocation, but force the next frame to clear it and restart accumulation.
    pub(super) fn invalidate(&mut self) {
        self.pass_count = 0;
        self.signature = None;
    }

    pub(super) fn accum(&self) -> &Allocation {
        self.accum
            .as_ref()
            .expect("film allocation follows successful prepare")
    }

    fn element_count_for(&self, extent: UVec2) -> render::Result<u32> {
        extent
            .x
            .checked_mul(extent.y)
            .and_then(|pixels| pixels.checked_mul(self.stride))
            .ok_or(Error::Capacity(
                "spectral film exceeds u32 element capacity",
            ))
    }
}

fn copy_to_readback(
    device: &Device,
    source: &Allocation,
    destination: &Allocation,
) -> render::Result<Vec<f32>> {
    device.wait_idle();
    let mut cmd = device.create_command_buffer()?;
    cmd.memcpy(destination.gpu(), source.gpu(), source.size());
    cmd.end();
    let queue = device.queue();
    queue.submit(cmd)?;
    queue.wait_idle();
    Ok(destination.as_slice::<f32>()?.to_vec())
}
