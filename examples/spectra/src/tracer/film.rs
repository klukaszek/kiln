use glam::{UVec2, Vec3};
use kiln_rhi::{Allocation, Device, Format, MemoryType};

use crate::render::{self as render, Error};

use super::frame::CameraGpu;
use super::program::{DISPLAY_GAMMA, EXPOSURE};
use super::sampling::SpatialSchedule;

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
                    device.destroy(stale);
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
        device.destroy(readback);
        result
    }

    pub(super) fn destroy(mut self, device: &Device) {
        if let Some(accum) = self.accum.take() {
            device.destroy(accum);
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

pub(super) fn format_is_srgb(format: Format) -> bool {
    matches!(format, Format::R8G8B8A8Srgb | Format::B8G8R8A8Srgb)
}

/// CPU twin of `displayFs`'s tonemap for headless readback: exposure, Reinhard, then gamma. Shares
/// its constants with the shader.
fn tonemap_linear(linear: f32) -> u8 {
    let exposed = linear * EXPOSURE;
    let reinhard = exposed / (exposed + 1.0);
    let srgb = reinhard.max(0.0).powf(1.0 / DISPLAY_GAMMA);
    (srgb.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Resolve one film row to a pre-exposure linear-sRGB colour, summing the per-bin sensor response.
fn resolve_linear(row: &[f32], sample_count: u32, cmf: &[Vec3]) -> Vec3 {
    let inv = 1.0 / sample_count.max(1) as f32;
    cmf.iter()
        .enumerate()
        .map(|(j, c)| *c * (row[j] * inv))
        .sum()
}

/// Tonemap the whole film to RGBA8 for PNG readback, mirroring the blit.
pub(super) fn to_rgba8(
    rows: &[f32],
    stride: usize,
    extent: UVec2,
    schedule: SpatialSchedule,
    pass_count: u32,
    cmf: &[Vec3],
    spectral_capture: bool,
) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(rows.len() / stride * 4);
    for ((x, y), row) in pixels(extent).zip(rows.chunks_exact(stride)) {
        let sample_count = schedule.sample_count_for_pixel(pass_count, x, y);
        let lin = if spectral_capture {
            resolve_linear(row, sample_count, cmf)
        } else {
            Vec3::from_slice(&row[..3]) / sample_count.max(1) as f32
        };
        rgba.extend_from_slice(&[
            tonemap_linear(lin.x),
            tonemap_linear(lin.y),
            tonemap_linear(lin.z),
            255,
        ]);
    }
    rgba
}

/// Per-pixel band-integrated spectral radiance `bin / count`, row-major `[height][width][bins]`.
pub(super) fn to_bands(
    rows: &[f32],
    stride: usize,
    extent: UVec2,
    schedule: SpatialSchedule,
    pass_count: u32,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows.len());
    for ((x, y), row) in pixels(extent).zip(rows.chunks_exact(stride)) {
        let inv = 1.0 / schedule.sample_count_for_pixel(pass_count, x, y).max(1) as f32;
        out.extend(row.iter().map(|&b| b * inv));
    }
    out
}

fn pixels(extent: UVec2) -> impl Iterator<Item = (u32, u32)> {
    (0..extent.y).flat_map(move |y| (0..extent.x).map(move |x| (x, y)))
}
