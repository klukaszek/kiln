//! Renderer lifecycle traits and per-frame context.

mod arena;
mod upload;

pub(crate) use arena::FrameArenas;
pub(crate) use upload::{GpuArray, GpuUploadBatch};

use glam::UVec2;
use kiln_rhi::{CommandBuffer, Device, Format};

use crate::scene::Camera;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Rhi(#[from] kiln_rhi::RhiError),
    #[error("invalid renderer settings: {0}")]
    Settings(&'static str),
    #[error("unsupported renderer feature: {0}")]
    Unsupported(&'static str),
    #[error("renderer capacity exceeded: {0}")]
    Capacity(&'static str),
}

pub type Result<T> = std::result::Result<T, Error>;

/// GPU context shared by windowed and headless rendering work.
pub struct RenderFrame<'a> {
    pub device: &'a Device,
    pub extent: UVec2,
    pub slot: usize,
}

/// Core GPU rendering lifecycle shared by all backends.
pub trait Renderer {
    fn encode(
        &mut self,
        frame: &RenderFrame<'_>,
        commands: &mut CommandBuffer,
        camera: &Camera,
    ) -> Result<()>;

    fn destroy(self: Box<Self>, device: &Device);
}

/// Presentation capability for renderers that draw into a window target.
pub trait PresentRenderer: Renderer {
    fn depth_format(&self) -> Option<Format> {
        None
    }

    fn encode_present(&mut self, frame: &RenderFrame<'_>, commands: &mut CommandBuffer);
}
