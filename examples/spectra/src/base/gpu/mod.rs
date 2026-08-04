//! RHI-backed resources shared by renderer storage implementations.

pub(crate) mod frame;
pub(crate) mod texture;
pub(crate) mod upload;

pub(crate) use frame::FrameArenas;
pub(crate) use texture::{GpuTextureBinding, TextureResources};
pub(crate) use upload::{GpuArray, GpuUploadBatch};
