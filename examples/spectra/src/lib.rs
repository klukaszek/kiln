//! Spectral path tracer over USD scenes.
//!
//! [`scene`] is the editable CPU scene the [`importers`] produce. [`tracer`] builds its own device
//! representation from a snapshot of it and renders progressively.

pub mod importers;
pub mod scene;
pub mod tracer;

pub mod render {
    //! The renderer's error type and per-frame context.

    use glam::UVec2;
    use kiln_rhi::Device;

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
}
