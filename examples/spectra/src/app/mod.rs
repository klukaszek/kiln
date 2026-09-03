//! Command-line and windowed application wiring.

mod config;
mod controls;
mod headless;
mod output;
mod ui;
mod viewer;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Rhi(#[from] kiln_rhi::RhiError),
    #[error(transparent)]
    Render(#[from] spectra::render::Error),
    #[error(transparent)]
    Usd(#[from] spectra::importers::usd::Error),
    #[error(transparent)]
    Spectrum(#[from] spectra::tracer::spectrum::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Image(#[from] image::ImageError),
    #[error("invalid configuration: {0}")]
    Config(String),
    #[error("invalid output: {0}")]
    Output(String),
}

pub type Result<T> = std::result::Result<T, Error>;

pub use config::Config;

pub fn run(config: Config) -> std::result::Result<(), Box<dyn std::error::Error>> {
    if let Some(resolution) = config.headless {
        headless::run(&config, resolution)?;
        Ok(())
    } else {
        viewer::run(config)
    }
}
