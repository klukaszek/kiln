//! Progressive spectral path tracer over USD scenes, rendered through Kiln.
//!
//! Run with `cargo run -p spectra -- --spp 1024 --light-spectrum A`.
//! Windowed controls are WASD + Q/E, Shift to move faster, left-drag to look,
//! and R to restore the authored camera.

mod config;
mod controls;
mod headless;
mod output;
mod viewer;

use clap::Parser;

use config::Config;

#[derive(Debug, thiserror::Error)]
enum Error {
    #[error(transparent)]
    Rhi(#[from] kiln_rhi::RhiError),
    #[error(transparent)]
    Render(#[from] spectra::render::Error),
    #[error(transparent)]
    Usd(#[from] spectra::usd::Error),
    #[error(transparent)]
    Spectrum(#[from] spectra::spectrum::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Png(#[from] png::EncodingError),
    #[error("invalid configuration: {0}")]
    Config(String),
    #[error("invalid output: {0}")]
    Output(String),
}

type Result<T> = std::result::Result<T, Error>;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let config = Config::parse();
    if let Some(resolution) = config.headless {
        headless::run(&config, resolution)?;
        Ok(())
    } else {
        viewer::run(config)
    }
}
