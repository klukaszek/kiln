//! Progressive spectral path tracer over USD scenes, rendered through Kiln.
//!
//! Run with `cargo run -p spectra -- --spp 1024 --light-spectrum A`.
//! Windowed controls are WASD + Q/E, Shift to move faster, left-drag to look,
//! and R to restore the authored camera.

mod config;
mod controls;
mod export;
mod frame_arena;
mod headless;
mod pathtracer;
mod png;
mod raster;
mod scene;
mod viewer;

use clap::Parser;

use config::Config;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::parse();
    if let Some(resolution) = config.headless {
        headless::run(&config, resolution)?;
        Ok(())
    } else {
        viewer::run(config)
    }
}
