//! Progressive spectral path tracer over USD scenes, rendered through Kiln.
//!
//! Run with `cargo run -p spectra -- --spp 1024 --light-spectrum A`.
//! Windowed controls are WASD + Q/E, Shift to move faster, left-drag to look,
//! and R to restore the authored camera.

mod app;

use clap::Parser;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    app::run(app::Config::parse())
}
