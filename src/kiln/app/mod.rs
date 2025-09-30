pub mod events;
pub mod renderer;
pub mod swapchain;
pub mod windowing;
pub mod app; // runtime entrypoints and traits

// Re-export so users can `use kiln::app::{run_app, KilnApp, ...}`
pub use app::*;
