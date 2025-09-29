//! Runtime-facing API surface only. Example code lives in examples/.
#![allow(clippy::too_many_arguments)]

// Re-export types needed to implement a runtime outside the library.
pub use crate::kiln::events::{self, AppEvent, ElementState, EventQueue, Modifiers, MouseButton, TouchPhase};
pub use crate::kiln::renderer::{self, Renderer};
pub use crate::kiln::swapchain::{self, ColorSpace, PresentMode, RenderSurface, SwapchainConfig};
pub use crate::kiln::windowing;

// No demo runner here; see examples/kiln_example.rs
