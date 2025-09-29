#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

pub mod kiln {
    pub mod swapchain;
    pub mod windowing;
    pub mod events;
}

pub mod renderer;
pub use renderer as shared_renderer;

// Example binaries live under `examples/` and are not part of the library API.
