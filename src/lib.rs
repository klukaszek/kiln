#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

pub mod kiln {
    pub mod app;
    pub mod events;
    pub mod renderer;
    pub mod swapchain;
    pub mod windowing;
}

// Example binaries live under `examples/` and are not part of the library API.
