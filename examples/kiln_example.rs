//! Kiln example entry-point — backend-agnostic.
//! The selected backend (AppKit-only or Winit) is chosen at compile time
//! via crate features. The example logic lives in the library.

#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

fn main() {
    kiln::kiln::app::run_example();
}
