//! Spectral light-transport compute kernel.

pub const THREADS_X: u32 = 8;
pub const THREADS_Y: u32 = 8;

const SOURCE_PARTS: [&str; 4] = [
    include_str!("integrator/sampling.slang"),
    include_str!("integrator/bsdf.slang"),
    include_str!("integrator/direct_light.slang"),
    include_str!("integrator/trace.slang"),
];

/// Complete Slang source for the transport kernel.
pub fn source() -> String {
    SOURCE_PARTS.join("\n\n")
}
