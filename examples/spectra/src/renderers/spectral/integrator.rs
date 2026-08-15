//! Spectral light-transport compute kernel.

pub(super) const THREADS_X: u32 = 8;
pub(super) const THREADS_Y: u32 = 8;

const SOURCE_PARTS: [&str; 4] = [
    include_str!("integrator/sampling.slang"),
    include_str!("integrator/bsdf.slang"),
    include_str!("integrator/direct_light.slang"),
    include_str!("integrator/trace.slang"),
];

/// Complete Slang source for the transport kernel. Light count and spatial sampling settings are
/// supplied through TraceRoot, so editor changes do not require a pipeline rebuild.
pub(super) fn source(spectrum_len: u32) -> String {
    let spectrum_len = spectrum_len.max(1);
    format!(
        "static const uint SPECTRAL_BINS = {}u;\n\
         static const uint LIGHT_LANE_COUNT = {}u;\n\
         static const uint UNIFORM_LANE_COUNT = {}u;\n\
         static const uint LIGHT_TRIANGLE = 0u;\n\
         static const uint LIGHT_RECT = 1u;\n\
         static const uint LIGHT_DISK = 2u;\n\
         static const uint LIGHT_POINT = 3u;\n\
         static const uint LIGHT_DIRECTIONAL = 4u;\n\
         static const uint LIGHT_DOME = 5u;\n\
         static const uint LIGHT_SPHERE = 6u;\n\
         static const uint LIGHT_MESH = 7u;\n\
         static const uint SPECTRUM_LEN = {spectrum_len}u;\n\
         \n{}",
        super::spectrum::SPECTRAL_BINS,
        super::N_LIGHT_LANES,
        super::N_UNIFORM_LANES,
        SOURCE_PARTS.join("\n\n")
    )
}
