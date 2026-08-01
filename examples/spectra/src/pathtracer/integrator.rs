//! Spectral light-transport compute kernel.

pub const THREADS_X: u32 = 8;
pub const THREADS_Y: u32 = 8;

const SOURCE_PARTS: [&str; 4] = [
    include_str!("integrator/sampling.slang"),
    include_str!("integrator/bsdf.slang"),
    include_str!("integrator/direct_light.slang"),
    include_str!("integrator/trace.slang"),
];

/// Complete Slang source for the transport kernel, specialized to immutable scene and renderer
/// settings.
pub fn source(pixel_stride: u32, light_count: u32, spectrum_len: u32) -> String {
    let phase_count = pixel_stride
        .checked_mul(pixel_stride)
        .expect("validated pixel stride squared");
    // The host does not dispatch tracing for an empty-light scene, but keep the generated
    // denominator/index expressions valid because the shader is still compiled in that case.
    let light_count = light_count.max(1);
    let spectrum_len = spectrum_len.max(1);
    format!(
        "static const uint SPECTRAL_BINS = {}u;\n\
         static const uint LIGHT_LANE_COUNT = {}u;\n\
         static const uint UNIFORM_LANE_COUNT = {}u;\n\
         static const uint LIGHT_COUNT = {light_count}u;\n\
         static const uint SPECTRUM_LEN = {spectrum_len}u;\n\
         static const uint PIXEL_STRIDE = {pixel_stride}u;\n\
         static const uint PHASE_COUNT = {phase_count}u;\n\n{}",
        super::super::scene::spectral::SPECTRAL_BINS,
        super::N_LIGHT_LANES,
        super::N_UNIFORM_LANES,
        SOURCE_PARTS.join("\n\n")
    )
}
