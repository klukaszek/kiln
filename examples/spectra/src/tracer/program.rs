//! Shader source assembly and the pipeline objects built from it.
//!
//! Constants the host and shader both depend on are defined in Rust and emitted here. There is no
//! define mechanism in the RHI compiler, so the alternative is a second copy in Slang that nothing
//! keeps in step.

use kiln_rhi::{
    BlendState, ColorTarget, ComputePso, ComputePsoDesc, Cull, Device, Format, GraphicsPso,
    GraphicsPsoDesc, SampleCount, ShaderStage, Topology, compiler,
};

use super::device::GpuTextureBinding;
use crate::render;

use super::device::{
    GpuEmissiveHit, GpuInstance, GpuLight, GpuMaterial, GpuMeshLightTriangle, GpuTriangle, LightTag,
};
use super::frame::{ClearRoot, DisplayRoot, TraceRoot};
use super::spectrum;

/// Trace threadgroup shape. Dispatch and `[numthreads]` both read it from here.
pub(super) const TRACE_THREADS: [u32; 3] = [8, 8, 1];

/// Film-clear threadgroup width.
pub(super) const CLEAR_THREADS: u32 = 256;

/// Wavelengths carried per path by the MIS estimator.
pub(super) const WAVELENGTH_LANES: u32 = 4;

/// Scene-referred exposure applied before the Reinhard curve.
pub(super) const EXPOSURE: f32 = 0.25;

/// Encoding gamma for non-sRGB targets, which get no hardware transfer function.
pub(super) const DISPLAY_GAMMA: f32 = 2.2;

/// Floor on authored roughness, keeping the GGX distribution out of its singular case.
const MIN_GGX_ROUGHNESS: f32 = 0.045;

pub(super) fn trace() -> String {
    [
        &constants(),
        GpuMaterial::SLANG,
        GpuEmissiveHit::SLANG,
        GpuTextureBinding::SLANG,
        GpuLight::SLANG,
        GpuMeshLightTriangle::SLANG,
        GpuTriangle::SLANG,
        GpuInstance::SLANG,
        TraceRoot::SLANG,
        include_str!("shaders/sampler.slang"),
        include_str!("shaders/sampling.slang"),
        include_str!("shaders/material.slang"),
        include_str!("shaders/bsdf.slang"),
        include_str!("shaders/direct_light.slang"),
        include_str!("shaders/trace.slang"),
    ]
    .concat()
}

pub(super) fn clear() -> String {
    [
        &constants(),
        ClearRoot::SLANG,
        include_str!("shaders/clear.slang"),
    ]
    .concat()
}

pub(super) fn display() -> String {
    [
        &constants(),
        DisplayRoot::SLANG,
        include_str!("shaders/display.slang"),
    ]
    .concat()
}

/// Every scalar the shaders share with the host.
fn constants() -> String {
    let mut out = format!(
        "static const uint SPECTRAL_BINS = {bins}u;\n\
         static const uint SPECTRAL_BIN_VECS = {vecs}u;\n\
         static const uint SPECTRUM_LEN = {spectrum_len}u;\n\
         static const uint WAVELENGTH_LANE_COUNT = {lanes}u;\n\
         static const uint SOBOL_BYTES = {sobol_bytes}u;\n\
         static const uint SOBOL_VALUES = {sobol_values}u;\n\
         static const uint TRACE_THREADS_X = {threads_x}u;\n\
         static const uint TRACE_THREADS_Y = {threads_y}u;\n\
         static const uint CLEAR_THREADS = {CLEAR_THREADS}u;\n\
         static const float EXPOSURE = {EXPOSURE};\n\
         static const float DISPLAY_GAMMA = {DISPLAY_GAMMA};\n\
         static const float MIN_GGX_ROUGHNESS = {MIN_GGX_ROUGHNESS};\n\
         static const float3 REC709_LUMA = float3(0.2126, 0.7152, 0.0722);\n",
        bins = spectrum::SPECTRAL_BINS,
        vecs = spectrum::SPECTRAL_BINS / 4,
        spectrum_len = spectrum::DEFAULT_RESOLUTION,
        lanes = WAVELENGTH_LANES,
        sobol_bytes = super::sampling::SOBOL_BYTES,
        sobol_values = super::sampling::SOBOL_VALUES,
        threads_x = TRACE_THREADS[0],
        threads_y = TRACE_THREADS[1],
    );
    out.push_str(&LightTag::slang());
    out
}

pub(super) struct Pipelines {
    pub(super) trace: ComputePso,
    pub(super) clear: ComputePso,
    pub(super) display: GraphicsPso,
}

impl Pipelines {
    pub(super) fn new(device: &Device, color_format: Format) -> render::Result<Self> {
        let trace_source = trace();
        let trace = device.create_compute_pso(
            &ComputePsoDesc {
                threads_per_threadgroup: TRACE_THREADS,
                label: Some("spectral-trace".into()),
            },
            &compiler::compile(
                device,
                &trace_source,
                "traceMain",
                ShaderStage::Compute,
                &["spvRayQueryKHR"],
            )?,
        )?;

        let clear_source = clear();
        let clear = device.create_compute_pso(
            &ComputePsoDesc {
                threads_per_threadgroup: [CLEAR_THREADS, 1, 1],
                label: Some("spectral-film-clear".into()),
            },
            &compiler::compile(
                device,
                &clear_source,
                "clearMain",
                ShaderStage::Compute,
                &[],
            )?,
        )?;

        let display_source = display();
        let display = device.create_graphics_pso(
            &GraphicsPsoDesc {
                topology: Topology::TriangleList,
                color_targets: vec![ColorTarget::new(color_format)],
                depth_format: None,
                sample_count: SampleCount::S1,
                cull: Cull::None,
                blendstate: Some(BlendState::default()),
                label: Some("spectral-display".into()),
                ..Default::default()
            },
            &compiler::compile(
                device,
                &display_source,
                "displayVs",
                ShaderStage::Vertex,
                &[],
            )?,
            &compiler::compile(
                device,
                &display_source,
                "displayFs",
                ShaderStage::Pixel,
                &[],
            )?,
        )?;

        Ok(Self {
            trace,
            clear,
            display,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::process::Command;

    use super::*;

    /// slangc is the only thing that can tell us the assembled sources are valid.
    #[test]
    fn assembled_sources_compile() {
        if !compiler::slangc_available() {
            return;
        }
        check(&trace(), "traceMain", "compute", &["spvRayQueryKHR"]);
        check(&clear(), "clearMain", "compute", &[]);
        check(&display(), "displayVs", "vertex", &[]);
        check(&display(), "displayFs", "fragment", &[]);
    }

    fn check(source: &str, entry: &str, stage: &str, capabilities: &[&str]) {
        let stem = format!("spectra_shader_test_{}_{}", std::process::id(), entry);
        let source_path = std::env::temp_dir().join(format!("{stem}.slang"));
        let output_path = std::env::temp_dir().join(format!("{stem}.spv"));
        std::fs::write(&source_path, source).expect("write shader source");

        let mut command = Command::new("slangc");
        command.args([
            source_path
                .to_str()
                .expect("temporary source path is UTF-8"),
            "-target",
            "spirv",
            "-entry",
            entry,
            "-stage",
            stage,
            "-O2",
            "-fvk-use-entrypoint-name",
            "-fvk-bind-globals",
            "0",
            "1",
        ]);
        for capability in capabilities {
            command.args(["-capability", capability]);
        }
        command.args([
            "-o",
            output_path
                .to_str()
                .expect("temporary output path is UTF-8"),
        ]);

        let result = command.output().expect("run slangc");
        let _ = std::fs::remove_file(&source_path);
        let _ = std::fs::remove_file(&output_path);
        assert!(
            result.status.success(),
            "slangc failed compiling {entry}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
}
