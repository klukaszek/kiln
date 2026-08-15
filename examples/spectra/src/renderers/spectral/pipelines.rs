use kiln_rhi::{
    BlendState, ColorTarget, ComputePso, ComputePsoDesc, Device, Format, GraphicsPso,
    GraphicsPsoDesc, SampleCount, ShaderStage, Topology,
};

use crate::base::gpu::GpuTextureBinding;
use crate::base::renderer;

use super::display;
use super::integrator;
use super::sampler;
use super::scene::{
    GpuBsdf, GpuInstance, GpuLight, GpuMaterialTexture, GpuMeshLightTriangle, GpuTriangle,
};
use super::shader_types::{CLEAR_SOURCE, CLEAR_THREADS, ClearRoot, DisplayRoot, TraceRoot};

pub(super) struct Pipelines {
    pub(super) trace: ComputePso,
    pub(super) clear: ComputePso,
    pub(super) display: GraphicsPso,
}

impl Pipelines {
    pub(super) fn new(
        device: &Device,
        color_format: Format,
        spectrum_len: u32,
    ) -> renderer::Result<Self> {
        let trace_source = format!(
            "{}{}{}{}{}{}{}{}{}{}",
            GpuBsdf::SLANG,
            GpuMaterialTexture::SLANG,
            GpuTextureBinding::SLANG,
            GpuLight::SLANG,
            GpuMeshLightTriangle::SLANG,
            GpuTriangle::SLANG,
            GpuInstance::SLANG,
            TraceRoot::SLANG,
            sampler::source(),
            integrator::source(spectrum_len)
        );
        let trace_shader = kiln_rhi::compiler::compile_with_caps(
            device,
            &trace_source,
            "traceMain",
            ShaderStage::Compute,
            &["spvRayQueryKHR"],
        );
        let trace = device.create_compute_pso(
            &ComputePsoDesc {
                threads_per_threadgroup: [integrator::THREADS_X, integrator::THREADS_Y, 1],
                label: Some("spectral-trace".into()),
            },
            &trace_shader,
        )?;

        let clear_source = format!("{}{}", ClearRoot::SLANG, CLEAR_SOURCE);
        let clear_shader =
            kiln_rhi::compiler::compile(device, &clear_source, "clearMain", ShaderStage::Compute);
        let clear = device.create_compute_pso(
            &ComputePsoDesc {
                threads_per_threadgroup: [CLEAR_THREADS, 1, 1],
                label: Some("spectral-film-clear".into()),
            },
            &clear_shader,
        )?;

        let display_source = format!("{}{}", DisplayRoot::SLANG, display::source());
        let display_vs =
            kiln_rhi::compiler::compile(device, &display_source, "displayVs", ShaderStage::Vertex);
        let display_fs =
            kiln_rhi::compiler::compile(device, &display_source, "displayFs", ShaderStage::Pixel);
        let display = device.create_graphics_pso(
            &GraphicsPsoDesc {
                topology: Topology::TriangleList,
                color_targets: vec![ColorTarget::new(color_format)],
                depth_format: None,
                sample_count: SampleCount::S1,
                cull: kiln_rhi::Cull::None,
                blendstate: Some(BlendState::default()),
                label: Some("spectral-display".into()),
                ..Default::default()
            },
            &display_vs,
            &display_fs,
        )?;

        Ok(Self {
            trace,
            clear,
            display,
        })
    }
}

#[cfg(test)]
mod shader_source_tests {
    use std::process::Command;

    use super::*;

    #[test]
    fn specialized_sources_compile() {
        if !kiln_rhi::compiler::SlangCompiler::available() {
            return;
        }

        let trace_source = format!(
            "{}{}{}{}{}{}{}{}{}{}",
            GpuBsdf::SLANG,
            GpuMaterialTexture::SLANG,
            GpuTextureBinding::SLANG,
            GpuLight::SLANG,
            GpuMeshLightTriangle::SLANG,
            GpuTriangle::SLANG,
            GpuInstance::SLANG,
            TraceRoot::SLANG,
            sampler::source(),
            integrator::source(64)
        );
        compile(&trace_source, "traceMain", "compute", &["spvRayQueryKHR"]);

        let display_source = format!("{}{}", DisplayRoot::SLANG, display::source());
        compile(&display_source, "displayVs", "vertex", &[]);
        compile(&display_source, "displayFs", "fragment", &[]);
    }

    fn compile(source: &str, entry: &str, stage: &str, capabilities: &[&str]) {
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
