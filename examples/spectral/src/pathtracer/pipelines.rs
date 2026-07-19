use kiln_rhi::{
    BlendState, ColorTarget, ComputePso, ComputePsoDesc, Device, Format, GpuAddress, GraphicsPso,
    GraphicsPsoDesc, SampleCount, ShaderStage, Topology,
};

use crate::scene::gpu::{GpuBsdf, GpuLight, GpuTriangle};

use super::display;
use super::integrator;
use super::roots::{CLEAR_SOURCE, CLEAR_THREADS, ClearRoot, DisplayRoot, TraceRoot};
use super::sampler;

pub struct Pipelines {
    pub trace: ComputePso,
    pub clear: ComputePso,
    pub display: GraphicsPso,
}

impl Pipelines {
    pub fn new(device: &Device, color_format: Format) -> anyhow::Result<Self> {
        let trace_source = format!(
            "{}{}{}{}{}{}",
            GpuBsdf::SLANG,
            GpuLight::SLANG,
            GpuTriangle::SLANG,
            TraceRoot::SLANG,
            sampler::source(),
            integrator::source()
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
                root_constant_size: std::mem::size_of::<GpuAddress>() as u32,
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
                root_constant_size: std::mem::size_of::<GpuAddress>() as u32,
                threads_per_threadgroup: [CLEAR_THREADS, 1, 1],
                label: Some("spectral-film-clear".into()),
            },
            &clear_shader,
        )?;

        let display_source = format!("{}{}", DisplayRoot::SLANG, display::SOURCE);
        let display_vs =
            kiln_rhi::compiler::compile(device, &display_source, "displayVs", ShaderStage::Vertex);
        let display_fs =
            kiln_rhi::compiler::compile(device, &display_source, "displayFs", ShaderStage::Pixel);
        let display = device.create_graphics_pso(
            &GraphicsPsoDesc {
                topology: Topology::TriangleList,
                color_targets: vec![ColorTarget::new(color_format)],
                depth_format: Some(Format::D32Float),
                sample_count: SampleCount::S1,
                root_constant_size: 16,
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
