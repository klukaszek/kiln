//! Headless path-tracing application mode.

use std::time::Instant;

use glam::UVec2;
use kiln_rhi::{Device, DeviceDesc, Format};

use spectra::path_tracer::{PathTracer, Settings};
use spectra::render::{RenderFrame, Renderer};

use super::Result;
use super::config::Config;
use super::controls::debug_camera_roundtrip;

pub fn run(config: &Config, resolution: UVec2) -> Result<()> {
    let device = Device::new(&DeviceDesc {
        validation: false,
        label: Some("spectral-headless".into()),
        ..Default::default()
    })?;
    let scene = spectra::usd::load(&config.scene_path()?)?;
    if std::env::var_os("SPECTRAL_DEBUG_CAMERA").is_some() {
        debug_camera_roundtrip(&scene);
    }

    let light = config.light_spectrum()?;
    let settings = Settings {
        target_spp: config.spp,
        passes_per_frame: config.passes_per_frame,
        render_scale: 1,
        pixel_stride: config.headless_pixel_stride,
    };
    let mut renderer = PathTracer::new(&device, Format::B8G8R8A8Srgb, &scene, &light, settings)?;

    let result = (|| {
        eprintln!(
            "spectral headless: {}x{}, target spp={}, passes/frame={}, light spectrum {}",
            resolution.x,
            resolution.y,
            renderer.target_spp(),
            renderer.passes_per_frame(),
            light.name,
        );

        let frame = RenderFrame {
            device: &device,
            extent: resolution,
            slot: 0,
        };
        let start = Instant::now();
        render_to_completion(&mut renderer, &frame, &scene.camera)?;

        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        let samples = f64::from(renderer.sample_count().max(1));
        let passes = f64::from(renderer.pass_count().max(1));
        let paths = f64::from(resolution.x) * f64::from(resolution.y) * samples;
        eprintln!(
            "spectral trace: {:.1} ms for {} spp over {} spatial passes = {:.2} ms/spp, {:.2} ms/pass ({:.2} Mpath/s)",
            elapsed_ms,
            renderer.sample_count(),
            renderer.pass_count(),
            elapsed_ms / samples,
            elapsed_ms / passes,
            paths / (elapsed_ms * 1e3),
        );

        let extent = renderer.extent();
        let rgba = renderer.tonemapped_rgba8(&device)?;
        let name = format!(
            "{}_{}x{}_{}spp",
            config.scene_name(),
            extent.x,
            extent.y,
            renderer.sample_count()
        );
        let path = super::output::save_rgba_png(&name, extent.x, extent.y, &rgba)?;
        eprintln!("spectral headless wrote {}", path.display());

        super::output::emit_spectral(
            &renderer,
            &device,
            extent,
            config.spectral_probe,
            config.spectral_dump.as_deref(),
        )
    })();

    renderer.destroy(&device);
    result
}

/// Drive any finite progressive renderer without coupling the loop to its implementation.
fn render_to_completion(
    renderer: &mut PathTracer,
    frame: &RenderFrame<'_>,
    camera: &spectra::scene::Camera,
) -> Result<()> {
    while !renderer.is_complete() {
        // Each pass creates autoreleased encoders; the frame scope keeps a long headless trace
        // from retaining every encoder until process exit.
        kiln_rhi::frame_scope(|| -> Result<()> {
            let mut cmd = frame.device.create_command_buffer()?;
            renderer.encode(frame, &mut cmd, camera)?;
            cmd.end();
            let queue = frame.device.queue();
            queue.submit(cmd)?;
            queue.wait_idle();
            Ok(())
        })?;
    }
    Ok(())
}
