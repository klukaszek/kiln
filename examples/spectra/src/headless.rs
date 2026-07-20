use std::time::Instant;

use glam::UVec2;
use kiln_app::FrameCtx;
use kiln_rhi::{Device, DeviceDesc, Format};

use crate::config::Config;
use crate::controls::debug_camera_roundtrip;
use crate::pathtracer::PathTracer;
use crate::scene::gpu::{GpuGeometry, SpectralGpuScene};

pub fn run(config: &Config, resolution: UVec2) -> anyhow::Result<()> {
    let device = Device::new(&DeviceDesc {
        validation: false,
        label: Some("spectral-headless".into()),
        ..Default::default()
    })?;
    let scene = crate::scene::load(&config.scene_path()?)?;
    if std::env::var_os("SPECTRAL_DEBUG_CAMERA").is_some() {
        debug_camera_roundtrip(&scene);
    }

    let light = config.light_spectrum()?;
    let geometry = GpuGeometry::build(&device, &scene)?;
    anyhow::ensure!(
        geometry.accel.is_some(),
        "headless render needs ray tracing support"
    );
    let spectral_scene = SpectralGpuScene::build(&device, &scene, &light)?;
    let mut tracer = PathTracer::new(
        &device,
        Format::B8G8R8A8Srgb,
        config.spp,
        config.passes_per_frame,
        1,
        config.headless_pixel_stride,
    )?;

    let result = (|| {
        eprintln!(
            "spectral headless: {}x{}, target spp={}, passes/frame={}, light spectrum {}",
            resolution.x,
            resolution.y,
            tracer.target_spp(),
            tracer.passes_per_frame(),
            light.name,
        );

        let ctx = FrameCtx {
            device: &device,
            extent: resolution,
            slot: 0,
        };
        let start = Instant::now();
        while !tracer.is_complete() {
            let previous_passes = tracer.pass_count();
            let mut cmd = device.create_command_buffer()?;
            tracer.pre_render(&ctx, &mut cmd, &scene, &geometry, &spectral_scene);
            cmd.end();
            let queue = device.queue();
            queue.submit(cmd)?;
            queue.wait_idle();
            anyhow::ensure!(
                tracer.pass_count() > previous_passes,
                "path tracer made no progress; lights={}",
                spectral_scene.light_count
            );
        }

        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        let samples = f64::from(tracer.sample_count().max(1));
        let passes = f64::from(tracer.pass_count().max(1));
        let paths = f64::from(resolution.x) * f64::from(resolution.y) * samples;
        eprintln!(
            "spectral trace: {:.1} ms for {} spp over {} spatial passes = {:.2} ms/spp, {:.2} ms/pass ({:.2} Mpath/s)",
            elapsed_ms,
            tracer.sample_count(),
            tracer.pass_count(),
            elapsed_ms / samples,
            elapsed_ms / passes,
            paths / (elapsed_ms * 1e3),
        );

        let extent = tracer.extent();
        let rgba = tracer.tonemapped_rgba8(&device)?;
        let name = format!(
            "{}_{}x{}_{}spp",
            config.scene_name(),
            extent.x,
            extent.y,
            tracer.sample_count()
        );
        let path = crate::png::save_rgba_png(&name, extent.x, extent.y, &rgba)?;
        eprintln!("spectral headless wrote {}", path.display());

        crate::export::emit(
            &tracer,
            &device,
            extent,
            config.spectral_probe,
            config.spectral_dump.as_deref(),
        )
    })();

    tracer.destroy(&device);
    spectral_scene.destroy(&device);
    geometry.destroy(&device);
    result
}
