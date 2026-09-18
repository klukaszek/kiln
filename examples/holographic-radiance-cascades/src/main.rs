//! Holographic Radiance Cascades (Freeman, Sannikov & Margel, arXiv:2505.02041), rendered through
//! Kiln.
//!
//! Interactive 2D global illumination against an analytic signed-distance scene, traced with
//! hardware ray query. A port of a SlangPy implementation; `README.md` records what moved and why.
//!
//! Drag to move a light, `1`-`4` to switch scene, space to pause, esc to quit.

mod app;
mod cascades;
mod scene;
mod ui;
mod verify;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(about, version)]
struct Config {
    /// Probe budget in thousands. The light field is budgeted by probe count, not by a fixed side:
    /// cost is set by how many probes exist, so anything else makes frame time swing with the
    /// window's aspect.
    #[arg(long, default_value_t = cascades::Settings::default().probes_thousands,
          value_parser = clap::value_parser!(u32).range(30..=2000))]
    probes: u32,

    /// Scene to open.
    #[arg(long, value_enum, default_value_t = SceneArg::ThreeLights)]
    scene: SceneArg,

    /// Measure the solver against a brute-force angular integral instead of opening a window.
    #[arg(long)]
    verify: bool,

    /// Verification resolution; the probe grid and the measured image are both square at this size.
    #[arg(long, default_value_t = 512)]
    res: u32,

    /// Probe grid to solve on, when it should differ from `--res`. Prices the probe budget: the
    /// field is solved this coarsely, reconstructed at `--res`, and measured there.
    #[arg(long, default_value_t = 0)]
    probe_res: u32,

    /// Reference directions per pixel. The reference converges as 1/N.
    #[arg(long, default_value_t = verify::DEFAULT_DIRECTIONS)]
    dirs: u32,

    /// Write rc.png, reference.png and error.png alongside the verification numbers.
    #[arg(long)]
    save: bool,

    #[command(flatten)]
    harness: kiln_app::HarnessOpts,
}

/// The scene list, as a CLI value. Mirrors [`scene::SCENES`] in order, which the test below pins.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum SceneArg {
    ThreeLights,
    Penumbra,
    ManyLights,
    Slit,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::parse();
    if config.verify {
        verify::run(&verify::Options {
            res: config.res,
            probe_res: config.probe_res,
            dirs: config.dirs,
            scene: config.scene as usize,
            save: config.save,
        })?;
        return Ok(());
    }
    kiln_app::run_with::<app::App>(
        "Kiln \u{00b7} Holographic Radiance Cascades",
        [0.0, 0.0, 0.0, 1.0],
        config.harness.clone(),
        app::Config {
            probes: config.probes,
            scene: config.scene as usize,
        },
    )
}

#[cfg(test)]
mod frames_in_flight {
    use glam::UVec2;
    use kiln_rhi::{
        ColorAttachment, Device, DeviceDesc, Format, LoadOp, MemoryType, RenderPassDesc,
        SampleCount, StageFlags, StoreOp, TextureDesc, TextureDimension, TextureUsage,
        TimelineSemaphore,
    };

    use crate::cascades::{self, Frame, HrcRenderer, Settings};
    use crate::scene::SCENES;

    /// A rebuild racing the previous frame's ray queries tears the structure, and a torn read
    /// misses every primitive: the frame comes back unlit. Low probe counts expose it.
    #[test]
    fn a_low_probe_count_never_drops_a_frames_lighting() {
        let device = Device::new(&DeviceDesc {
            validation: false,
            label: Some("hrc-frames-in-flight"),
            ..Default::default()
        })
        .unwrap();
        let format = Format::R8G8B8A8Unorm;
        let mut renderer = HrcRenderer::new(&device, format).unwrap();
        let settings = Settings::default();
        let (_, build) = SCENES[0];

        let out = UVec2::new(800, 600);
        // The bottom of the slider's range, where the artifact was reported.
        let res = cascades::plan_resolution(out, 30).0;
        let frames = 64;
        // The tear is rare per frame, so sample several rounds of frames rather than allocating a
        // target for every frame of one long run.
        let rounds = 5;

        let desc = TextureDesc {
            width: out.x,
            height: out.y,
            depth: 1,
            mip_levels: 1,
            array_layers: 1,
            format,
            dimension: TextureDimension::D2,
            sample_count: SampleCount::S1,
            usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::TRANSFER_SRC,
            label: Some("hrc-frames-in-flight-target"),
        };
        let sa = device.texture_size_align(&desc).unwrap();
        let targets: Vec<_> = (0..frames)
            .map(|_| {
                let mem = device
                    .allocate_aligned(sa.size, sa.align, MemoryType::GpuOnly)
                    .unwrap();
                let texture = device.create_texture(&desc, mem.gpu()).unwrap();
                (texture, mem)
            })
            .collect();
        let readback = device
            .allocate(u64::from(out.x * out.y * 4), MemoryType::Readback)
            .unwrap();
        let mut unlit = Vec::new();
        for round in 0..rounds {
            let timeline: Vec<Option<TimelineSemaphore>> = (0..frames)
                .map(|_| Some(device.create_timeline_semaphore(0).unwrap()))
                .collect();

            for index in 0..frames {
                if index >= kiln_rhi::MAX_FRAMES_IN_FLIGHT {
                    let earlier: &TimelineSemaphore = timeline
                        [index - kiln_rhi::MAX_FRAMES_IN_FLIGHT]
                        .as_ref()
                        .unwrap();
                    assert!(earlier.wait(1, u64::MAX).unwrap());
                }
                let slot = index % kiln_rhi::MAX_FRAMES_IN_FLIGHT;
                let prims = build((round * frames + index) as f32 * 0.016);
                let mut cmd = device.create_command_buffer().unwrap();
                let frame = Frame {
                    device: &device,
                    slot,
                    res,
                    out_res: out,
                };
                renderer.record(&frame, &mut cmd, &prims, settings).unwrap();
                cmd.begin_render_pass(&RenderPassDesc {
                    color_attachments: &[ColorAttachment {
                        target: targets[index].0.target(),
                        load_op: LoadOp::Clear,
                        store_op: StoreOp::Store,
                        clear_color: [0.0, 0.0, 0.0, 1.0],
                    }],
                    depth_attachment: None,
                    render_area: [0, 0, out.x, out.y],
                    label: Some("example"),
                });
                cmd.set_viewport(0.0, 0.0, out.x as f32, out.y as f32, 0.0, 1.0);
                cmd.set_scissor(0, 0, out.x, out.y);
                renderer.record_resolve(&mut cmd);
                cmd.end_render_pass();
                cmd.end().expect("end command buffer");

                let signal = [(timeline[index].as_ref().unwrap(), 1u64)];
                device
                    .queue()
                    .submit_with_desc(
                        cmd,
                        &kiln_rhi::SubmitDesc {
                            wait_semaphores: &[],
                            signal_semaphores: &signal,
                        },
                    )
                    .unwrap();
            }
            device.queue().wait_idle();

            // Emitters survive a lost structure (scene data, not transport), so count lit pixels.
            for index in 0..frames {
                let mut copy = device.create_command_buffer().unwrap();
                copy.barrier(StageFlags::RASTER_COLOR_OUT, StageFlags::TRANSFER);
                copy.copy_texture_to_buffer(&targets[index].0, readback.gpu(), None);
                copy.barrier(StageFlags::TRANSFER, StageFlags::ALL_COMMANDS);
                copy.end().expect("end command buffer");
                device.queue().submit(copy).unwrap();
                device.queue().wait_idle();
                let lit = readback
                    .as_slice::<u8>()
                    .unwrap()
                    .chunks_exact(4)
                    .filter(|p| p[0] > 4 || p[1] > 4 || p[2] > 4)
                    .count();
                if lit * 20 < (out.x * out.y) as usize {
                    unlit.push((round, index, lit));
                }
            }
        }

        renderer.destroy(&device);
        for (texture, memory) in targets {
            device.destroy(texture);
            device.destroy(memory);
        }
        device.destroy(readback);
        assert!(unlit.is_empty(), "frames came back unlit: {unlit:?}");
    }
}
