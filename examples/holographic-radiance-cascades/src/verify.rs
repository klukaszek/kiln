//! Measure the solver against a brute-force angular integral of the same scene.
//!
//! Radiance cascades are an approximation: probes are spaced apart, directions are quantised, and
//! intervals are stitched together with interpolation. This is what says how far that lands from a
//! dense per-pixel ray integral, which is the reference the method should be judged against.
//!
//! The reference sphere-marches the signed-distance scene directly, so it shares neither the BVH
//! nor the broadphase with the solver; agreement between them checks the tessellation too.
//!
//! Ported from the original's `verify.py`, down to the statistics, so the numbers in its README
//! are directly comparable.

use glam::{UVec2, Vec4};
use kiln_rhi::{Device, DeviceDesc, Format, MemoryType, RhiResult, StageFlags};

use crate::cascades::{Frame, HrcRenderer, REFERENCE_DIRECTIONS, Settings};
use crate::scene::SCENES;

/// Directions per dispatch. A thousand directions per pixel in one command buffer is long enough
/// for the GPU to be reset out from under it.
const DIRECTIONS_PER_DISPATCH: u32 = 64;

/// Frames run before measuring. One would do with the bounce off, but the first also pays for the
/// allocation and the field clear.
const SETTLE_FRAMES: u32 = 2;

pub struct Options {
    pub res: u32,
    /// Probe grid to solve on, when it should differ from the measured resolution. This is what
    /// prices the probe budget: the field is solved coarsely, reconstructed at `res`, and compared
    /// against a reference computed at `res`.
    pub probe_res: u32,
    pub dirs: u32,
    pub scene: usize,
    pub save: bool,
}

pub fn run(options: &Options) -> RhiResult<()> {
    let device = Device::new(&DeviceDesc {
        validation: false,
        label: Some("hrc-verify"),
        ..Default::default()
    })?;
    // The format only reaches the tonemap, and nothing here is tonemapped on the GPU.
    let mut renderer = HrcRenderer::new(&device, Format::R8G8B8A8Unorm)?;

    let res = UVec2::splat(options.res);
    let probes = UVec2::splat(if options.probe_res > 0 {
        options.probe_res
    } else {
        options.res
    });
    let pixels = u64::from(res.x) * u64::from(res.y);
    let bytes = pixels * 16;
    // No bounce: the reference models direct transport only, so the solver has to as well.
    let settings = Settings {
        bounce: 0.0,
        sky: 0.0,
        ..Settings::default()
    };
    let (name, build) = SCENES[options.scene];
    let prims = build(0.0);

    let solved = device.allocate_array::<Vec4>(pixels as usize, MemoryType::GpuOnly)?;
    let reference = device.allocate_array::<Vec4>(pixels as usize, MemoryType::GpuOnly)?;
    let readback = device.allocate_array::<Vec4>(pixels as usize, MemoryType::Readback)?;

    for _ in 0..SETTLE_FRAMES {
        let mut cmd = device.create_command_buffer()?;
        let frame = Frame {
            device: &device,
            slot: 0,
            res: probes,
            out_res: res,
        };
        renderer.record(&frame, &mut cmd, &prims, settings)?;
        cmd.end()?;
        device.queue().submit(cmd)?;
        device.queue().wait_idle();
    }

    let mut cmd = device.create_command_buffer()?;
    renderer.reset_arena(0);
    renderer.record_light_field_linear(&mut cmd, 0, settings, solved.gpu());
    // The reference accumulates, so it starts from a cleared target.
    renderer.record_clear(&mut cmd, 0, reference.gpu(), bytes / 4);
    cmd.end()?;
    device.queue().submit(cmd)?;
    device.queue().wait_idle();

    let mut begin = 0;
    while begin < options.dirs {
        let end = (begin + DIRECTIONS_PER_DISPATCH).min(options.dirs);
        let mut cmd = device.create_command_buffer()?;
        renderer.reset_arena(0);
        renderer.record_reference(
            &mut cmd,
            0,
            settings,
            reference.gpu(),
            options.dirs,
            begin..end,
        );
        cmd.end()?;
        device.queue().submit(cmd)?;
        device.queue().wait_idle();
        begin = end;
    }

    let solved_rgb = read_rgb(&device, &solved, &readback, bytes, 1.0)?;
    let reference_rgb = read_rgb(&device, &reference, &readback, bytes, options.dirs as f32)?;

    let stats = Stats::compare(&solved_rgb, &reference_rgb);
    let (a, b) = renderer.levels_per_axis();
    println!("scene            {name}  {}x{}", res.x, res.y);
    println!(
        "probe grid       {}x{}  ({}k probes)",
        probes.x,
        probes.y,
        probes.x * probes.y / 1000
    );
    println!("cascades         {a}/{b} levels per axis");
    println!("reference        {} directions/pixel", options.dirs);
    println!("relative L1      {:.4}", stats.rel_l1);
    println!("relative L2      {:.4}", stats.rel_l2);
    println!("log RMSE         {:.4}", stats.log_rmse);
    println!(
        "energy ratio     {:.4}  (1.0 = no gained/lost light)",
        stats.energy
    );
    println!(
        "max abs error    {:.4}   mean reference {:.4}",
        stats.max_error, stats.mean_reference
    );

    if options.save {
        write_png("rc.png", res, &solved_rgb, 1.0)?;
        write_png("reference.png", res, &reference_rgb, 1.0)?;
        let error: Vec<f32> = solved_rgb
            .iter()
            .zip(&reference_rgb)
            .map(|(a, b)| (a - b).abs() * 8.0)
            .collect();
        write_png("error.png", res, &error, 1.0)?;
        println!("wrote rc.png, reference.png, error.png");
    }

    renderer.destroy(&device);
    for allocation in [solved, reference, readback] {
        device.destroy(allocation);
    }
    Ok(())
}

/// Copy a device buffer back and flatten it to interleaved RGB, scaled by `1 / divisor`.
fn read_rgb(
    device: &Device,
    source: &kiln_rhi::Allocation<Vec4>,
    readback: &kiln_rhi::Allocation<Vec4>,
    bytes: u64,
    divisor: f32,
) -> RhiResult<Vec<f32>> {
    let mut cmd = device.create_command_buffer()?;
    cmd.barrier(StageFlags::COMPUTE, StageFlags::TRANSFER);
    cmd.memcpy(readback.gpu(), source.gpu(), bytes);
    cmd.end()?;
    device.queue().submit(cmd)?;
    device.queue().wait_idle();

    let texels: &[Vec4] = readback.as_slice()?;
    let inverse = 1.0 / divisor;
    Ok(texels
        .iter()
        .flat_map(|t| [t.x * inverse, t.y * inverse, t.z * inverse])
        .collect())
}

struct Stats {
    rel_l1: f32,
    rel_l2: f32,
    log_rmse: f32,
    energy: f32,
    max_error: f32,
    mean_reference: f32,
}

impl Stats {
    fn compare(solved: &[f32], reference: &[f32]) -> Self {
        let n = solved.len() as f32;
        let mean_reference = reference.iter().sum::<f32>() / n;
        let denominator = mean_reference + 1e-6;
        let mut sum_abs = 0.0;
        let mut sum_square = 0.0;
        let mut sum_log_square = 0.0;
        let mut max_error: f32 = 0.0;
        for (a, b) in solved.iter().zip(reference) {
            let error = (a - b).abs();
            sum_abs += error;
            sum_square += error * error;
            sum_log_square += (a.ln_1p() - b.ln_1p()).powi(2);
            max_error = max_error.max(error);
        }
        Self {
            rel_l1: (sum_abs / n) / denominator,
            rel_l2: (sum_square / n).sqrt() / denominator,
            log_rmse: (sum_log_square / n).sqrt(),
            energy: solved.iter().sum::<f32>() / reference.iter().sum::<f32>().max(1e-9),
            max_error,
            mean_reference,
        }
    }
}

/// Exposure, Reinhard, then gamma — the same curve the resolve applies, so a saved image is
/// comparable with the original's.
fn write_png(path: &str, res: UVec2, rgb: &[f32], exposure: f32) -> RhiResult<()> {
    let bytes: Vec<u8> = rgb
        .iter()
        .map(|c| {
            let exposed = (c * exposure).max(0.0);
            let reinhard = exposed / (1.0 + exposed);
            (reinhard.powf(1.0 / 2.2) * 255.0).round().clamp(0.0, 255.0) as u8
        })
        .collect();
    image::RgbImage::from_raw(res.x, res.y, bytes)
        .expect("image dimensions match the buffer")
        .save(path)
        .map_err(|error| {
            kiln_rhi::RhiError::AllocationFailed(kiln_rhi::ErrorDetail::with_source(
                format!("write {path}"),
                error,
            ))
        })?;
    Ok(())
}

/// Default direction count, re-exported so the CLI and the renderer agree.
pub const DEFAULT_DIRECTIONS: u32 = REFERENCE_DIRECTIONS;
