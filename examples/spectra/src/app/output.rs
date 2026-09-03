//! Headless render output: PNG previews and optional spectral captures.

use std::path::{Path, PathBuf};

use glam::UVec2;
use kiln_rhi::Device;

use spectra::tracer::PathTracer;

use super::{Error, Result};

/// Emit the requested spectral captures after a headless render. No-op unless a
/// probe pixel or a dump path was requested.
pub fn emit_spectral(
    tracer: &PathTracer,
    device: &Device,
    extent: UVec2,
    probe: Option<(u32, u32)>,
    dump: Option<&Path>,
) -> Result<()> {
    if probe.is_none() && dump.is_none() {
        return Ok(());
    }
    let centers = PathTracer::spectral_bin_centers();
    let bands = tracer.spectral_bands(device)?;

    if let Some((px, py)) = probe {
        if px >= extent.x || py >= extent.y {
            return Err(Error::Output(format!(
                "spectral probe ({px},{py}) is outside the {}x{} film",
                extent.x, extent.y
            )));
        }
        let pixel_index = py as usize * extent.x as usize + px as usize;
        let idx = pixel_index * centers.len();
        eprintln!("spectral probe ({px},{py}) — wavelength(nm), band radiance:");
        for (j, c) in centers.iter().enumerate() {
            eprintln!("  {:6.1}  {:.6}", c, bands[idx + j]);
        }
    }

    if let Some(dump) = dump {
        let shape = [extent.y as usize, extent.x as usize, centers.len()];
        write_npy_f32(dump, &shape, &bands)?;
        eprintln!(
            "spectral dump wrote {} — shape ({}, {}, {}), bin centers {:.1}..{:.1} nm",
            dump.display(),
            extent.y,
            extent.x,
            centers.len(),
            centers[0],
            centers[centers.len() - 1],
        );
    }
    Ok(())
}

/// Save an RGBA8 image under the workspace's `target/test-images` directory.
pub fn save_rgba_png(name: &str, width: u32, height: u32, rgba: &[u8]) -> Result<PathBuf> {
    let expected_len = width as usize * height as usize * 4;
    if rgba.len() != expected_len {
        return Err(Error::Output(format!(
            "pixel buffer is {} bytes, expected {expected_len}",
            rgba.len()
        )));
    }

    let dir = workspace_target_dir().join("test-images");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{name}.png"));
    image::save_buffer(&path, rgba, width, height, image::ExtendedColorType::Rgba8)?;
    Ok(path)
}

/// Write a C-order float32 array as a NumPy `.npy` (v1.0) — dependency-free,
/// so the spectral capture loads with a plain `numpy.load`.
fn write_npy_f32(path: &Path, shape: &[usize], data: &[f32]) -> Result<()> {
    use std::io::Write;

    if shape.iter().product::<usize>() != data.len() {
        return Err(Error::Output(format!(
            "shape {shape:?} does not match {} film elements",
            data.len()
        )));
    }

    let shape_str = shape.iter().map(|d| format!("{d}, ")).collect::<String>();
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({shape_str}), }}");
    // The 10-byte prefix + header + trailing '\n' must be a multiple of 64.
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let header_len = header.len() as u16;

    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    file.write_all(b"\x93NUMPY\x01\x00")?;
    file.write_all(&header_len.to_le_bytes())?;
    file.write_all(header.as_bytes())?;
    for &v in data {
        file.write_all(&v.to_le_bytes())?;
    }
    file.flush()?;
    Ok(())
}

/// The shared workspace `target/` directory (honouring `CARGO_TARGET_DIR`).
fn workspace_target_dir() -> PathBuf {
    std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .and_then(Path::parent)
                .expect("spectra must live under <workspace>/examples")
                .join("target")
        })
}
