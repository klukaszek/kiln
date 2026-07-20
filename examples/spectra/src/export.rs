//! Spectral capture output for the headless path: a single-pixel probe to
//! stderr and/or the full per-pixel band-radiance film as a NumPy `.npy`.

use std::path::Path;

use anyhow::Context;
use glam::UVec2;
use kiln_rhi::Device;

use crate::pathtracer::PathTracer;

/// Emit the requested spectral captures after a headless render. No-op unless a
/// probe pixel or a dump path was requested.
pub fn emit(
    tracer: &PathTracer,
    device: &Device,
    extent: UVec2,
    probe: Option<(u32, u32)>,
    dump: Option<&Path>,
) -> anyhow::Result<()> {
    if probe.is_none() && dump.is_none() {
        return Ok(());
    }
    let centers = PathTracer::spectral_bin_centers();
    let bands = tracer.spectral_bands(device)?;

    if let Some((px, py)) = probe {
        anyhow::ensure!(
            px < extent.x && py < extent.y,
            "spectral probe ({px},{py}) is outside the {}x{} film",
            extent.x,
            extent.y
        );
        let pixel_index = u64::from(py) * u64::from(extent.x) + u64::from(px);
        let idx = usize::try_from(pixel_index)?
            .checked_mul(centers.len())
            .context("spectral probe index overflow")?;
        eprintln!("spectral probe ({px},{py}) — wavelength(nm), band radiance:");
        for (j, c) in centers.iter().enumerate() {
            eprintln!("  {:6.1}  {:.6}", c, bands[idx + j]);
        }
    }

    if let Some(dump) = dump {
        let shape = [
            usize::try_from(extent.y)?,
            usize::try_from(extent.x)?,
            centers.len(),
        ];
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

/// Write a C-order float32 array as a NumPy `.npy` (v1.0) — dependency-free, so
/// the spectral capture loads with a plain `numpy.load`.
fn write_npy_f32(path: &Path, shape: &[usize], data: &[f32]) -> anyhow::Result<()> {
    use std::io::Write;

    let element_count = shape
        .iter()
        .try_fold(1usize, |count, &dimension| count.checked_mul(dimension));
    anyhow::ensure!(
        element_count == Some(data.len()),
        "shape {shape:?} describes {} elements, but the film contains {}",
        element_count.unwrap_or(usize::MAX),
        data.len()
    );

    let shape_str = shape.iter().map(|d| format!("{d}, ")).collect::<String>();
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({shape_str}), }}");
    // The 10-byte prefix + header + trailing '\n' must be a multiple of 64.
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let header_len = u16::try_from(header.len()).context("NumPy header exceeds v1.0 limit")?;

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
