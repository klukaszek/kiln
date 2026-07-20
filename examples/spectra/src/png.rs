use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

pub fn save_rgba_png(name: &str, width: u32, height: u32, rgba: &[u8]) -> anyhow::Result<PathBuf> {
    let expected_len = usize::try_from(width)?
        .checked_mul(usize::try_from(height)?)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| anyhow::anyhow!("image dimensions overflow: {width}x{height}"))?;
    anyhow::ensure!(
        rgba.len() == expected_len,
        "pixel buffer is {} bytes, expected {expected_len}",
        rgba.len()
    );

    let dir = workspace_target_dir().join("test-images");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{name}.png"));
    let file = BufWriter::new(File::create(&path)?);
    let mut encoder = png::Encoder::new(file, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(rgba)?;
    Ok(path)
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
