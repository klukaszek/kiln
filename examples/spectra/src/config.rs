use std::path::{Path, PathBuf};

use clap::Parser;
use glam::UVec2;

use crate::{pathtracer, scene::spectral};

const ASSETS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/assets");

#[derive(Parser, Clone, Debug)]
pub struct Config {
    /// Target samples per pixel for the progressive render
    #[arg(long, default_value_t = pathtracer::DEFAULT_TARGET_SPP)]
    pub spp: u32,
    /// Spatial tracing passes recorded per frame
    #[arg(
        long,
        visible_aliases = ["samples-per-frame", "spf"],
        default_value_t = pathtracer::DEFAULT_PASSES_PER_FRAME
    )]
    pub passes_per_frame: u32,
    /// Render offscreen at WxH and write a PNG under target/test-images
    #[arg(long, value_name = "WxH", value_parser = parse_resolution)]
    pub headless: Option<UVec2>,
    /// Light emission spectrum: A, D50, D65, E, FL2, FL7, FL11, <T>K
    /// (blackbody), <λ>nm (monochromatic), a path to an LSPDD CSV, or the
    /// stem of a CSV dropped into examples/spectra/assets/lspdd
    #[arg(long, default_value = "A")]
    light_spectrum: String,
    /// Scene to render: the stem of a .usda under examples/spectra/assets
    /// (e.g. cornell-box, cornell-box-copper) or a path to any USD file
    #[arg(long, default_value = "cornell-box")]
    scene: String,
    /// Windowed mode: trace at display resolution divided by this (the blit
    /// upscales). 2 quarters the per-frame path count on retina displays.
    #[arg(long, default_value_t = 1)]
    pub render_scale: u32,
    /// Windowed NxN spatial interleave; the film remains full resolution
    #[arg(long, default_value_t = 2)]
    pub pixel_stride: u32,
    /// Headless NxN spatial interleave; 1 is the dense reference
    #[arg(long, default_value_t = 1)]
    pub headless_pixel_stride: u32,
    /// Headless: write the full per-pixel spectral film to this path as a
    /// float32 `.npy` of shape (height, width, SPECTRAL_BINS) — band-integrated
    /// radiance per pixel.
    #[arg(long, value_name = "PATH")]
    pub spectral_dump: Option<PathBuf>,
    /// Headless: print one pixel's spectrum (X,Y in film pixels) to stderr.
    #[arg(long, value_name = "X,Y", value_parser = parse_pixel)]
    pub spectral_probe: Option<(u32, u32)>,
    /// Harness options (e.g. --validation), parsed here and forwarded to `run_with`.
    #[command(flatten)]
    pub harness: kiln_app::HarnessOpts,
}

impl Config {
    pub fn light_spectrum(&self) -> anyhow::Result<spectral::Spd> {
        if let Some(spd) = spectral::named(&self.light_spectrum) {
            return Ok(spd);
        }

        let literal = PathBuf::from(&self.light_spectrum);
        if literal.is_file() {
            return spectral::from_lspdd_csv(&literal);
        }

        let dropped = Path::new(ASSETS_DIR)
            .join("lspdd")
            .join(&self.light_spectrum)
            .with_extension("csv");
        if dropped.is_file() {
            return spectral::from_lspdd_csv(&dropped);
        }

        anyhow::bail!(
            "no spectrum named {:?}: not a built-in, not a CSV path, and {} does not exist",
            self.light_spectrum,
            dropped.display()
        )
    }

    pub fn scene_path(&self) -> anyhow::Result<PathBuf> {
        let literal = PathBuf::from(&self.scene);
        if literal.is_file() {
            return Ok(literal);
        }

        let bundled = Path::new(ASSETS_DIR)
            .join(&self.scene)
            .with_extension("usda");
        if bundled.is_file() {
            return Ok(bundled);
        }

        anyhow::bail!(
            "no scene named {:?}: not a file, and not one of the bundled scenes ({})",
            self.scene,
            bundled_scenes().join(", ")
        )
    }

    pub fn scene_name(&self) -> String {
        Path::new(&self.scene)
            .file_stem()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.scene.clone())
    }
}

fn bundled_scenes() -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(ASSETS_DIR) else {
        return Vec::new();
    };
    let mut scenes = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "usda")
        })
        .filter_map(|path| {
            path.file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
        })
        .collect::<Vec<_>>();
    scenes.sort();
    scenes
}

fn parse_pixel(value: &str) -> Result<(u32, u32), String> {
    let (x, y) = value
        .split_once(',')
        .ok_or_else(|| format!("expected X,Y, got {value:?}"))?;
    Ok((
        x.trim()
            .parse()
            .map_err(|_| format!("bad X in {value:?}"))?,
        y.trim()
            .parse()
            .map_err(|_| format!("bad Y in {value:?}"))?,
    ))
}

fn parse_resolution(value: &str) -> Result<UVec2, String> {
    let (width, height) = value
        .split_once(['x', 'X'])
        .ok_or_else(|| format!("expected a resolution in WxH form, got {value:?}"))?;
    let positive = |component: &str| {
        component
            .trim()
            .parse::<u32>()
            .ok()
            .filter(|&value| value > 0)
            .ok_or_else(|| format!("expected a positive integer, got {component:?}"))
    };
    Ok(UVec2::new(positive(width)?, positive(height)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_resolution_with_either_separator() {
        assert_eq!(
            parse_resolution("1920x1080").unwrap(),
            UVec2::new(1920, 1080)
        );
        assert_eq!(parse_resolution(" 8 X 6 ").unwrap(), UVec2::new(8, 6));
    }

    #[test]
    fn rejects_invalid_resolutions() {
        for value in ["1920", "0x1080", "1920x0", "ax1080"] {
            assert!(parse_resolution(value).is_err(), "{value}");
        }
    }

    #[test]
    fn parses_spectral_probe_coordinates() {
        assert_eq!(parse_pixel("12, 34").unwrap(), (12, 34));
        assert!(parse_pixel("12x34").is_err());
        assert!(parse_pixel("-1,0").is_err());
    }
}
