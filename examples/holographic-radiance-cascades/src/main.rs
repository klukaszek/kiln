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
    let harness = config.harness.clone();
    app::configure(config.probes, config.scene as usize);
    kiln_app::run_with::<app::App>(
        "Kiln \u{00b7} Holographic Radiance Cascades",
        [0.0, 0.0, 0.0, 1.0],
        harness,
    )
}
