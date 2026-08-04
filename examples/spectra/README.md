# Spectra

Spectra is the biggest thing I've built on Kiln. It's a spectral path tracer that loads USD scenes,
and it mostly exists to put the RHI under real load: geometry, an acceleration structure, bindless
materials, frames in flight. Most of the RHI changes I've made started with something breaking
here.

The tracer carries radiance in wavelength bands instead of RGB, so changing the light actually
changes the image instead of being a white balance multiply at the end. There's also a raster
backend that reads the same scene, for when I just want to see the geometry.

Run everything from the workspace root. Build and backend requirements are in the
[root README](../../README.md).

## Run it

With `slangc` on `PATH`:

```bash
cargo run -p spectra
```

That opens the bundled Cornell box and traces toward 1024 spp. If the spectral backend won't
initialize it prints why and falls back to raster.

Camera is `WASD`, `Q`/`E` for down and up, `Shift` to move faster, left-drag to look, `R` to go
back to the authored camera.

### Scenes

`--scene` takes the stem of a `.usda` in [`assets/`](assets/), so `cornell-box` or
`cornell-box-copper`, or a path to any USD file. The third bundled scene is nested, so pass the
path:

```bash
cargo run -p spectra -- --scene assets/salledebain/salledebain.usda
```

### Light

This is the knob I actually play with. `--light-spectrum` takes a named illuminant (`A`, `D50`,
`D65`, `E`, `FL2`, `FL7`, `FL11`), a blackbody temperature like `3200K`, a single wavelength like
`550nm`, a path to an LSPDD CSV, or the stem of a CSV you drop into
[`assets/lspdd/`](assets/lspdd/):

```bash
cargo run -p spectra -- --light-spectrum D65
```

The default is `A`, which is tungsten, so default renders look warm. Provenance for the bundled
LSPDD data is in [`assets/lspdd/README.md`](assets/lspdd/README.md).

### If it's slow

The viewer traces at display resolution, which hurts on a retina screen. `--render-scale 2` traces
at half resolution and upscales on the blit, so a quarter of the paths. `--pixel-stride N` spreads
an NxN pattern across frames and keeps the film at full resolution (defaults to 2 windowed, 1
headless). `--passes-per-frame` goes the other way if you'd rather converge fast than stay
interactive. `--help` has the rest.

## Headless rendering

Renders to the target sample count and writes a PNG under `target/test-images`:

```bash
cargo run --release -p spectra -- \
  --scene cornell-box \
  --spp 64 \
  --headless 1024x1024
```

`--headless-pixel-stride 1` is the default and gives you a dense reference render.

### Spectral output

The film keeps `SPECTRAL_BINS` band-integrated radiance estimates per pixel, spread uniformly from
360 to 830 nm. Right now that's 4, so each bin is 117.5 nm wide. That's coarse. It's really a memory
knob (`width * height * bins * 4` bytes) and I've kept it low while the transport is still moving
around. Raising it is a one-line change in `renderers/spectral/spectrum/mod.rs`.

Probe one pixel to stderr, or dump the whole film as a little-endian float32 `.npy`:

```bash
cargo run --release -p spectra -- \
  --headless 1024x1024 \
  --spectral-probe 512,512 \
  --spectral-dump target/cornell-box.npy
```

The dump is shape `(height, width, SPECTRAL_BINS)` and loads with `numpy.load`.

## How it's laid out

Organized by who owns the data, not by demo mode:

```text
src/
  base/
    scene/                  generic scene data and the Scene<S> storage contract
    renderer.rs             renderer and frame contracts
    gpu/                    shared GPU upload/resource helpers
  importers/
    usd/                    USD loading and conversion
  renderers/
    raster/                 raster GPU scene and meshlet renderer
    spectral/               progressive spectral path tracer
      spectrum/             wavelengths, colorimetry, illuminants, reflectance fitting
  app/                      CLI, windowed viewer, headless output, controls
```

The importer gives you `Scene<CpuStorage>`. Each renderer calls `prepare::<Storage>` to turn that
into its own storage type, which stays private to that renderer's module. Scene construction is
shared, and anything a backend has an opinion about (acceleration structures, GPU layout, material
lowering) stays inside the backend that owns it.

Raster isn't a second scene model. It reads the same imported scene and owns only the GPU side of
drawing it.

Inside the spectral renderer, `spectrum/` is the physics with no GPU knowledge in it. Everything
above it is the tracer: acceleration, sampling, film accumulation, and the GPU passes.

## Assets

Bundled USD scenes sit directly under [`assets/`](assets/), optional LSPDD light spectra under
[`assets/lspdd/`](assets/lspdd/). Don't commit large or generated assets. The ignore rules there are
deliberately conservative.

## Development

```bash
cargo fmt --all -- --check
cargo check -p spectra
cargo test -p spectra
```

Library tests cover scene loading, GPU layout, and the spectral math. Binary tests cover CLI
parsing. Nothing tests actual rendering except the headless path.
