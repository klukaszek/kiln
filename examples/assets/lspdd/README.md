# LSPDD lamp spectra

Drop LSPDD lamp-measurement CSVs here and select them by stem:

```sh
cargo run --example spectral -- --light-spectrum <stem>   # loads <stem>.csv from this folder
```

The data itself is **not** vendored: LSPDD (<https://lspdd.org>, Roby & Aubé) is
licensed CC BY-NC-ND, so download the lamps you want yourself. Export a lamp's
spectrum as CSV from its page on lspdd.org; the loader accepts the standard
export format (metadata lines, then `wavelength,intensity` rows — see
`from_lspdd_csv` in `examples/spectral/scene/spectral.rs`).

Everything in this folder except this README is ignored by version control.
