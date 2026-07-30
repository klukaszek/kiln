//! Shared helpers for the headless RHI integration tests.
#![allow(dead_code)]

//! Tests use the real backend without a window or swapchain and skip when no device is available.

use kiln_rhi::compiler::SlangCompiler;
use kiln_rhi::{Device, DeviceDesc};

/// Serializes GPU access across test threads in the same binary.
pub type GpuGuard = std::sync::MutexGuard<'static, ()>;

/// Route `log` records (including the Vulkan validation callback) to stderr, once per process.
/// Only used when `KILN_VALIDATION` is set; run with `-- --nocapture` to see the output.
fn install_stderr_logger() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    struct StderrLogger;
    impl log::Log for StderrLogger {
        fn enabled(&self, _: &log::Metadata) -> bool {
            true
        }
        fn log(&self, record: &log::Record) {
            eprintln!("[{}] {}", record.level(), record.args());
        }
        fn flush(&self) {}
    }
    static LOGGER: StderrLogger = StderrLogger;
    INIT.call_once(|| {
        let _ = log::set_logger(&LOGGER).map(|()| log::set_max_level(log::LevelFilter::Trace));
    });
}

pub fn device_or_skip() -> Option<(Device, GpuGuard)> {
    use std::sync::Mutex;
    static GPU_LOCK: Mutex<()> = Mutex::new(());
    let guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let validation = std::env::var_os("KILN_VALIDATION").is_some();
    if validation {
        install_stderr_logger();
    }

    let desc = DeviceDesc {
        validation,
        label: Some("rhi-headless-tests".into()),
        ..Default::default()
    };
    match Device::new(&desc) {
        Ok(device) => Some((device, guard)),
        Err(e) => {
            eprintln!("skipping: no headless GPU device available ({e})");
            None
        }
    }
}

use std::time::{Duration, Instant};

/// Format a duration with an adaptive unit (ns / µs / ms / s).
pub fn fmt_dur(d: Duration) -> String {
    let ns = d.as_nanos();
    if ns < 1_000 {
        format!("{ns} ns")
    } else if ns < 1_000_000 {
        format!("{:.2} µs", ns as f64 / 1e3)
    } else if ns < 1_000_000_000 {
        format!("{:.3} ms", ns as f64 / 1e6)
    } else {
        format!("{:.3} s", ns as f64 / 1e9)
    }
}

/// Time a single operation, print `⏱ <label>: <elapsed>`, and return its result.
pub fn timed<T>(label: &str, f: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let out = f();
    let elapsed = start.elapsed();
    eprintln!("    ⏱  {label}: {}", fmt_dur(elapsed));
    out
}

/// True if the `slangc` compiler is available. Shader-path tests skip when it is not.
pub fn slangc_available() -> bool {
    SlangCompiler::available()
}

/// Write `rgba` (exactly `width * height * 4` bytes, row-major, `R8G8B8A8`) to
/// `target/test-images/<name>.png` and return the path. Prints the path to
/// stderr so it's easy to find (run `cargo test -- --nocapture` to see it).
pub fn save_rgba_png(name: &str, width: u32, height: u32, rgba: &[u8]) -> std::path::PathBuf {
    assert_eq!(
        rgba.len(),
        (width * height * 4) as usize,
        "save_rgba_png: pixel buffer is {} bytes, expected {}",
        rgba.len(),
        width * height * 4,
    );

    let dir = workspace_target_dir().join("test-images");
    std::fs::create_dir_all(&dir).expect("create target/test-images");
    let path = dir.join(format!("{name}.png"));

    std::fs::write(&path, encode_png_rgba8(width, height, rgba)).expect("write png");
    eprintln!("    🖼  wrote {}", path.display());
    path
}

/// The shared workspace `target/` directory (honouring `CARGO_TARGET_DIR`), so test images land in
/// one place regardless of which member crate's manifest dir we were compiled under. This crate's
/// manifest is at `<workspace>/crates/kiln-rhi`, so the workspace root is two levels up.
fn workspace_target_dir() -> std::path::PathBuf {
    if let Some(dir) = std::env::var_os("CARGO_TARGET_DIR") {
        return std::path::PathBuf::from(dir);
    }
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root above crates/<crate>")
        .join("target")
}

/// Encode 8-bit RGBA pixels as a PNG. Single "None"-filtered image, stored
/// (uncompressed) deflate inside a minimal zlib stream.
fn encode_png_rgba8(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    // Prefix each scanline with a filter-type byte (0 = None).
    let stride = (width * 4) as usize;
    let mut raw = Vec::with_capacity(height as usize * (1 + stride));
    for y in 0..height as usize {
        raw.push(0);
        raw.extend_from_slice(&rgba[y * stride..(y + 1) * stride]);
    }

    let mut out = Vec::new();
    out.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);

    // IHDR: width, height, bit depth 8, colour type 6 (RGBA), no interlace.
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
    write_png_chunk(&mut out, b"IHDR", &ihdr);

    write_png_chunk(&mut out, b"IDAT", &zlib_stored(&raw));
    write_png_chunk(&mut out, b"IEND", &[]);
    out
}

/// Append a `length | type | data | CRC32` PNG chunk.
fn write_png_chunk(out: &mut Vec<u8>, tag: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(tag);
    out.extend_from_slice(data);
    let mut crc = 0xffff_ffffu32;
    for &b in tag.iter().chain(data) {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    out.extend_from_slice(&(crc ^ 0xffff_ffff).to_be_bytes());
}

/// Wrap `data` in a zlib stream using only stored (uncompressed) deflate blocks.
fn zlib_stored(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01]; // CMF/FLG: deflate, 32K window (0x7801 % 31 == 0)
    let mut chunks = data.chunks(0xffff).peekable();
    loop {
        let chunk = chunks.next().unwrap_or(&[]);
        let last = chunks.peek().is_none();
        out.push(last as u8); // BFINAL, BTYPE=00 (stored)
        let len = chunk.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(chunk);
        if last {
            break;
        }
    }
    // Adler-32 of the uncompressed data.
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    out.extend_from_slice(&((b << 16) | a).to_be_bytes());
    out
}
