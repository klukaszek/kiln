//! Shared helpers for the headless RHI integration tests.
//!
//! Each test binary pulls in this whole module but uses only part of it, so silence the
//! per-binary "unused helper" warnings here rather than at every call site.
#![allow(dead_code)]

//!
//! These tests drive the real backend (Metal 4 / Vulkan 1.3+) without a window or
//! swapchain. When no usable GPU/driver is present (e.g. CI without a GPU), tests
//! should *skip* rather than fail — use [`device_or_skip`] and bail out with
//! `let Some(device) = common::device_or_skip() else { return; };`.

use spectradio_rhi::{Device, DeviceDesc, ShaderModule, ShaderModuleDesc, ShaderStage};

/// Create a headless device for testing, or `None` if no usable backend is available.
///
/// Uses the default backend for the compiled feature set (Vulkan when the `vulkan`
/// feature is on, otherwise Metal). Validation layers are disabled so the tests don't
/// depend on the Vulkan SDK / Metal validation being installed.
/// Guard serializing GPU access across the parallel test threads in a binary. Held for the
/// duration of each test so independent devices/queues don't submit concurrently.
pub type GpuGuard = std::sync::MutexGuard<'static, ()>;

pub fn device_or_skip() -> Option<(Device, GpuGuard)> {
    use std::sync::Mutex;
    static GPU_LOCK: Mutex<()> = Mutex::new(());
    // Recover from poisoning: a panicking test holds no GPU invariant we care about.
    let guard = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let desc = DeviceDesc {
        validation: false,
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

// ---------------------------------------------------------------------------
// Timing
//
// Every test reports exact runtimes so we can see how the RHI performs. Output
// goes to stderr; run `cargo test -- --nocapture` to see it (it is hidden for
// passing tests otherwise).
// ---------------------------------------------------------------------------

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

/// Run `iters` iterations of `f` (after one warm-up), printing total and per-iter timings.
/// Use for throughput/latency of repeated RHI operations.
pub fn bench(label: &str, iters: u32, mut f: impl FnMut()) {
    assert!(iters > 0, "bench needs at least one iteration");
    f(); // warm-up (not measured)
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let total = start.elapsed();
    let per = total / iters;
    eprintln!(
        "    ⏱  {label}: {iters} iters · total {} · {}/iter",
        fmt_dur(total),
        fmt_dur(per)
    );
}

// ---------------------------------------------------------------------------
// Shared GPU data contract
//
// Define a GPU-facing struct ONCE: this generates a `#[repr(C)]` Rust struct plus a
// matching Slang declaration string (`Name::SLANG`) to prepend to the shader. Host and
// device layouts stay in lockstep, so root data is built type-safely (`Data { .. }`)
// instead of poking bytes at hand-computed offsets. Each field lists its Rust type and
// the equivalent Slang type (e.g. `u64 as "uint*"`, `u32 as "uint"`, `f32 as "float"`).
// ---------------------------------------------------------------------------

#[macro_export]
macro_rules! gpu_struct {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident {
            $( $fname:ident : $fty:ty as $slang:literal ),* $(,)?
        }
    ) => {
        $(#[$meta])*
        #[repr(C)]
        #[derive(Clone, Copy, zerocopy::IntoBytes, zerocopy::FromBytes, zerocopy::Immutable)]
        $vis struct $name {
            $( pub $fname : $fty ),*
        }
        impl $name {
            /// Slang declaration matching this struct's layout; prepend to shader source.
            pub const SLANG: &'static str = concat!(
                "struct ", stringify!($name), " {\n",
                $( "    ", $slang, " ", stringify!($fname), ";\n", )*
                "};\n"
            );
        }
    };
}


// ---------------------------------------------------------------------------
// Backend-agnostic shading via Slang
//
// Tests write ONE Slang source; `compile_shader` lowers it to whatever the active
// device consumes (SPIR-V for Vulkan, metallib for Metal) and registers the module.
// Tests never reference backend-specific shader formats.
// ---------------------------------------------------------------------------

use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static SHADER_SEQ: AtomicU64 = AtomicU64::new(0);

/// True if the `slangc` compiler is available. Shader-path tests skip when it is not.
pub fn slangc_available() -> bool {
    Command::new("slangc")
        .arg("-v")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Compile a Slang source to the active backend's shader format and register it as a
/// module. Returns `None` (skip) if `slangc` is unavailable; panics on a compile error
/// (that's a test bug, not an environment issue).
///
/// Modules are registered in creation order, so the first module compiled is index 0 —
/// the value to put in `ComputePsoDesc::compute_shader` etc.
pub fn compile_shader_or_skip(
    device: &Device,
    slang_src: &str,
    entry: &str,
    stage: ShaderStage,
) -> Option<ShaderModule> {
    compile_shader_caps_or_skip(device, slang_src, entry, stage, &[])
}

/// Like [`compile_shader_or_skip`] but enables extra Slang capabilities (e.g.
/// `spvRayQueryKHR` for inline ray tracing).
pub fn compile_shader_caps_or_skip(
    device: &Device,
    slang_src: &str,
    entry: &str,
    stage: ShaderStage,
    capabilities: &[&str],
) -> Option<ShaderModule> {
    if !slangc_available() {
        eprintln!("skipping: slangc not found on PATH");
        return None;
    }

    let (target, ext) = match device.backend_name() {
        "Vulkan" => ("spirv", "spv"),
        "Metal" => ("metallib", "metallib"),
        other => panic!("compile_shader: unsupported backend {other}"),
    };
    let slang_stage = match stage {
        ShaderStage::Compute => "compute",
        ShaderStage::Vertex => "vertex",
        ShaderStage::Pixel => "fragment",
        ShaderStage::Mesh => "mesh",
    };

    let seq = SHADER_SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir();
    let src_path = dir.join(format!("rhi_test_{}_{seq}.slang", std::process::id()));
    let out_path = dir.join(format!("rhi_test_{}_{seq}.{ext}", std::process::id()));
    std::fs::write(&src_path, slang_src).expect("write slang source");

    let output = common_timed_slangc(&src_path, &out_path, target, entry, slang_stage, capabilities);
    if !output.status.success() {
        let _ = std::fs::remove_file(&src_path);
        panic!(
            "slangc failed compiling entry `{entry}` for {target}:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let code = std::fs::read(&out_path).expect("read compiled shader");
    let _ = std::fs::remove_file(&src_path);
    let _ = std::fs::remove_file(&out_path);

    Some(
        device
            .create_shader_module(&ShaderModuleDesc {
                code: &code,
                entry_point: entry,
                stage,
                label: Some("slang"),
            })
            .expect("create_shader_module"),
    )
}

fn common_timed_slangc(
    src: &std::path::Path,
    out: &std::path::Path,
    target: &str,
    entry: &str,
    stage: &str,
    capabilities: &[&str],
) -> std::process::Output {
    let start = Instant::now();
    let mut cmd = Command::new("slangc");
    cmd.arg(src)
        .args(["-target", target, "-entry", entry, "-stage", stage]);
    for cap in capabilities {
        cmd.args(["-capability", cap]);
    }
    let output = cmd
        .arg("-o")
        .arg(out)
        .output()
        .expect("failed to run slangc");
    eprintln!(
        "    ⏱  slangc {entry} → {target}: {}",
        fmt_dur(start.elapsed())
    );
    output
}
