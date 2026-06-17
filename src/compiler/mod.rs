//! Slang shader compiler with a file-based binary cache.
//!
//! Provides a single canonical path for compiling Slang source to the active
//! backend's format (SPIR-V or metallib) and loading the result as a
//! [`ShaderModule`]. Compiled binaries are cached on disk keyed on source
//! content, entry point, stage, target, capabilities, and slangc version, so
//! repeated compilations of the same shader are instant.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{Device, ShaderModule, ShaderModuleDesc, ShaderStage};

static SEQ: AtomicU64 = AtomicU64::new(0);

/// Slang source compiler backed by a file-based cache.
///
/// Construct once (cheaply) and call [`compile`] or [`compile_or_skip`] for
/// each entry point. The cache lives at `{temp_dir}/kiln-shader-cache/` and
/// is automatically invalidated when the slangc version changes.
///
/// ## Vulkan flags applied on every compile
///
/// - `-fvk-use-entrypoint-name`: preserves the entry-point name in `OpEntryPoint`
///   so `ShaderModuleDesc::entry_point` matches what Vulkan expects.
/// - `-fvk-bind-globals 0 1`: redirects Slang's `$Globals` cbuffer (module-scope
///   uniforms) from set 0 to set 1. Set 0 is the bindless heap; a stray global
///   there silently aliases it. With this flag the collision becomes a
///   missing-binding error.
///
/// [`compile`]: SlangCompiler::compile
/// [`compile_or_skip`]: SlangCompiler::compile_or_skip
pub struct SlangCompiler {
    cache_dir: PathBuf,
    /// Hash of the slangc version string — changes on compiler update,
    /// invalidating all prior cache entries for that entry in the cache dir.
    version_hash: u64,
}

impl SlangCompiler {
    /// Create a compiler instance. The slangc version probe is cached
    /// process-wide via a `OnceLock`, so subsequent calls are free.
    pub fn new() -> Self {
        let cache_dir = std::env::temp_dir().join("kiln-shader-cache");
        std::fs::create_dir_all(&cache_dir).ok();
        Self {
            version_hash: slangc_version_hash(),
            cache_dir,
        }
    }

    /// Returns `true` if `slangc` is reachable on `PATH`.
    pub fn available() -> bool {
        slangc_version_hash_raw().is_some()
    }

    /// Compile `src` and return a [`ShaderModule`].
    ///
    /// Panics if `slangc` is missing or compilation fails.
    pub fn compile(
        &self,
        device: &Device,
        src: &str,
        entry: &str,
        stage: ShaderStage,
        capabilities: &[&str],
    ) -> ShaderModule {
        let (target, ext) = backend_target(device);
        let code = self.get_or_compile(src, entry, stage, target, ext, capabilities);
        make_module(device, &code, entry, stage)
    }

    /// Like [`compile`], but returns `None` if `slangc` is not on `PATH`.
    ///
    /// Intended for tests that should skip rather than fail when the compiler
    /// is absent. Still panics on a compile error (that's a shader bug, not an
    /// environment issue).
    pub fn compile_or_skip(
        &self,
        device: &Device,
        src: &str,
        entry: &str,
        stage: ShaderStage,
        capabilities: &[&str],
    ) -> Option<ShaderModule> {
        if !Self::available() {
            eprintln!("skipping: slangc not found on PATH");
            return None;
        }
        Some(self.compile(device, src, entry, stage, capabilities))
    }

    fn get_or_compile(
        &self,
        src: &str,
        entry: &str,
        stage: ShaderStage,
        target: &str,
        ext: &str,
        capabilities: &[&str],
    ) -> Vec<u8> {
        let key = cache_key(self.version_hash, src, entry, stage, target, capabilities);
        let path = self.cache_dir.join(format!("{key:016x}.{ext}"));
        if let Ok(cached) = std::fs::read(&path) {
            return cached;
        }
        let code = invoke_slangc(src, entry, stage, target, ext, capabilities);
        let _ = std::fs::write(&path, &code);
        code
    }
}

impl Default for SlangCompiler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Process-wide cached slangc version hash. `None` means slangc is unavailable.
static SLANGC_VERSION: OnceLock<Option<u64>> = OnceLock::new();

fn slangc_version_hash_raw() -> Option<u64> {
    *SLANGC_VERSION.get_or_init(|| {
        let out = Command::new("slangc").arg("-v").output().ok()?;
        if !out.status.success() {
            return None;
        }
        let mut h = DefaultHasher::new();
        out.stdout.hash(&mut h);
        out.stderr.hash(&mut h);
        Some(h.finish())
    })
}

fn slangc_version_hash() -> u64 {
    slangc_version_hash_raw().unwrap_or(0)
}

fn cache_key(
    version_hash: u64,
    src: &str,
    entry: &str,
    stage: ShaderStage,
    target: &str,
    capabilities: &[&str],
) -> u64 {
    let mut h = DefaultHasher::new();
    version_hash.hash(&mut h);
    src.hash(&mut h);
    entry.hash(&mut h);
    stage_str(stage).hash(&mut h);
    target.hash(&mut h);
    let mut caps = capabilities.to_vec();
    caps.sort_unstable();
    caps.hash(&mut h);
    h.finish()
}

fn invoke_slangc(
    src: &str,
    entry: &str,
    stage: ShaderStage,
    target: &str,
    ext: &str,
    capabilities: &[&str],
) -> Vec<u8> {
    let dir = std::env::temp_dir();
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let src_path = dir.join(format!("kiln_{pid}_{seq}.slang"));
    let out_path = dir.join(format!("kiln_{pid}_{seq}.{ext}"));

    std::fs::write(&src_path, src).expect("write slang source");

    let mut cmd = Command::new("slangc");
    cmd.arg(&src_path)
        .args(["-target", target, "-entry", entry, "-stage", stage_str(stage)]);
    if target == "spirv" {
        // Keep the real entry-point name so the RHI's `entry_point` matches
        // the module's `OpEntryPoint`. Without this, Vulkan pipeline creation
        // references a non-existent entry and fails with VK_ERROR_UNKNOWN.
        cmd.arg("-fvk-use-entrypoint-name");
        // Redirect stray module-scope uniforms to set 1 instead of letting
        // them silently collide with the bindless heap on set 0.
        cmd.args(["-fvk-bind-globals", "0", "1"]);
    }
    for cap in capabilities {
        cmd.args(["-capability", cap]);
    }
    cmd.arg("-o").arg(&out_path);

    let output = cmd.output().expect("failed to run slangc");
    let _ = std::fs::remove_file(&src_path);

    if !output.status.success() {
        let _ = std::fs::remove_file(&out_path);
        panic!(
            "slangc failed compiling `{entry}` for {target}:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let code = std::fs::read(&out_path).expect("read compiled shader");
    let _ = std::fs::remove_file(&out_path);
    code
}

fn make_module(device: &Device, code: &[u8], entry: &str, stage: ShaderStage) -> ShaderModule {
    device
        .create_shader_module(&ShaderModuleDesc {
            code,
            entry_point: entry,
            stage,
            label: Some(entry),
        })
        .expect("create_shader_module")
}

fn backend_target(device: &Device) -> (&'static str, &'static str) {
    match device.backend_name() {
        "Vulkan" => ("spirv", "spv"),
        "Metal" => ("metallib", "metallib"),
        other => panic!("SlangCompiler: unsupported backend `{other}`"),
    }
}

fn stage_str(stage: ShaderStage) -> &'static str {
    match stage {
        ShaderStage::Compute => "compute",
        ShaderStage::Vertex => "vertex",
        ShaderStage::Pixel => "fragment",
        ShaderStage::Mesh => "mesh",
    }
}
