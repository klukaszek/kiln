//! Slang shader compiler with a file-based binary cache.
//!
//! Provides a single canonical path for compiling Slang source to the active
//! backend's format (SPIR-V or metallib) and loading the result as a
//! [`ShaderModule`]. Compiled binaries are cached on disk keyed on source
//! content, entry point, stage, target, capabilities, and slangc version, so
//! repeated compilations of the same shader are instant.
//!
//! The cache and version probe are process-wide, so there is nothing to construct.
//!
//! # Vulkan flags applied on every compile
//!
//! - `-fvk-use-entrypoint-name`: preserves the entry-point name in `OpEntryPoint`
//!   so `ShaderModuleDesc::entry_point` matches what Vulkan expects.
//! - `-fvk-bind-globals 0 1`: redirects Slang's `$Globals` cbuffer (module-scope
//!   uniforms) from set 0 to set 1. Set 0 is the bindless heap; a stray global
//!   there silently aliases it. With this flag the collision becomes a
//!   missing-binding error.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{Backend, Device, RhiError, RhiResult, ShaderModule, ShaderModuleDesc, ShaderStage};

static SEQ: AtomicU64 = AtomicU64::new(0);

// Slang defaults to optimization level 1 when no `-O` flag is supplied. Level 2 enables the
// aggressive speed optimizations we want for runtime shaders without the code-size and compile-time
// tradeoffs of level 3.
const SLANG_OPTIMIZATION_LEVEL: &str = "2";

struct TempShaderFiles {
    source: PathBuf,
    output: PathBuf,
}

impl Drop for TempShaderFiles {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.source);
        let _ = std::fs::remove_file(&self.output);
    }
}

/// Returns `true` if `slangc` is reachable on `PATH`.
pub fn slangc_available() -> bool {
    slangc_version_hash_raw().is_some()
}

/// Compile `src`'s `entry` point for the device's backend. `capabilities` takes extra Slang
/// capabilities (e.g. `"spvRayQueryKHR"`), `&[]` for none. Cached in the temp dir.
pub fn compile(
    device: &Device,
    src: &str,
    entry: &str,
    stage: ShaderStage,
    capabilities: &[&str],
) -> RhiResult<ShaderModule> {
    let (target, ext) = backend_target(device);
    let code = get_or_compile(src, entry, stage, target, ext, capabilities)?;
    make_module(device, &code, entry, stage)
}

/// Like [`compile`], but returns `None` when `slangc` is missing, for tests that skip rather than
/// fail. A compile *error* still panics: that is a shader bug, not an environment issue.
pub fn compile_or_skip(
    device: &Device,
    src: &str,
    entry: &str,
    stage: ShaderStage,
    capabilities: &[&str],
) -> Option<ShaderModule> {
    if !slangc_available() {
        eprintln!("skipping: slangc not found on PATH");
        return None;
    }
    Some(
        compile(device, src, entry, stage, capabilities)
            .unwrap_or_else(|error| panic!("slangc failed: {error}")),
    )
}

/// Cache directory, created once per process.
fn cache_dir() -> &'static PathBuf {
    static CACHE_DIR: OnceLock<PathBuf> = OnceLock::new();
    CACHE_DIR.get_or_init(|| {
        let dir = std::env::temp_dir().join("kiln-shader-cache");
        std::fs::create_dir_all(&dir).ok();
        dir
    })
}

fn get_or_compile(
    src: &str,
    entry: &str,
    stage: ShaderStage,
    target: &str,
    ext: &str,
    capabilities: &[&str],
) -> RhiResult<Vec<u8>> {
    let key = cache_key(
        slangc_version_hash(),
        src,
        entry,
        stage,
        target,
        capabilities,
        SLANG_OPTIMIZATION_LEVEL,
    );
    let path = cache_dir().join(format!("{key:016x}.{ext}"));
    if let Ok(cached) = std::fs::read(&path)
        && valid_artifact(&cached, target)
    {
        return Ok(cached);
    }
    let code = invoke_slangc(src, entry, stage, target, ext, capabilities)?;
    if !valid_artifact(&code, target) {
        return Err(RhiError::ShaderCompilation(format!(
            "slangc produced an invalid {target} artifact for `{entry}`"
        )));
    }
    write_cache_atomically(&path, &code);
    Ok(code)
}

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
    optimization_level: &str,
) -> u64 {
    let mut h = DefaultHasher::new();
    version_hash.hash(&mut h);
    src.hash(&mut h);
    entry.hash(&mut h);
    stage_str(stage).hash(&mut h);
    target.hash(&mut h);
    optimization_level.hash(&mut h);
    let mut caps = capabilities.to_vec();
    caps.sort_unstable();
    caps.hash(&mut h);
    h.finish()
}

fn valid_artifact(code: &[u8], target: &str) -> bool {
    if target == "spirv" {
        code.len() >= 4 && code.len().is_multiple_of(4) && code[..4] == [0x03, 0x02, 0x23, 0x07]
    } else {
        !code.is_empty()
    }
}

fn write_cache_atomically(path: &std::path::Path, code: &[u8]) {
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let temp = path.with_extension(format!("{}.{}.tmp", std::process::id(), seq));
    if std::fs::write(&temp, code).is_err() {
        return;
    }

    // `rename` cannot replace an existing file on Windows. At this point an existing entry was
    // already found to be malformed, so remove only that cache entry before installing the fully
    // written replacement. If power is lost between these operations, the next run recompiles.
    let _ = std::fs::remove_file(path);
    if std::fs::rename(&temp, path).is_err() {
        let _ = std::fs::remove_file(&temp);
    }
}

fn invoke_slangc(
    src: &str,
    entry: &str,
    stage: ShaderStage,
    target: &str,
    ext: &str,
    capabilities: &[&str],
) -> RhiResult<Vec<u8>> {
    let dir = std::env::temp_dir();
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let files = TempShaderFiles {
        source: dir.join(format!("kiln_{pid}_{seq}.slang")),
        output: dir.join(format!("kiln_{pid}_{seq}.{ext}")),
    };

    std::fs::write(&files.source, src).map_err(|error| {
        RhiError::ShaderCompilation(format!("write `{}`: {error}", files.source.display()))
    })?;

    let mut cmd = Command::new("slangc");
    cmd.arg(&files.source).args([
        "-target",
        target,
        "-entry",
        entry,
        "-stage",
        stage_str(stage),
    ]);
    cmd.arg(format!("-O{SLANG_OPTIMIZATION_LEVEL}"));
    if target == "spirv" {
        // Keep the entry-point name in `OpEntryPoint` so it matches the RHI.
        cmd.arg("-fvk-use-entrypoint-name");
        // Reserve set 0 for the bindless heap.
        cmd.args(["-fvk-bind-globals", "0", "1"]);
    }
    for cap in capabilities {
        cmd.args(["-capability", cap]);
    }
    cmd.arg("-o").arg(&files.output);

    let output = cmd
        .output()
        .map_err(|error| RhiError::ShaderCompilation(format!("run slangc: {error}")))?;

    if !output.status.success() {
        return Err(RhiError::ShaderCompilation(format!(
            "slangc failed compiling `{entry}` for {target}:\n{}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    std::fs::read(&files.output).map_err(|error| {
        RhiError::ShaderCompilation(format!("read `{}`: {error}", files.output.display()))
    })
}

fn make_module(
    device: &Device,
    code: &[u8],
    entry: &str,
    stage: ShaderStage,
) -> RhiResult<ShaderModule> {
    device.create_shader_module(&ShaderModuleDesc {
        code,
        entry_point: entry,
        stage,
        label: Some(entry),
    })
}

/// slangc `-target` plus the artifact extension for the device's backend.
fn backend_target(device: &Device) -> (&'static str, &'static str) {
    match device.backend() {
        Backend::Vulkan => ("spirv", "spv"),
        Backend::Metal => ("metallib", "metallib"),
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
