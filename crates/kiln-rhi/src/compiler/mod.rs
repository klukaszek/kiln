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

use crate::{Device, RhiError, RhiResult, ShaderModule, ShaderModuleDesc, ShaderStage};

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
        self.try_compile(device, src, entry, stage, capabilities)
            .unwrap_or_else(|error| panic!("SlangCompiler failed: {error}"))
    }

    /// Fallible form of [`compile`](Self::compile), suitable for library code that should
    /// propagate shader and toolchain failures to its caller.
    pub fn try_compile(
        &self,
        device: &Device,
        src: &str,
        entry: &str,
        stage: ShaderStage,
        capabilities: &[&str],
    ) -> RhiResult<ShaderModule> {
        let (target, ext) = backend_target(device)?;
        let code = self.get_or_compile(src, entry, stage, target, ext, capabilities)?;
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
    ) -> RhiResult<Vec<u8>> {
        let key = cache_key(
            self.version_hash,
            src,
            entry,
            stage,
            target,
            capabilities,
            SLANG_OPTIMIZATION_LEVEL,
        );
        let path = self.cache_dir.join(format!("{key:016x}.{ext}"));
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
}

impl Default for SlangCompiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Compile `src`'s `entry` point with a fresh [`SlangCompiler`] and no extra Slang capabilities.
///
/// The one-call convenience over `SlangCompiler::new().compile(.., &[])` for the common case (the
/// version probe and cache are process-wide, so constructing the compiler per call is free).
/// Panics if `slangc` is missing or compilation fails.
pub fn compile(device: &Device, src: &str, entry: &str, stage: ShaderStage) -> ShaderModule {
    compile_with_caps(device, src, entry, stage, &[])
}

/// Like [`compile`], but with explicit Slang capabilities (e.g. `"spvRayQueryKHR"`).
pub fn compile_with_caps(
    device: &Device,
    src: &str,
    entry: &str,
    stage: ShaderStage,
    capabilities: &[&str],
) -> ShaderModule {
    SlangCompiler::new().compile(device, src, entry, stage, capabilities)
}

/// Like [`compile`], but returns `None` (instead of panicking) when `slangc` is not on `PATH`.
///
/// For tests that should skip rather than fail when the compiler is absent. A compile *error*
/// still panics (that's a shader bug, not an environment issue).
pub fn compile_or_skip(
    device: &Device,
    src: &str,
    entry: &str,
    stage: ShaderStage,
) -> Option<ShaderModule> {
    compile_caps_or_skip(device, src, entry, stage, &[])
}

/// Like [`compile_or_skip`], but with explicit Slang capabilities.
pub fn compile_caps_or_skip(
    device: &Device,
    src: &str,
    entry: &str,
    stage: ShaderStage,
    capabilities: &[&str],
) -> Option<ShaderModule> {
    SlangCompiler::new().compile_or_skip(device, src, entry, stage, capabilities)
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

fn backend_target(device: &Device) -> RhiResult<(&'static str, &'static str)> {
    match device.backend_name() {
        "Vulkan" => Ok(("spirv", "spv")),
        "Metal" => Ok(("metallib", "metallib")),
        other => Err(RhiError::Unsupported(format!(
            "SlangCompiler: unsupported backend `{other}`"
        ))),
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

#[cfg(test)]
mod tests {
    use super::valid_artifact;

    #[test]
    fn rejects_truncated_or_corrupt_spirv_cache_entries() {
        assert!(!valid_artifact(&[], "spirv"));
        assert!(!valid_artifact(&[0, 0, 0, 0], "spirv"));
        assert!(!valid_artifact(&[0x03, 0x02, 0x23, 0x07, 0], "spirv"));
        assert!(valid_artifact(&[0x03, 0x02, 0x23, 0x07], "spirv"));
    }
}
