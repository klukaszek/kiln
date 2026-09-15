//! Slang shader compiler with a file-based binary cache.
//!
//! One canonical path from Slang source to the active backend's format (SPIR-V or metallib),
//! loaded as a [`ShaderModule`]. Results are cached on disk, keyed on source content, entry point,
//! stage, target, capabilities and slangc version, so repeated compilations are instant. The cache
//! and version probe are process-wide, so there is nothing to construct.
//!
//! It shells out to a `slangc` on `PATH` at runtime, which suits tests, examples and iteration but
//! not a shipped application: for that, compile offline and pass the bytes to
//! [`Device::create_shader_module`] directly. Being a runtime capability, no Cargo feature can
//! decide whether it works — calls report
//! [`ShaderCompilation`](crate::RhiError::ShaderCompilation) when `slangc` is missing.
//!
//! # Vulkan flags applied on every compile
//!
//! - `-fvk-use-entrypoint-name`: preserves the entry-point name in `OpEntryPoint`
//!   so `ShaderModuleDesc::entry_point` matches what Vulkan expects.
//! - `-capability spvDescriptorHeapEXT`: lowers `DescriptorHandle<T>` onto
//!   `SPV_EXT_descriptor_heap`'s `ResourceHeapEXT`/`SamplerHeapEXT` builtins instead of an
//!   unbounded runtime array. The result carries no descriptor set or binding decorations at
//!   all, which is what lets pipelines be created with a null layout.
//!
//! Shader source itself is backend-agnostic and reaches slangc unmodified, prefixed only by the
//! `kiln::accel` prelude: every resource is a `DescriptorHandle<T>` that each backend resolves
//! through its own heap, and acceleration structures go through that one accessor.

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

/// Lowers `DescriptorHandle<T>` onto `SPV_EXT_descriptor_heap`'s heap builtins instead of an
/// unbounded runtime descriptor array.
const SPIRV_DESCRIPTOR_HEAP_CAPABILITY: &str = "spvDescriptorHeapEXT";

/// Declares `kiln::accel`, which turns an [`AccelHandle`](crate::AccelHandle)'s eight bytes into
/// the `RaytracingAccelerationStructure` that a `gpu_struct!` field's property returns.
///
/// The one resource with no shared model: Vulkan reaches it by device address, Metal only as a
/// bindless resource id. `__target_switch` keeps that the sole backend-specific line in the shader
/// pipeline, and the handle is eight bytes either way, so root layout never varies by target.
///
/// Only SPIR-V converts, because only Metal can hold the value natively -- MSL builds an
/// `acceleration_structure` from its opaque handle type and nothing else, so a `uint64_t` handle
/// would need a cast Metal rejects.
///
/// Do not collapse the arms: each is silently wrong on the other backend, compiling to an empty
/// function body on Metal and to a heap load of a never-populated slot on Vulkan.
const ACCEL_PRELUDE: &str = concat!(
    "namespace kiln { RaytracingAccelerationStructure accel(",
    "DescriptorHandle<RaytracingAccelerationStructure> h) { __target_switch { ",
    "case spirv: return RaytracingAccelerationStructure(reinterpret<uint64_t>(h)); ",
    "default: return h; } } }
",
);

/// The MSL definition of `RayDesc` that Slang omits, `-include`d into the Metal translation unit.
///
/// Slang 2026.17.1 drops this declaration on `metal`/`metallib` whenever a shader mentions
/// `DescriptorHandle<T>`, leaving the type used but undefined; 2026.14 emitted it, and SPIR-V is
/// unaffected. Shader source cannot work around it: `TraceRayInline` lowers to a `RayDesc`
/// temporary even where the shader never names the type. Fields match by name, not position.
/// Delete this and its `-Xmetal` flag once Slang emits the declaration again.
const RAYDESC_MSL_DEFINITION: &str =
    "struct RayDesc { float3 Origin; float TMin; float3 Direction; float TMax; };\n";

struct TempShaderFiles {
    source: PathBuf,
    output: PathBuf,
    /// Metal only: holds [`RAYDESC_MSL_DEFINITION`] for the downstream `-include`.
    prelude: Option<PathBuf>,
}

impl Drop for TempShaderFiles {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.source);
        let _ = std::fs::remove_file(&self.output);
        if let Some(prelude) = &self.prelude {
            let _ = std::fs::remove_file(prelude);
        }
    }
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
    // Fold the heap capability into the requested set so it reaches both slangc and the cache
    // key; leaving it out of the key would serve pre-descriptor-heap SPIR-V from an old cache.
    let mut effective: Vec<&str> = capabilities.to_vec();
    if target == "spirv" {
        effective.push(SPIRV_DESCRIPTOR_HEAP_CAPABILITY);
    }
    let src = format!("{ACCEL_PRELUDE}{src}");
    let code = get_or_compile(&src, entry, stage, target, ext, &effective)?;
    make_module(device, &code, entry, stage)
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
) -> u64 {
    let mut h = DefaultHasher::new();
    version_hash.hash(&mut h);
    src.hash(&mut h);
    entry.hash(&mut h);
    stage_str(stage).hash(&mut h);
    target.hash(&mut h);
    SLANG_OPTIMIZATION_LEVEL.hash(&mut h);
    let mut caps = capabilities.to_vec();
    caps.sort_unstable();
    caps.hash(&mut h);
    // Part of the Metal translation unit, so editing it has to invalidate cached metallibs.
    if target != "spirv" {
        RAYDESC_MSL_DEFINITION.hash(&mut h);
    }
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
        prelude: (target != "spirv").then(|| dir.join(format!("kiln_{pid}_{seq}_prelude.h"))),
    };

    std::fs::write(&files.source, src).map_err(|error| {
        RhiError::ShaderCompilation(format!("write `{}`: {error}", files.source.display()))
    })?;
    if let Some(prelude) = &files.prelude {
        std::fs::write(prelude, RAYDESC_MSL_DEFINITION).map_err(|error| {
            RhiError::ShaderCompilation(format!("write `{}`: {error}", prelude.display()))
        })?;
    }

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
    }
    if let Some(prelude) = &files.prelude {
        // Hands the Metal compiler a definition Slang leaves out; see `RAYDESC_MSL_DEFINITION`.
        cmd.arg("-Xmetal")
            .arg(format!("--include={}", prelude.display()));
    }
    // `spv*` capabilities are SPIR-V-only. Slang accepts them silently on a metallib compile,
    // so filter here rather than making every caller branch on the backend.
    for cap in capabilities {
        if target != "spirv" && cap.starts_with("spv") {
            continue;
        }
        cmd.args(["-capability", cap]);
    }
    cmd.arg("-o").arg(&files.output);

    let output = cmd.output().map_err(|error| {
        RhiError::ShaderCompilation(match error.kind() {
            std::io::ErrorKind::NotFound => concat!(
                "`slangc` was not found on PATH; install the Slang toolchain, ",
                "or compile shaders offline and pass the bytes to ",
                "`Device::create_shader_module`",
            )
            .to_string(),
            _ => format!("run slangc: {error}"),
        })
    })?;

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
