//! Slang shader compiler with a file-based binary cache.
//!
//! Compiles Slang source to the active backend's format (SPIR-V or metallib) and loads it as a
//! [`ShaderModule`]. Artifacts are cached on disk, keyed on source, entry point, stage, target,
//! capabilities and slangc version.
//!
//! Shells out to `slangc` on `PATH`, which suits tests, examples and iteration. A shipped
//! application should compile offline and pass the bytes to [`Device::create_shader_module`].
//! Calls report [`ShaderCompilation`](crate::RhiError::ShaderCompilation) when `slangc` is missing.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::ErrorDetail;
use crate::{Backend, Device, RhiError, RhiResult, ShaderModule, ShaderModuleDesc, ShaderStage};

static SEQ: AtomicU64 = AtomicU64::new(0);

/// Slang defaults to `-O1`. `-O3` costs code size and compile time for little runtime gain.
const SLANG_OPTIMIZATION_LEVEL: &str = "2";

/// Lowers `DescriptorHandle<T>` onto `SPV_EXT_descriptor_heap` instead of a runtime array. The
/// output has no set or binding decorations, so pipelines can use a null layout.
const SPIRV_DESCRIPTOR_HEAP_CAPABILITY: &str = "spvDescriptorHeapEXT";

/// Emits `[numthreads]` as `[[required_threads_per_threadgroup]]`.
///
/// Naming any capability also stops Slang auto-promoting `descriptor_handle` into the target,
/// which leaves `RayDesc` undeclared in the Metal output (shader-slang/slang#13329).
const METAL_CAPABILITY: &str = "metallib_4_0";

/// Mesh shaders get plain `metal`: Slang's reflection omits their `[numthreads]`, so the draw
/// cannot match a required size and Metal would drop it.
const METAL_MESH_CAPABILITY: &str = "metal";

/// Declares `kiln::accel`, which turns an [`AccelHandle`](crate::AccelHandle) into a
/// `RaytracingAccelerationStructure`.
///
/// Vulkan reaches an acceleration structure by device address and Metal by resource id, and
/// each arm is silently wrong on the other target, so keep both.
const ACCEL_PRELUDE: &str = concat!(
    "namespace kiln { RaytracingAccelerationStructure accel(",
    "DescriptorHandle<RaytracingAccelerationStructure> h) { __target_switch { ",
    "case spirv: return RaytracingAccelerationStructure(reinterpret<uint64_t>(h)); ",
    "default: return h; } } }\n",
);

/// The scratch files one compile writes, removed when it finishes.
struct TempShaderFiles {
    source: PathBuf,
    output: PathBuf,
    /// Compute and mesh only: slangc's reflection JSON, read for `[numthreads]`.
    reflection: Option<PathBuf>,
}

impl TempShaderFiles {
    fn new(output_ext: &str, stage: ShaderStage) -> RhiResult<Self> {
        let dir = scratch_dir()?;
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let stem = dir.join(format!("kiln_{}_{seq}", std::process::id()));
        Ok(Self {
            source: stem.with_extension("slang"),
            output: stem.with_extension(output_ext),
            reflection: matches!(stage, ShaderStage::Compute | ShaderStage::Mesh)
                .then(|| stem.with_extension("json")),
        })
    }
}

impl Drop for TempShaderFiles {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.source);
        let _ = std::fs::remove_file(&self.output);
        if let Some(reflection) = &self.reflection {
            let _ = std::fs::remove_file(reflection);
        }
    }
}

/// Compile `src`'s `entry` point for the device's backend. `capabilities` takes extra Slang
/// capabilities (e.g. `"spvRayQueryKHR"`), `&[]` for none.
pub fn compile(
    device: &Device,
    src: &str,
    entry: &str,
    stage: ShaderStage,
    capabilities: &[&str],
) -> RhiResult<ShaderModule> {
    let (target, ext) = backend_target(device);
    // Added here rather than at the command line so they are part of the cache key.
    let mut effective: Vec<&str> = capabilities.to_vec();
    effective.push(match (device.backend(), stage) {
        (Backend::Vulkan, _) => SPIRV_DESCRIPTOR_HEAP_CAPABILITY,
        (Backend::Metal, ShaderStage::Mesh) => METAL_MESH_CAPABILITY,
        (Backend::Metal, _) => METAL_CAPABILITY,
    });
    let src = format!("{ACCEL_PRELUDE}{src}");
    let compiled = get_or_compile(&src, entry, stage, target, ext, &effective)?;
    make_module(device, &compiled, entry, stage)
}

struct Compiled {
    code: Vec<u8>,
    /// `[numthreads]` for a compute or mesh entry point, when slangc reports it.
    threads_per_threadgroup: Option<[u32; 3]>,
}

/// A per-user, owner-only directory under the system temp dir. Everything in it is fed to a
/// compiler or the GPU, so it must not be writable by other users.
fn kiln_temp_root() -> RhiResult<&'static PathBuf> {
    static ROOT: OnceLock<Option<PathBuf>> = OnceLock::new();
    ROOT.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("kiln-rhi-{}", current_user_id()));
        std::fs::create_dir_all(&dir).ok()?;
        restrict_to_owner(&dir).ok()?;
        Some(dir)
    })
    .as_ref()
    .ok_or_else(|| {
        RhiError::ShaderCompilation(
            "could not create a private directory for the shader compiler under the system \
             temp directory"
                .into(),
        )
    })
}

fn current_user_id() -> u32 {
    #[cfg(unix)]
    {
        // SAFETY: `getuid` takes no arguments, cannot fail, and touches no memory.
        unsafe { libc_getuid() }
    }
    #[cfg(not(unix))]
    {
        // Windows temp directories are already per-user.
        0
    }
}

#[cfg(unix)]
unsafe extern "C" {
    #[link_name = "getuid"]
    fn libc_getuid() -> u32;
}

fn restrict_to_owner(dir: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
    }
    Ok(())
}

fn cache_dir() -> RhiResult<PathBuf> {
    let dir = kiln_temp_root()?.join("shader-cache");
    std::fs::create_dir_all(&dir).map_err(|error| {
        RhiError::ShaderCompilation(ErrorDetail::with_source("create the shader cache", error))
    })?;
    Ok(dir)
}

fn scratch_dir() -> RhiResult<PathBuf> {
    let dir = kiln_temp_root()?.join("scratch");
    std::fs::create_dir_all(&dir).map_err(|error| {
        RhiError::ShaderCompilation(ErrorDetail::with_source(
            "create the shader compiler scratch directory",
            error,
        ))
    })?;
    Ok(dir)
}

fn get_or_compile(
    src: &str,
    entry: &str,
    stage: ShaderStage,
    target: &str,
    ext: &str,
    capabilities: &[&str],
) -> RhiResult<Compiled> {
    let key = cache_key(
        slangc_version_hash(),
        src,
        entry,
        stage,
        target,
        capabilities,
    );
    let path = cache_dir()?.join(format!("{key:032x}.{ext}"));
    // The threadgroup size is stored beside the artifact so a cache hit needs no reflection.
    let sidecar = path.with_extension(format!("{ext}.threadgroup"));
    if let Ok(cached) = std::fs::read(&path)
        && valid_artifact(&cached, target)
    {
        let threads = std::fs::read_to_string(&sidecar)
            .ok()
            .and_then(|text| parse_threadgroup_sidecar(&text));
        if threads.is_some() || stage != ShaderStage::Compute {
            return Ok(Compiled {
                code: cached,
                threads_per_threadgroup: threads,
            });
        }
        // A compute entry without its sidecar is recompiled.
    }
    let compiled = invoke_slangc(src, entry, stage, target, ext, capabilities)?;
    if !valid_artifact(&compiled.code, target) {
        return Err(RhiError::ShaderCompilation(
            format!("slangc produced an invalid {target} artifact for `{entry}`").into(),
        ));
    }
    write_cache_atomically(&path, &compiled.code);
    if let Some([x, y, z]) = compiled.threads_per_threadgroup {
        write_cache_atomically(&sidecar, format!("{x} {y} {z}").as_bytes());
    }
    Ok(compiled)
}

fn parse_threadgroup_sidecar(text: &str) -> Option<[u32; 3]> {
    let mut dims = text.split_whitespace().map(str::parse::<u32>);
    let size = [dims.next()?.ok()?, dims.next()?.ok()?, dims.next()?.ok()?];
    (dims.next().is_none() && size.iter().all(|d| *d > 0)).then_some(size)
}

/// FNV-1a, 128-bit. `DefaultHasher` is not stable across Rust releases, and this value names a
/// file that outlives the process.
struct CacheHasher {
    state: u128,
}

impl CacheHasher {
    const OFFSET_BASIS: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
    const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;

    fn new() -> Self {
        Self {
            state: Self::OFFSET_BASIS,
        }
    }

    fn write(&mut self, bytes: &[u8]) -> &mut Self {
        for &byte in bytes {
            self.state ^= u128::from(byte);
            self.state = self.state.wrapping_mul(Self::PRIME);
        }
        self
    }

    /// Length-prefixed, so `["ab", "c"]` and `["a", "bc"]` hash differently.
    fn field(&mut self, bytes: &[u8]) -> &mut Self {
        self.write(&(bytes.len() as u64).to_le_bytes());
        self.write(bytes)
    }

    fn finish(&self) -> u128 {
        self.state
    }
}

/// Hash of `slangc -v`, probed once. Zero when slangc is unavailable.
fn slangc_version_hash() -> u128 {
    static VERSION: OnceLock<Option<u128>> = OnceLock::new();
    VERSION
        .get_or_init(|| {
            let out = Command::new("slangc").arg("-v").output().ok()?;
            if !out.status.success() {
                return None;
            }
            let mut h = CacheHasher::new();
            h.field(&out.stdout).field(&out.stderr);
            Some(h.finish())
        })
        .unwrap_or(0)
}

fn cache_key(
    version_hash: u128,
    src: &str,
    entry: &str,
    stage: ShaderStage,
    target: &str,
    capabilities: &[&str],
) -> u128 {
    let mut h = CacheHasher::new();
    h.field(&version_hash.to_le_bytes())
        .field(src.as_bytes())
        .field(entry.as_bytes())
        .field(stage_str(stage).as_bytes())
        .field(target.as_bytes())
        .field(SLANG_OPTIMIZATION_LEVEL.as_bytes());

    let mut caps = capabilities.to_vec();
    caps.sort_unstable();
    h.field(&(caps.len() as u64).to_le_bytes());
    for cap in caps {
        h.field(cap.as_bytes());
    }
    h.finish()
}

/// `"threadGroupSize": [x, y, z]` from slangc's reflection JSON. Scanned rather than parsed to
/// avoid a JSON dependency for one key. Each invocation passes a single `-entry`, so the first
/// match is the only one.
fn thread_group_size(reflection: &str) -> Option<[u32; 3]> {
    let rest = reflection.split_once("\"threadGroupSize\"")?.1;
    let inside = rest.split_once('[')?.1.split_once(']')?.0;
    let mut dims = inside.split(',').map(|v| v.trim().parse::<u32>());
    let size = [dims.next()?.ok()?, dims.next()?.ok()?, dims.next()?.ok()?];
    (size.iter().all(|d| *d > 0) && dims.next().is_none()).then_some(size)
}

fn valid_artifact(code: &[u8], target: &str) -> bool {
    if target == "spirv" {
        code.len() >= 4 && code.len().is_multiple_of(4) && code[..4] == [0x03, 0x02, 0x23, 0x07]
    } else {
        !code.is_empty()
    }
}

fn write_cache_atomically(path: &Path, code: &[u8]) {
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let temp = path.with_extension(format!("{}.{}.tmp", std::process::id(), seq));
    if std::fs::write(&temp, code).is_err() {
        return;
    }
    // `rename` cannot replace an existing file on Windows. Any existing entry was already found
    // invalid, so remove it first; an interrupted swap just recompiles next run.
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
) -> RhiResult<Compiled> {
    let files = TempShaderFiles::new(ext, stage)?;
    std::fs::write(&files.source, src).map_err(|error| {
        RhiError::ShaderCompilation(ErrorDetail::with_source(
            format!("write `{}`", files.source.display()),
            error,
        ))
    })?;

    let mut cmd = Command::new("slangc");
    cmd.arg(&files.source)
        .args([
            "-target",
            target,
            "-entry",
            entry,
            "-stage",
            stage_str(stage),
        ])
        .arg(format!("-O{SLANG_OPTIMIZATION_LEVEL}"));
    if let Some(reflection) = &files.reflection {
        cmd.arg("-reflection-json").arg(reflection);
    }
    if target == "spirv" {
        // Keep the entry-point name in `OpEntryPoint` so it matches the RHI.
        cmd.arg("-fvk-use-entrypoint-name");
    }
    // Slang silently accepts `spv*` capabilities on other targets, so callers can pass them
    // unconditionally and they are dropped here.
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
            .into(),
            _ => ErrorDetail::with_source("run slangc", error),
        })
    })?;

    if !output.status.success() {
        return Err(RhiError::ShaderCompilation(
            format!(
                "slangc failed compiling `{entry}` for {target}:\n{}",
                String::from_utf8_lossy(&output.stderr)
            )
            .into(),
        ));
    }

    let code = std::fs::read(&files.output).map_err(|error| {
        RhiError::ShaderCompilation(ErrorDetail::with_source(
            format!("read `{}`", files.output.display()),
            error,
        ))
    })?;

    // Missing reflection is not fatal here: Vulkan reads `[numthreads]` from the SPIR-V, and
    // Metal reports the missing size when the pipeline is created.
    let threads_per_threadgroup = files.reflection.as_ref().and_then(|path| {
        let size = thread_group_size(&std::fs::read_to_string(path).ok()?);
        if size.is_none() && stage == ShaderStage::Compute {
            log::warn!(
                "slangc reflection for `{entry}` carries no threadGroupSize; \
                 ComputePsoDesc::threads_per_threadgroup must be set explicitly"
            );
        }
        size
    });

    Ok(Compiled {
        code,
        threads_per_threadgroup,
    })
}

fn make_module(
    device: &Device,
    compiled: &Compiled,
    entry: &str,
    stage: ShaderStage,
) -> RhiResult<ShaderModule> {
    device.create_shader_module(&ShaderModuleDesc {
        code: &compiled.code,
        entry_point: entry,
        stage,
        threads_per_threadgroup: compiled.threads_per_threadgroup,
        label: Some(entry),
    })
}

/// slangc `-target` and artifact extension for the device's backend.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cache_key_format_is_stable() {
        let mut h = CacheHasher::new();
        assert_eq!(h.finish(), CacheHasher::OFFSET_BASIS);
        // Cross-checked against an independent FNV-1a-128.
        h.write(b"kiln");
        assert_eq!(h.finish(), 0x6946_4f9f_0f75_7277_b806_e969_f75b_2213);

        let mut a = CacheHasher::new();
        a.field(b"ab").field(b"c");
        let mut b = CacheHasher::new();
        b.field(b"a").field(b"bc");
        assert_ne!(a.finish(), b.finish());
    }

    #[test]
    fn a_threadgroup_sidecar_round_trips() {
        assert_eq!(parse_threadgroup_sidecar("64 2 1"), Some([64, 2, 1]));
        assert_eq!(parse_threadgroup_sidecar("64 2"), None);
        assert_eq!(parse_threadgroup_sidecar("64 2 0"), None);
        assert_eq!(parse_threadgroup_sidecar("64 2 1 8"), None);
        assert_eq!(parse_threadgroup_sidecar(""), None);
    }
}
