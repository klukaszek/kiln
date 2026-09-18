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
//! Metal compute detours through MSL and `xcrun metal` to put `[numthreads]` back; see
//! `METAL_THREADGROUP_FIXUP`. That needs the Xcode command line tools, and without them the
//! compile falls back to slangc's direct metallib output.
//!
//! Vulkan compiles add `-fvk-use-entrypoint-name`, so `ShaderModuleDesc::entry_point` matches
//! `OpEntryPoint`, and `-capability spvDescriptorHeapEXT` (see
//! `SPIRV_DESCRIPTOR_HEAP_CAPABILITY`).
//!
//! Shader source is backend-agnostic and reaches slangc unmodified, prefixed only by
//! `ACCEL_PRELUDE`.

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::ErrorDetail;
use crate::{Backend, Device, RhiError, RhiResult, ShaderModule, ShaderModuleDesc, ShaderStage};

static SEQ: AtomicU64 = AtomicU64::new(0);

// Slang defaults to optimization level 1 when no `-O` flag is supplied. Level 2 enables the
// aggressive speed optimizations we want for runtime shaders without the code-size and compile-time
// tradeoffs of level 3.
const SLANG_OPTIMIZATION_LEVEL: &str = "2";

/// Lowers `DescriptorHandle<T>` onto `SPV_EXT_descriptor_heap`'s heap builtins rather than an
/// unbounded runtime array. The result carries no set or binding decorations, which is what lets
/// pipelines be created with a null layout.
const SPIRV_DESCRIPTOR_HEAP_CAPABILITY: &str = "spvDescriptorHeapEXT";

/// Declares `kiln::accel`, turning an [`AccelHandle`](crate::AccelHandle)'s eight bytes into the
/// `RaytracingAccelerationStructure` a `gpu_struct!` field's property returns.
///
/// The one resource with no shared model: Vulkan reaches it by device address, Metal as a resource
/// id. Only SPIR-V converts -- MSL builds an `acceleration_structure` from its own opaque handle
/// and rejects a `uint64_t` cast.
///
/// Do not collapse the arms. Each is silently wrong on the other target: an empty function body on
/// Metal, a heap load of a never-populated slot on Vulkan.
const ACCEL_PRELUDE: &str = concat!(
    "namespace kiln { RaytracingAccelerationStructure accel(",
    "DescriptorHandle<RaytracingAccelerationStructure> h) { __target_switch { ",
    "case spirv: return RaytracingAccelerationStructure(reinterpret<uint64_t>(h)); ",
    "default: return h; } } }
",
);

/// Slang emits MSL compute entry points bare, with no `[[max_total_threads_per_threadgroup(N)]]`,
/// so the compute path routes through MSL to inject it. In the cache key, so upgrading past this
/// reuses nothing stale.
///
/// Without it Metal allocates registers not knowing the requested size and caps the pipeline
/// wherever it lands -- 32 threads for a ray-query shader -- and a dispatch wider than the cap is
/// *silently dropped*; only the backend's `maxTotalThreadsPerThreadgroup` check makes that an
/// error rather than a black frame. The cap also moves with unrelated edits nearby.
///
/// `MTL4ComputePipelineDescriptor::requiredThreadsPerThreadgroup` does not substitute: the
/// reported ceiling is invariant to it. Delete once Slang emits the attribute itself.
const METAL_THREADGROUP_FIXUP: &str = "metal-numthreads-attribute-v1";

/// The MSL definition of `RayDesc` that Slang omits, `-include`d into the Metal translation unit.
///
/// Slang 2026.17.1 drops it on `metal`/`metallib` whenever a shader mentions
/// `DescriptorHandle<T>`, leaving the type used but undefined; 2026.14 emitted it and SPIR-V is
/// unaffected. Source cannot work around it -- `TraceRayInline` lowers to a `RayDesc` temporary
/// even where the shader never names the type. Fields match by name. Delete this and its
/// `-Xmetal` flag once Slang emits the declaration again.
const RAYDESC_MSL_DEFINITION: &str =
    "struct RayDesc { float3 Origin; float TMin; float3 Direction; float TMax; };\n";

/// The scratch files one compile works through, removed when it finishes. Both paths — direct,
/// and the Metal compute detour through MSL — differ only in which optional ones they ask for.
struct TempShaderFiles {
    source: PathBuf,
    output: PathBuf,
    /// Metal only: holds [`RAYDESC_MSL_DEFINITION`] for the downstream `-include`.
    prelude: Option<PathBuf>,
    /// Compute only: slangc's reflection JSON, read for `[numthreads]`.
    reflection: Option<PathBuf>,
    /// Metal compute detour only: the generated MSL, and the AIR the Metal compiler emits.
    msl: Option<PathBuf>,
    air: Option<PathBuf>,
}

impl TempShaderFiles {
    /// Reserve a fresh stem under the compiler's own scratch directory and name the required
    /// files after it. The optional ones are added by the builder methods.
    fn new(output_ext: &str) -> RhiResult<Self> {
        let dir = scratch_dir()?;
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let stem = format!("kiln_{}_{seq}", std::process::id());
        Ok(Self {
            source: dir.join(format!("{stem}.slang")),
            output: dir.join(format!("{stem}.{output_ext}")),
            prelude: None,
            reflection: None,
            msl: None,
            air: None,
        })
    }

    fn with_prelude(mut self) -> Self {
        self.prelude = Some(self.source.with_extension("prelude.h"));
        self
    }

    fn with_reflection(mut self) -> Self {
        self.reflection = Some(self.source.with_extension("json"));
        self
    }

    fn with_msl_detour(mut self) -> Self {
        self.msl = Some(self.source.with_extension("metal"));
        self.air = Some(self.source.with_extension("air"));
        self
    }

    fn msl(&self) -> &std::path::Path {
        self.msl.as_deref().expect("the MSL detour was requested")
    }

    fn air(&self) -> &std::path::Path {
        self.air.as_deref().expect("the MSL detour was requested")
    }

    fn prelude(&self) -> &std::path::Path {
        self.prelude.as_deref().expect("a prelude was requested")
    }

    fn reflection(&self) -> &std::path::Path {
        self.reflection
            .as_deref()
            .expect("reflection was requested")
    }
}

impl Drop for TempShaderFiles {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.source);
        let _ = std::fs::remove_file(&self.output);
        for path in [&self.prelude, &self.reflection, &self.msl, &self.air]
            .into_iter()
            .flatten()
        {
            let _ = std::fs::remove_file(path);
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
    let compiled = get_or_compile(&src, entry, stage, target, ext, &effective)?;
    make_module(device, &compiled, entry, stage)
}

/// A compiled artifact plus whatever reflection the RHI needs alongside it.
struct Compiled {
    code: Vec<u8>,
    /// `[numthreads]` for a compute entry point, from slangc's reflection. `None` for every other
    /// stage, and for a cache entry written before the sidecar existed.
    threads_per_threadgroup: Option<[u32; 3]>,
}

/// The compiler's own directory under the system temp dir, owner-only and per-user: everything
/// below it is fed to a compiler or the GPU, and a predictable path in a world-writable `/tmp`
/// lets another user plant a cache entry or a symlink.
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

/// Something stable and per-user to name the directory after.
fn current_user_id() -> u32 {
    #[cfg(unix)]
    {
        // SAFETY: `getuid` takes no arguments, cannot fail, and touches no memory.
        unsafe { libc_getuid() }
    }
    #[cfg(not(unix))]
    {
        // Windows gives each user their own temp directory already.
        0
    }
}

#[cfg(unix)]
unsafe extern "C" {
    #[link_name = "getuid"]
    fn libc_getuid() -> u32;
}

/// Make `dir` inaccessible to other users. A no-op where the platform already does this.
fn restrict_to_owner(dir: &std::path::Path) -> std::io::Result<()> {
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

/// Where compiled artifacts are cached between runs.
fn cache_dir() -> RhiResult<PathBuf> {
    let dir = kiln_temp_root()?.join("shader-cache");
    std::fs::create_dir_all(&dir).map_err(|error| {
        RhiError::ShaderCompilation(ErrorDetail::with_source("create the shader cache", error))
    })?;
    Ok(dir)
}

/// Where a single compile invocation puts its scratch files.
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
    // The threadgroup size rides in a sidecar rather than in the artifact, so a cache hit does
    // not have to re-run slangc just to recover it.
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
        // A compute entry with no sidecar predates it; fall through and recompile once.
    }
    let compiled = if uses_metal_compute_fixup(target, stage) {
        compile_metal_compute(src, entry, capabilities)?
    } else {
        invoke_slangc(src, entry, stage, target, ext, capabilities)?
    };
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

/// Read back what `get_or_compile` wrote to the threadgroup sidecar.
fn parse_threadgroup_sidecar(text: &str) -> Option<[u32; 3]> {
    let mut dims = text.split_whitespace().map(str::parse::<u32>);
    let size = [dims.next()?.ok()?, dims.next()?.ok()?, dims.next()?.ok()?];
    (dims.next().is_none() && size.iter().all(|d| *d > 0)).then_some(size)
}

/// FNV-1a, 128-bit. Hand-rolled because `DefaultHasher` is not stable across Rust releases and
/// this value names a file that outlives the process; 128 bits because a collision would serve
/// the wrong compiled shader. Not cryptographic, which is fine for inputs we choose.
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

    /// Length-prefixed, so `["ab", "c"]` and `["a", "bc"]` do not hash alike.
    fn field(&mut self, bytes: &[u8]) -> &mut Self {
        self.write(&(bytes.len() as u64).to_le_bytes());
        self.write(bytes)
    }

    fn finish(&self) -> u128 {
        self.state
    }
}

/// Process-wide cached slangc version hash. `None` means slangc is unavailable.
static SLANGC_VERSION: OnceLock<Option<u128>> = OnceLock::new();

fn slangc_version_hash_raw() -> Option<u128> {
    *SLANGC_VERSION.get_or_init(|| {
        let out = Command::new("slangc").arg("-v").output().ok()?;
        if !out.status.success() {
            return None;
        }
        let mut h = CacheHasher::new();
        h.field(&out.stdout).field(&out.stderr);
        Some(h.finish())
    })
}

fn slangc_version_hash() -> u128 {
    slangc_version_hash_raw().unwrap_or(0)
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

    // Part of the Metal translation unit, so editing it has to invalidate cached metallibs.
    if target != "spirv" {
        h.field(RAYDESC_MSL_DEFINITION.as_bytes());
    }
    if uses_metal_compute_fixup(target, stage) {
        h.field(METAL_THREADGROUP_FIXUP.as_bytes());
    }
    h.finish()
}

/// Whether this compile takes the MSL detour that restores `[numthreads]`.
///
/// Compute only: the attribute has no meaning for the other stages, and routing them through a
/// second compiler for nothing would only add a way to fail. Requires the Metal toolchain, which
/// `slangc` alone does not; without it this falls back to the direct path and the backend's
/// threadgroup check remains the safety net.
fn uses_metal_compute_fixup(target: &str, stage: ShaderStage) -> bool {
    target != "spirv" && stage == ShaderStage::Compute && metal_toolchain().is_some()
}

/// `xcrun` paths for the Metal compiler and archiver, probed once.
fn metal_toolchain() -> Option<&'static (PathBuf, PathBuf)> {
    static TOOLCHAIN: OnceLock<Option<(PathBuf, PathBuf)>> = OnceLock::new();
    TOOLCHAIN
        .get_or_init(|| Some((xcrun_find("metal")?, xcrun_find("metallib")?)))
        .as_ref()
}

fn xcrun_find(tool: &str) -> Option<PathBuf> {
    let out = Command::new("xcrun")
        .args(["-sdk", "macosx", "-f", tool])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let path = PathBuf::from(String::from_utf8(out.stdout).ok()?.trim());
    path.exists().then_some(path)
}

/// Compile a compute entry point to a metallib by way of MSL, injecting the threadgroup size Slang
/// leaves out. See [`METAL_THREADGROUP_FIXUP`].
fn compile_metal_compute(src: &str, entry: &str, capabilities: &[&str]) -> RhiResult<Compiled> {
    let (metal, metallib) = metal_toolchain().ok_or_else(|| {
        RhiError::ShaderCompilation("the Metal toolchain went missing mid-compile".into())
    })?;
    let files = TempShaderFiles::new("metallib")?
        .with_prelude()
        .with_reflection()
        .with_msl_detour();

    write_file(&files.source, src.as_bytes())?;
    write_file(files.prelude(), RAYDESC_MSL_DEFINITION.as_bytes())?;

    // Slang to MSL, asking for the reflection that carries the threadgroup size. Taking it from
    // reflection rather than from the pipeline description keeps `[numthreads]` the single source
    // of truth, exactly as it is on the SPIR-V path.
    let mut cmd = Command::new("slangc");
    cmd.arg(&files.source)
        .args(["-target", "metal", "-entry", entry, "-stage", "compute"])
        .arg(format!("-O{SLANG_OPTIMIZATION_LEVEL}"))
        .arg("-reflection-json")
        .arg(files.reflection());
    for cap in capabilities {
        if !cap.starts_with("spv") {
            cmd.args(["-capability", cap]);
        }
    }
    cmd.arg("-o").arg(files.msl());
    run(&mut cmd, "slangc", entry)?;

    let msl = read_scratch(files.msl(), |p| std::fs::read_to_string(p))?;
    let reflection = read_scratch(files.reflection(), |p| std::fs::read_to_string(p))?;
    let threads = thread_group_size(&reflection).ok_or_else(|| {
        RhiError::ShaderCompilation(
            format!("slangc reflection for `{entry}` carries no threadGroupSize").into(),
        )
    })?;
    write_file(
        files.msl(),
        inject_threadgroup_attribute(&msl, entry, threads)?.as_bytes(),
    )?;

    let mut cmd = Command::new(metal);
    cmd.arg("-c")
        .arg(files.msl())
        .arg("-include")
        .arg(files.prelude())
        .arg("-o")
        .arg(files.air());
    run(&mut cmd, "metal", entry)?;

    let mut cmd = Command::new(metallib);
    cmd.arg(files.air()).arg("-o").arg(&files.output);
    run(&mut cmd, "metallib", entry)?;

    Ok(Compiled {
        code: read_scratch(&files.output, |p| std::fs::read(p))?,
        threads_per_threadgroup: Some(threads),
    })
}

fn write_file(path: &std::path::Path, bytes: &[u8]) -> RhiResult<()> {
    std::fs::write(path, bytes).map_err(|error| {
        RhiError::ShaderCompilation(ErrorDetail::with_source(
            format!("write `{}`", path.display()),
            error,
        ))
    })
}

/// `std::fs::read`/`read_to_string` with the path in the error.
fn read_scratch<T>(
    path: &std::path::Path,
    read: impl FnOnce(&std::path::Path) -> std::io::Result<T>,
) -> RhiResult<T> {
    read(path).map_err(|error| {
        RhiError::ShaderCompilation(ErrorDetail::with_source(
            format!("read `{}`", path.display()),
            error,
        ))
    })
}

fn run(cmd: &mut Command, tool: &str, entry: &str) -> RhiResult<()> {
    let output = cmd
        .output()
        .map_err(|error| RhiError::ShaderCompilation(format!("run {tool}: {error}").into()))?;
    if !output.status.success() {
        return Err(RhiError::ShaderCompilation(
            format!(
                "{tool} failed compiling `{entry}`:\n{}",
                String::from_utf8_lossy(&output.stderr)
            )
            .into(),
        ));
    }
    Ok(())
}

/// `"threadGroupSize": [x, y, z]` out of slangc's reflection JSON. Scanned rather than parsed: the
/// RHI has no JSON dependency, and one well-known key from one tool does not justify one.
///
/// Takes the first occurrence, which is the only one: every `slangc` invocation here passes a
/// single `-entry`, so the reflection describes exactly one entry point. Compiling several at
/// once would need a real parser to tell their sizes apart.
fn thread_group_size(reflection: &str) -> Option<[u32; 3]> {
    let rest = reflection.split_once("\"threadGroupSize\"")?.1;
    let inside = rest.split_once('[')?.1.split_once(']')?.0;
    let mut dims = inside.split(',').map(|v| v.trim().parse::<u32>());
    let size = [dims.next()?.ok()?, dims.next()?.ok()?, dims.next()?.ok()?];
    (size.iter().all(|d| *d > 0) && dims.next().is_none()).then_some(size)
}

/// Put `[[max_total_threads_per_threadgroup(N)]]` on the entry point's `[[kernel]]` declaration.
fn inject_threadgroup_attribute(msl: &str, entry: &str, threads: [u32; 3]) -> RhiResult<String> {
    let declaration = format!("[[kernel]] void {entry}(");
    let at = msl.find(&declaration).ok_or_else(|| {
        RhiError::ShaderCompilation(
            format!("no `{declaration}` in the MSL slangc generated").into(),
        )
    })?;
    let total = threads[0]
        .checked_mul(threads[1])
        .and_then(|xy| xy.checked_mul(threads[2]))
        .ok_or_else(|| {
            RhiError::ShaderCompilation(
                format!("threadgroup size {threads:?} for `{entry}` overflows a u32").into(),
            )
        })?;
    Ok(format!(
        "{}[[max_total_threads_per_threadgroup({total})]]\n{}",
        &msl[..at],
        &msl[at..]
    ))
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
) -> RhiResult<Compiled> {
    let mut files = TempShaderFiles::new(ext)?;
    if target != "spirv" {
        files = files.with_prelude();
    }
    // Compute only: `[numthreads]` is the one piece of reflection the RHI consumes, and it
    // reaches `ShaderModule` so a `ComputePsoDesc` never has to restate it.
    if stage == ShaderStage::Compute {
        files = files.with_reflection();
    }

    write_file(&files.source, src.as_bytes())?;
    if let Some(prelude) = &files.prelude {
        write_file(prelude, RAYDESC_MSL_DEFINITION.as_bytes())?;
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
    if let Some(reflection) = &files.reflection {
        cmd.arg("-reflection-json").arg(reflection);
    }
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
            .into(),
            // The io::Error is kept as the source so a caller can tell a permissions failure
            // from a missing binary without reading the message.
            _ => crate::error::ErrorDetail::with_source("run slangc", error),
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
        RhiError::ShaderCompilation(format!("read `{}`: {error}", files.output.display()).into())
    })?;

    // Best effort: Vulkan reads `[numthreads]` out of the SPIR-V itself, so a slangc that
    // stopped emitting this should degrade to "unknown" rather than fail every compute compile.
    // Metal cannot, which is why `compile_metal_compute` does treat it as required.
    let threads_per_threadgroup = files.reflection.as_ref().and_then(|path| {
        let reflection = std::fs::read_to_string(path).ok()?;
        let size = thread_group_size(&reflection);
        if size.is_none() {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The key names a file that outlives the process, so it is an on-disk format.
    #[test]
    fn the_cache_key_format_is_stable() {
        let mut h = CacheHasher::new();
        assert_eq!(h.finish(), CacheHasher::OFFSET_BASIS);
        // Cross-checked against an independent FNV-1a-128, so this pins the algorithm.
        h.write(b"kiln");
        assert_eq!(h.finish(), 0x6946_4f9f_0f75_7277_b806_e969_f75b_2213);

        // Fields are length-prefixed; without that these two keys collide and the wrong
        // shader loads from cache.
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
