//! Slang shader compiler with an on-disk cache.
//!
//! Compiles Slang source to the active backend's format (SPIR-V or metallib) and loads it as a
//! [`ShaderModule`]. Artifacts are cached in the user's cache directory, keyed on source, entry
//! point, stage, target, capabilities and slangc version.
//!
//! Shells out to `slangc` on `PATH`, which suits tests, examples and iteration. A shipped
//! application should compile offline and pass the bytes to [`Device::create_shader_module`].
//! Calls report [`ShaderCompilation`](crate::RhiError::ShaderCompilation) when `slangc` is missing.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use twox_hash::XxHash3_128;

use crate::error::ErrorDetail;
use crate::{Backend, Device, RhiError, RhiResult, ShaderModule, ShaderModuleDesc, ShaderStage};

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

/// Mesh shaders get plain `metal`: Slang's reflection omits their `[numthreads]`
/// (shader-slang/slang#13493), so the draw cannot match a required size and Metal would drop it.
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

/// A cached artifact starts with its `[numthreads]` as three little-endian `u32`s, zero when
/// unknown, so a cache hit needs no reflection.
const CACHE_HEADER_BYTES: usize = 12;

/// Compile `src`'s `entry` point for the device's backend. `capabilities` takes extra Slang
/// capabilities (e.g. `"spvRayQueryKHR"`), `&[]` for none.
pub fn compile(
    device: &Device,
    src: &str,
    entry: &str,
    stage: ShaderStage,
    capabilities: &[&str],
) -> RhiResult<ShaderModule> {
    let (target, ext) = match device.backend() {
        Backend::Vulkan => ("spirv", "spv"),
        Backend::Metal => ("metallib", "metallib"),
    };
    // Added here rather than at the command line so they are part of the cache key.
    let mut capabilities = capabilities.to_vec();
    capabilities.push(match (device.backend(), stage) {
        (Backend::Vulkan, _) => SPIRV_DESCRIPTOR_HEAP_CAPABILITY,
        (Backend::Metal, ShaderStage::Mesh) => METAL_MESH_CAPABILITY,
        (Backend::Metal, _) => METAL_CAPABILITY,
    });
    let src = format!("{ACCEL_PRELUDE}{src}");

    let path = cache_dir()?.join(format!(
        "{:032x}.{ext}",
        cache_key(&src, entry, stage, target, &capabilities)
    ));
    let compiled = match read_cache(&path) {
        Some(compiled) => compiled,
        None => {
            let compiled = invoke_slangc(&src, entry, stage, target, ext, &capabilities)?;
            write_cache(&path, &compiled);
            compiled
        }
    };

    device.create_shader_module(&ShaderModuleDesc {
        code: &compiled.code,
        entry_point: entry,
        stage,
        threads_per_threadgroup: compiled.threads_per_threadgroup,
        label: Some(entry),
    })
}

struct Compiled {
    code: Vec<u8>,
    /// `[numthreads]` for a compute or mesh entry point, when slangc reports it.
    threads_per_threadgroup: Option<[u32; 3]>,
}

fn io_error(what: String) -> impl FnOnce(std::io::Error) -> RhiError {
    move |error| RhiError::ShaderCompilation(ErrorDetail::with_source(what, error))
}

fn cache_dir() -> RhiResult<PathBuf> {
    let dir = dirs::cache_dir()
        .ok_or_else(|| {
            RhiError::ShaderCompilation("this platform has no user cache directory".into())
        })?
        .join("kiln-rhi")
        .join("shaders");
    std::fs::create_dir_all(&dir).map_err(io_error(format!("create `{}`", dir.display())))?;
    Ok(dir)
}

fn cache_key(
    src: &str,
    entry: &str,
    stage: ShaderStage,
    target: &str,
    capabilities: &[&str],
) -> u128 {
    let mut caps = capabilities.to_vec();
    caps.sort_unstable();
    let fields = [
        slangc_version(),
        src.as_bytes(),
        entry.as_bytes(),
        stage_str(stage).as_bytes(),
        target.as_bytes(),
        SLANG_OPTIMIZATION_LEVEL.as_bytes(),
    ]
    .into_iter()
    .chain(caps.iter().map(|cap| cap.as_bytes()));

    // Length-prefixed, so `["ab", "c"]` and `["a", "bc"]` hash differently.
    let mut bytes = Vec::with_capacity(src.len() + 256);
    for field in fields {
        bytes.extend_from_slice(&(field.len() as u64).to_le_bytes());
        bytes.extend_from_slice(field);
    }
    XxHash3_128::oneshot(&bytes)
}

/// `slangc -v`'s output, probed once. Empty when slangc is unavailable.
fn slangc_version() -> &'static [u8] {
    static VERSION: OnceLock<Vec<u8>> = OnceLock::new();
    VERSION.get_or_init(|| {
        Command::new("slangc")
            .arg("-v")
            .output()
            .ok()
            .filter(|out| out.status.success())
            .map(|out| [out.stdout, out.stderr].concat())
            .unwrap_or_default()
    })
}

fn read_cache(path: &Path) -> Option<Compiled> {
    let mut bytes = std::fs::read(path).ok()?;
    if bytes.len() <= CACHE_HEADER_BYTES {
        return None;
    }
    let code = bytes.split_off(CACHE_HEADER_BYTES);
    let dim = |i: usize| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
    let size = [dim(0), dim(1), dim(2)];
    Some(Compiled {
        code,
        threads_per_threadgroup: (!size.contains(&0)).then_some(size),
    })
}

/// Best effort: a failed write only costs a recompile next time. Written to a temporary file and
/// renamed into place, so a reader never sees a partial artifact.
fn write_cache(path: &Path, compiled: &Compiled) {
    let Some(dir) = path.parent() else { return };
    let Ok(mut file) = tempfile::NamedTempFile::new_in(dir) else {
        return;
    };
    let threads = compiled.threads_per_threadgroup.unwrap_or([0; 3]);
    let header: Vec<u8> = threads.iter().flat_map(|d| d.to_le_bytes()).collect();
    if file.write_all(&header).is_ok() && file.write_all(&compiled.code).is_ok() {
        let _ = file.persist(path);
    }
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

fn invoke_slangc(
    src: &str,
    entry: &str,
    stage: ShaderStage,
    target: &str,
    ext: &str,
    capabilities: &[&str],
) -> RhiResult<Compiled> {
    let scratch = tempfile::Builder::new()
        .prefix("kiln-slangc")
        .tempdir()
        .map_err(io_error("create a scratch directory for slangc".into()))?;
    let source = scratch.path().join("shader.slang");
    let output = scratch.path().join(format!("shader.{ext}"));
    let reflection = matches!(stage, ShaderStage::Compute | ShaderStage::Mesh)
        .then(|| scratch.path().join("reflection.json"));
    std::fs::write(&source, src).map_err(io_error(format!("write `{}`", source.display())))?;

    let mut cmd = Command::new("slangc");
    cmd.arg(&source)
        .args([
            "-target",
            target,
            "-entry",
            entry,
            "-stage",
            stage_str(stage),
        ])
        .arg(format!("-O{SLANG_OPTIMIZATION_LEVEL}"));
    if let Some(reflection) = &reflection {
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
    cmd.arg("-o").arg(&output);

    let result = cmd.output().map_err(|error| {
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
    if !result.status.success() {
        return Err(RhiError::ShaderCompilation(
            format!(
                "slangc failed compiling `{entry}` for {target}:\n{}",
                String::from_utf8_lossy(&result.stderr)
            )
            .into(),
        ));
    }

    let code = std::fs::read(&output).map_err(io_error(format!("read `{}`", output.display())))?;

    // Missing reflection is not fatal here: Vulkan reads `[numthreads]` from the SPIR-V, and
    // Metal reports the missing size when the pipeline is created.
    let threads_per_threadgroup = reflection.as_ref().and_then(|path| {
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
    fn a_cached_artifact_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shader.spv");
        for threads in [Some([64, 2, 1]), None] {
            write_cache(
                &path,
                &Compiled {
                    code: vec![1, 2, 3],
                    threads_per_threadgroup: threads,
                },
            );
            let read = read_cache(&path).unwrap();
            assert_eq!(read.code, [1, 2, 3]);
            assert_eq!(read.threads_per_threadgroup, threads);
        }
    }
}
