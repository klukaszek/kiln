//! The root struct the shaders read, the assembled Slang sources, and the pipelines built from
//! them.
//!
//! Constants the host and shader both depend on are defined here and emitted into the source. The
//! RHI compiler has no define mechanism and no include path, so a module is a concatenation of
//! strings rather than a file with `#include`s.

use glam::{IVec2, Vec4};

use crate::scene::Kind;
use kiln_rhi::{
    ColorTarget, ComputePso, ComputePsoDesc, Cull, Device, Format, GraphicsPso, GraphicsPsoDesc,
    RhiResult, SampleCount, ShaderStage, Topology, compiler, gpu_struct,
};

/// Threadgroup width for the passes that walk a cascade level, flattened over (direction, column).
///
/// Two SIMD groups. 32 measures 18% slower across the frame, and 128 and 256 are slower again:
/// `traceSegments` carries a ray query, and past 64 the Metal compiler has to spill to fit the
/// group, which costs more than the extra latency hiding buys.
pub(super) const CASCADE_THREADS: u32 = 64;

/// Threadgroup shape for the passes that walk a 2D grid: the fluence field and the broadphase.
pub(super) const FIELD_THREADS: [u32; 2] = [8, 8];

/// Threadgroup width for the buffer clear.
pub(super) const CLEAR_THREADS: u32 = 256;

/// Outline segments emitted per primitive when tessellating for the BVH. Uniform across kinds so
/// the whole scene tessellates in one flat dispatch.
pub(super) const EDGES_PER_PRIM: u32 = 64;

/// Bytes per stored `T` entry (`Half4`) and per stored `R` entry (`Half3`).
pub(super) const SEGMENT_STRIDE: u64 = 8;
pub(super) const CONE_STRIDE: u64 = 6;

/// Primitives one broadphase cell can hold. `buildGrid` reports the true overlap count, so an
/// overflow is detectable rather than silent.
pub(super) const GRID_CAPACITY: u32 = 48;

/// Broadphase resolution, on both axes.
pub(super) const GRID_CELLS: u32 = 128;

/// Cascade levels traced against the scene. The rest are merged from these, and raising it reduces
/// light leaking around sealed occluders at a cost of about a millisecond per level.
pub const DIRECT_TRACE_LEVELS: u32 = 3;

/// A `float16_t4` on the device: `T`'s radiance and transmittance. Only ever a pointee, so the host
/// never builds one; the field states the stride the cascade index arithmetic assumes.
#[repr(transparent)]
#[derive(Clone, Copy, zerocopy::IntoBytes, zerocopy::FromBytes, zerocopy::Immutable)]
pub struct Half4(pub [u16; 4]);

/// Three halves on the device: `R`'s angular fluence, which has no fourth channel to carry.
///
/// Six bytes, not eight. `R` is the most-read structure in the frame and the sweep is bandwidth
/// bound, so the padding a `float16_t4` would carry is worth removing. Declared as a struct of
/// scalars rather than a `float16_t3`, which both backends would round back up to eight.
#[repr(transparent)]
#[derive(Clone, Copy, zerocopy::IntoBytes, zerocopy::FromBytes, zerocopy::Immutable)]
pub struct Half3(pub [u16; 3]);

gpu_struct! {
    /// Everything every pass reads. One root rather than one per entry point: the passes overlap
    /// heavily in what they need, and a single layout keeps the host's per-dispatch write to the
    /// three or four fields that actually change.
    ///
    /// `Vec4` leads so the struct's own 16-byte alignment needs no interior padding. Three-float
    /// fields are deliberately absent: Slang gives `float3` 12 bytes on SPIR-V and 16 on Metal, so
    /// one in a root struct would silently disagree across backends.
    pub struct Root {
        sky: Vec4,

        scene: GpuPtr<Vec4, Read>,
        cell_count: GpuPtr<u32>,
        cell_prims: GpuPtr<u32>,
        cell_clear: GpuPtr<f32>,
        // Written by `tessellate`, read by every ray hit, so it cannot take the read-only
        // annotation the other scene buffers do.
        tri_prim: GpuPtr<u32>,
        vertices: GpuPtr<f32>,
        indices: GpuPtr<u32>,
        // The previous frame's resolved field, which surfaces re-emit.
        history: GpuPtr<Vec4, Read>,
        fluence_in: GpuPtr<Vec4, Read>,
        fluence_out: GpuPtr<Vec4>,
        segments_out: GpuPtr<Half4>,
        segments_prev: GpuPtr<Half4, Read>,
        segments_cur: GpuPtr<Half4, Read>,
        segments_next: GpuPtr<Half4, Read>,
        cones_out: GpuPtr<Half3>,
        cones_next: GpuPtr<Half3, Read>,
        cones0_q0: GpuPtr<Half3, Read>,
        cones0_q1: GpuPtr<Half3, Read>,
        cones0_q2: GpuPtr<Half3, Read>,
        cones0_q3: GpuPtr<Half3, Read>,
        tlas: AccelHandle,

        // Probe grid resolution, and the pixel space the scene is packed into.
        res: IVec2,
        // Swapchain resolution the final image is resolved at.
        out_res: IVec2,
        // (X, Y) of the canonical frame for the quadrant being solved; X is its gather axis.
        canon_res: IVec2,
        cells: IVec2,

        exposure: f32,
        surface_offset: f32,
        surface_falloff: f32,
        bounce: f32,
        quadrant: i32,
        // n, the level being written.
        level: i32,
        // N, the top level, whose cones hold the sky.
        levels: i32,
        prim_count: i32,
        surface_shading: i32,
        view_mode: u32,
        target_is_srgb: u32,
        pad0: u32,
        pad1: u32,
        pad2: u32,
    }
}

gpu_struct! {
    /// What the two verification passes read.
    ///
    /// These fields started out on [`Root`], on the reasoning that a few more bytes on a struct
    /// uploaded a hundred times a frame could not matter. They do, but not through bandwidth:
    /// Metal budgets registers per pipeline, and a root wide enough to push `buildSegments` over
    /// that budget gets the whole pipeline refused. Verification is not on the hot path and has no
    /// business setting the register ceiling for the passes that are.
    pub struct ReferenceRoot {
        sky: Vec4,
        scene: GpuPtr<Vec4, Read>,
        fluence_in: GpuPtr<Vec4, Read>,
        fluence_out: GpuPtr<Vec4>,
        res: IVec2,
        out_res: IVec2,
        prim_count: i32,
        ref_dir_begin: u32,
        ref_dir_count: u32,
        ref_dirs: u32,
        ref_max_steps: u32,
        ref_max_dist: f32,
        ref_eps: f32,
        pad0: u32,
        pad1: u32,
        pad2: u32,
    }
}

gpu_struct! {
    /// Zeroing a buffer that is read before anything writes it. Only the fluence field needs this:
    /// every cascade level is fully written each frame before it is read.
    pub struct ClearRoot {
        target: GpuPtr<u32>,
        count: u32,
        pad: u32,
    }
}

/// Every scalar the shaders share with the host.
fn constants() -> String {
    format!(
        "static const int KIND_CIRCLE = {circle};\n\
         static const int KIND_BOX = {box_};\n\
         static const int KIND_SEGMENT = {segment};\n\
         static const float PI = 3.14159265359;\n\
         static const float TWO_PI = 6.28318530718;\n\
         static const int EDGES_PER_PRIM = {EDGES_PER_PRIM};\n\
         static const int GRID_CAPACITY = {GRID_CAPACITY};\n\
         static const int DIRECT_TRACE_LEVELS = {DIRECT_TRACE_LEVELS};\n\
         typedef float16_t4 Half4;\n\
         struct Half3 {{ float16_t x;\n float16_t y;\n float16_t z; }};\n",
        circle = Kind::Circle as i32,
        box_ = Kind::Box as i32,
        segment = Kind::Segment as i32,
    )
}

/// Threadgroup dimensions, emitted separately so the clear module does not carry the rest.
fn thread_constants() -> String {
    format!(
        "static const uint CASCADE_THREADS = {CASCADE_THREADS}u;\n\
         static const uint FIELD_THREADS_X = {x}u;\n\
         static const uint FIELD_THREADS_Y = {y}u;\n",
        x = FIELD_THREADS[0],
        y = FIELD_THREADS[1],
    )
}

pub(super) fn solver_source() -> String {
    [
        &constants(),
        &thread_constants(),
        Root::SLANG,
        ReferenceRoot::SLANG,
        include_str!("shaders/scene.slang"),
        include_str!("shaders/holographic.slang"),
    ]
    .concat()
}

fn clear_source() -> String {
    [
        &format!("static const uint CLEAR_THREADS = {CLEAR_THREADS}u;\n"),
        ClearRoot::SLANG,
        include_str!("shaders/clear.slang"),
    ]
    .concat()
}

/// Ray query reaches the solver module through `traceIntervalRT`, so every entry point compiled
/// from it is declared with the capability even where the entry itself casts no rays.
const SOLVER_CAPABILITIES: &[&str] = &["spvRayQueryKHR"];

/// Directions the reference integrates per pixel unless `--dirs` says otherwise.
pub const REFERENCE_DIRECTIONS: u32 = 1024;

/// Sphere-marching steps one reference ray may take.
pub const REFERENCE_MAX_STEPS: u32 = 512;

/// Surface proximity a reference ray stops at, in scene pixels.
pub const REFERENCE_EPSILON: f32 = 0.1;

pub(super) struct Pipelines {
    pub(super) trace_segments: ComputePso,
    pub(super) build_segments: ComputePso,
    pub(super) build_fluence: ComputePso,
    pub(super) resolve_fluence: ComputePso,
    pub(super) blur_fluence: ComputePso,
    pub(super) tessellate: ComputePso,
    pub(super) build_grid: ComputePso,
    pub(super) light_field_linear: ComputePso,
    pub(super) reference_linear: ComputePso,
    pub(super) clear: ComputePso,
    pub(super) resolve: GraphicsPso,
}

impl Pipelines {
    pub(super) fn new(device: &Device, color_format: Format) -> RhiResult<Self> {
        let solver = solver_source();
        // Both shapes are declared by the shaders' own `[numthreads]`, which reflection now
        // carries onto the module, so they are no longer restated here.
        let cascade = |entry: &str| compute(device, &solver, entry);
        let field = |entry: &str| compute(device, &solver, entry);

        let clear_src = clear_source();
        let clear = device.create_compute_pso(
            // The threadgroup size comes from the shader's `[numthreads]` via reflection.
            &ComputePsoDesc {
                label: Some("hrc-clear"),
                ..Default::default()
            },
            &compiler::compile(device, &clear_src, "clearMain", ShaderStage::Compute, &[])?,
        )?;

        let resolve = device.create_graphics_pso(
            &GraphicsPsoDesc {
                topology: Topology::TriangleList,
                color_targets: &[ColorTarget::new(color_format)],
                depth_format: None,
                sample_count: SampleCount::S1,
                cull: Cull::None,
                label: Some("hrc-resolve"),
                ..Default::default()
            },
            &compiler::compile(
                device,
                &solver,
                "resolveVs",
                ShaderStage::Vertex,
                SOLVER_CAPABILITIES,
            )?,
            &compiler::compile(
                device,
                &solver,
                "resolveFs",
                ShaderStage::Pixel,
                SOLVER_CAPABILITIES,
            )?,
        )?;

        Ok(Self {
            trace_segments: cascade("traceSegments")?,
            build_segments: cascade("buildSegments")?,
            build_fluence: cascade("buildFluence")?,
            resolve_fluence: field("resolveFluence")?,
            blur_fluence: field("blurFluence")?,
            tessellate: cascade("tessellate")?,
            build_grid: field("buildGrid")?,
            light_field_linear: field("lightFieldLinear")?,
            reference_linear: field("referenceLinear")?,
            clear,
            resolve,
        })
    }
}

fn compute(device: &Device, source: &str, entry: &str) -> RhiResult<ComputePso> {
    let label = format!("hrc-{entry}");
    device.create_compute_pso(
        &ComputePsoDesc {
            label: Some(&label),
            ..Default::default()
        },
        &compiler::compile(
            device,
            source,
            entry,
            ShaderStage::Compute,
            SOLVER_CAPABILITIES,
        )?,
    )
}
