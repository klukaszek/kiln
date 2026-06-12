//! Progressive spectral GPU path tracer.
//!
//! [`PathTracer`] owns only pipelines and per-frame state — the scene lives in
//! [`crate::scene::gpu::GpuScene`] and is handed in each frame, so swapping
//! scenes never rebuilds a PSO. Per frame it records one compute accumulation
//! pass into the [`Film`] and a fullscreen blit of the running average.
//!
//! Realtime invariants:
//! - All per-frame GPU arguments (trace + display roots) come from a bump arena
//!   owned by the frame's slot, so a recording frame never touches memory an
//!   in-flight frame still reads.
//! - The [`Film`] is keyed by an invalidation hash of the camera (and sample
//!   target); any change restarts accumulation automatically.
//!
//! Submodules hold the shader-heavy pieces: [`sampler`] (Owen-scrambled
//! Sobol'), [`integrator`] (the spectral transport kernel), [`display`]
//! (blit + readback tonemap). The film and the root/camera layouts live below —
//! they are small and only this file consumes them.

mod display;
mod integrator;
mod sampler;

use glam::{DVec3, UVec2, UVec4, Vec4};
use kiln_rhi::{
    BlendState, BufferDesc, BumpAllocator, ColorTarget, CommandBuffer, CompareOp, ComputePso,
    ComputePsoDesc, DepthFlags, DepthStencilState, Device, Format, GpuAddress, GpuAllocation,
    GraphicsPso, GraphicsPsoDesc, MAX_FRAMES_IN_FLIGHT, MemoryType, SampleCount, ShaderStage,
    StageFlags, Topology, gpu_struct,
};

use crate::common::{self, FrameCtx};
use crate::scene::Scene;
use crate::scene::gpu::{GpuMaterial, GpuScene};
use crate::scene::Vertex;

pub const DEFAULT_TARGET_SPP: u32 = 1024;
pub const DEFAULT_SAMPLES_PER_FRAME: u32 = 16;

/// Bytes of transient root data each frame slot may allocate.
const FRAME_ARENA_SIZE: u64 = 4096;

/// Zero the accumulation buffer on the GPU. Recorded ahead of the trace pass
/// whenever the film resets, so history invalidation needs no CPU writes and
/// stays ordered against in-flight frames by queue submission order.
const CLEAR_SOURCE: &str = /*slang*/
    r#"
[shader("compute")]
[numthreads(256, 1, 1)]
void clearMain(uint3 tid : SV_DispatchThreadID, uniform ClearRoot* r)
{
    if (tid.x < r.count) {
        r.accum[tid.x] = float4(0.0);
    }
}
"#;

const CLEAR_THREADS: u32 = 256;

gpu_struct! {
    struct ClearRoot {
        accum: GpuAddress as "float4*",
        count: u32 as "uint",
        _pad: u32 as "uint",
    }
}

pub struct PathTracer {
    trace_pso: ComputePso,
    clear_pso: ComputePso,
    display_pso: GraphicsPso,
    /// One transient-argument arena per frame in flight, reset when its slot records.
    frame_arenas: [BumpAllocator; MAX_FRAMES_IN_FLIGHT],
    film: Film,
    target_spp: u32,
    samples_per_frame: u32,
    display_target_is_srgb: bool,
}

impl PathTracer {
    /// Compile the pipelines. Scene-independent: fails only when the device can't
    /// trace (no ray-query support) or shaders don't compile.
    pub fn new(
        device: &Device,
        color_format: Format,
        target_spp: u32,
        samples_per_frame: u32,
    ) -> anyhow::Result<Self> {
        let trace_src = format!(
            "{}{}{}{}{}",
            Vertex::SLANG,
            GpuMaterial::SLANG,
            TraceRoot::SLANG,
            sampler::source(),
            integrator::source()
        );
        let trace_shader = common::compile_with_caps(
            device,
            &trace_src,
            "traceMain",
            ShaderStage::Compute,
            &["spvRayQueryKHR"],
        );
        let trace_pso = device.create_compute_pso(
            &ComputePsoDesc {
                root_constant_size: std::mem::size_of::<GpuAddress>() as u32,
                threads_per_threadgroup: [integrator::THREADS_X, integrator::THREADS_Y, 1],
                label: Some("cornell-trace".into()),
            },
            &trace_shader,
        )?;

        let clear_src = format!("{}{}", ClearRoot::SLANG, CLEAR_SOURCE);
        let clear_shader = common::compile(device, &clear_src, "clearMain", ShaderStage::Compute);
        let clear_pso = device.create_compute_pso(
            &ComputePsoDesc {
                root_constant_size: std::mem::size_of::<GpuAddress>() as u32,
                threads_per_threadgroup: [CLEAR_THREADS, 1, 1],
                label: Some("cornell-film-clear".into()),
            },
            &clear_shader,
        )?;

        let display_src = format!("{}{}", DisplayRoot::SLANG, display::SOURCE);
        let display_vs = common::compile(device, &display_src, "displayVs", ShaderStage::Vertex);
        let display_fs = common::compile(device, &display_src, "displayFs", ShaderStage::Pixel);
        let display_pso = device.create_graphics_pso(
            &GraphicsPsoDesc {
                topology: Topology::TriangleList,
                color_targets: vec![ColorTarget::new(color_format)],
                depth_format: Some(Format::D32Float),
                sample_count: SampleCount::S1,
                root_constant_size: 16,
                cull: kiln_rhi::Cull::None,
                blendstate: Some(BlendState::default()),
                label: Some("cornell-display".into()),
                ..Default::default()
            },
            &display_vs,
            &display_fs,
        )?;

        let frame_arenas = std::array::from_fn(|slot| {
            BumpAllocator::new(
                device
                    .create_buffer(&BufferDesc {
                        size: FRAME_ARENA_SIZE,
                        memory: MemoryType::Default,
                        label: Some(format!("cornell-frame-arena-{slot}")),
                    })
                    .expect("create frame arena"),
            )
        });

        Ok(Self {
            trace_pso,
            clear_pso,
            display_pso,
            frame_arenas,
            film: Film::new(),
            target_spp,
            samples_per_frame,
            display_target_is_srgb: display::format_is_srgb(color_format),
        })
    }

    /// Record this frame's accumulation pass. Must run before [`Self::render`]
    /// each frame: it resets the slot's arena and prepares the film.
    pub fn pre_render(
        &mut self,
        ctx: &FrameCtx,
        cmd: &mut CommandBuffer,
        scene: &Scene,
        gpu_scene: &GpuScene,
    ) {
        let Some(accel) = gpu_scene.accel.as_ref() else {
            return;
        };
        self.frame_arenas[ctx.slot].reset();

        let camera = CameraGpu::from_scene(scene, ctx.extent);
        let film_key = camera.film_key(self.target_spp as u64);
        if self.film.prepare(ctx.device, ctx.extent, film_key) {
            self.record_film_clear(ctx, cmd);
            eprintln!(
                "cornell path tracer reset: {}x{}, target spp={}, samples/frame={}, materials={}, lights={}",
                ctx.extent.x,
                ctx.extent.y,
                self.target_spp,
                self.samples_per_frame,
                gpu_scene.material_count,
                gpu_scene.light_count
            );
        }
        if self.film.sample_count() >= self.target_spp || gpu_scene.light_count == 0 {
            return;
        }

        let samples = self
            .samples_per_frame
            .min(self.target_spp - self.film.sample_count());
        let accum = self.film.accum().expect("film prepared");
        let root = self.frame_arenas[ctx.slot]
            .alloc(std::mem::size_of::<TraceRoot>() as u64, 16)
            .expect("trace root from frame arena");
        root.upload(&TraceRoot {
            cam_pos: camera.pos,
            cam_right: camera.right,
            cam_up: camera.up,
            cam_forward: camera.forward,
            lens: camera.lens,
            accum: accum.gpu(),
            verts: gpu_scene.vertex_buffer.gpu(),
            triangle_materials: gpu_scene.triangle_material_buffer.gpu(),
            materials: gpu_scene.material_buffer.gpu(),
            light_triangles: gpu_scene.light_triangle_buffer.gpu(),
            spectrum: gpu_scene.spectrum_buffer.gpu(),
            dims0: UVec4::new(
                ctx.extent.x,
                ctx.extent.y,
                self.film.sample_count(),
                self.target_spp,
            ),
            dims1: UVec4::new(
                gpu_scene.triangle_count,
                gpu_scene.light_count,
                samples,
                gpu_scene.spectrum_len,
            ),
        })
        .expect("upload trace root");

        cmd.set_compute_pipeline(&self.trace_pso);
        cmd.bind_acceleration_structure(1, &accel.tlas);
        cmd.dispatch(
            root.gpu,
            ctx.extent.x.div_ceil(integrator::THREADS_X),
            ctx.extent.y.div_ceil(integrator::THREADS_Y),
            1,
        );
        self.film.add_samples(samples);
        self.log_progress();
        cmd.barrier(StageFlags::COMPUTE, StageFlags::PIXEL_SHADER);
    }

    /// Blit the running average to the bound render target.
    pub fn render(&mut self, ctx: &FrameCtx, cmd: &mut CommandBuffer) {
        let Some(accum) = self.film.accum() else {
            return;
        };
        if self.film.extent() != ctx.extent {
            return;
        }

        let root = self.frame_arenas[ctx.slot]
            .alloc(std::mem::size_of::<DisplayRoot>() as u64, 16)
            .expect("display root from frame arena");
        root.upload(&DisplayRoot {
            dims: UVec4::new(
                ctx.extent.x,
                ctx.extent.y,
                self.film.sample_count().max(1),
                self.display_target_is_srgb as u32,
            ),
            accum: accum.gpu(),
        })
        .expect("upload display root");

        cmd.set_graphics_pipeline(&self.display_pso);
        cmd.set_depth_stencil_state(&DepthStencilState {
            depth_mode: DepthFlags::empty(),
            depth_test: CompareOp::Always,
            stencil_read_mask: 0,
            stencil_write_mask: 0,
            ..Default::default()
        });
        cmd.draw(None, root.gpu, 3, 1, 0, 0);
    }

    pub fn is_complete(&self) -> bool {
        self.film.sample_count() >= self.target_spp
    }

    pub fn sample_count(&self) -> u32 {
        self.film.sample_count()
    }

    pub fn target_spp(&self) -> u32 {
        self.target_spp
    }

    pub fn samples_per_frame(&self) -> u32 {
        self.samples_per_frame
    }

    pub fn extent(&self) -> UVec2 {
        self.film.extent()
    }

    pub fn tonemapped_rgba8(&self) -> anyhow::Result<Vec<u8>> {
        self.film.tonemapped_rgba8(display::tonemap_channel)
    }

    /// Record the GPU zero of the film's accumulation buffer.
    fn record_film_clear(&mut self, ctx: &FrameCtx, cmd: &mut CommandBuffer) {
        let accum = self.film.accum().expect("film prepared");
        let pixel_count = ctx.extent.x * ctx.extent.y;
        let root = self.frame_arenas[ctx.slot]
            .alloc(std::mem::size_of::<ClearRoot>() as u64, 16)
            .expect("clear root from frame arena");
        root.upload(&ClearRoot {
            accum: accum.gpu(),
            count: pixel_count,
            _pad: 0,
        })
        .expect("upload clear root");

        cmd.set_compute_pipeline(&self.clear_pso);
        cmd.dispatch(root.gpu, pixel_count.div_ceil(CLEAR_THREADS), 1, 1);
        cmd.barrier(StageFlags::COMPUTE, StageFlags::COMPUTE);
    }

    fn log_progress(&self) {
        let sample_count = self.film.sample_count();
        if sample_count == self.target_spp
            || (sample_count >= 64 && sample_count.is_power_of_two())
        {
            eprintln!(
                "cornell path tracer progress: {}/{} spp",
                sample_count, self.target_spp
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Film: progressive accumulation state. The caller hands `prepare` an
// invalidation key hashing every input that makes old samples stale (camera
// basis, sample target — later: scene generation, render scale); a key or
// extent change resets the history. Each texel accumulates the linear-sRGB
// sensor response of the spectral estimator plus a sample count.
// ---------------------------------------------------------------------------

struct Film {
    accum: Option<GpuAllocation>,
    extent: UVec2,
    sample_count: u32,
    key: u64,
}

impl Film {
    fn new() -> Self {
        Self {
            accum: None,
            extent: UVec2::ZERO,
            sample_count: 0,
            key: 0,
        }
    }

    /// Make the accumulation buffer match `extent` and `key`, resetting history
    /// when either changed. Returns `true` on reset — the caller must record a
    /// GPU clear of the buffer before the next trace pass; queue ordering keeps
    /// that clear correct even with frames in flight.
    fn prepare(&mut self, device: &Device, extent: UVec2, key: u64) -> bool {
        if self.extent == extent && self.key == key && self.accum.is_some() {
            return false;
        }

        let pixels = (extent.x as u64) * (extent.y as u64);
        let accum = match self.accum.take() {
            Some(existing) if self.extent == extent => existing,
            _ => device
                .malloc(
                    pixels * std::mem::size_of::<[f32; 4]>() as u64,
                    MemoryType::Default,
                )
                .expect("alloc accumulation buffer"),
        };

        self.accum = Some(accum);
        self.extent = extent;
        self.sample_count = 0;
        self.key = key;
        true
    }

    fn accum(&self) -> Option<&GpuAllocation> {
        self.accum.as_ref()
    }

    fn extent(&self) -> UVec2 {
        self.extent
    }

    fn sample_count(&self) -> u32 {
        self.sample_count
    }

    fn add_samples(&mut self, samples: u32) {
        self.sample_count += samples;
    }

    /// Read back the accumulated image as tonemapped RGBA8 (headless PNG path).
    fn tonemapped_rgba8(&self, tonemap: impl Fn(f32, f32) -> u8) -> anyhow::Result<Vec<u8>> {
        let accum = self
            .accum
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("film has no accumulation buffer"))?;
        let pixels = accum.as_slice::<[f32; 4]>()?;
        let sample_count = self.sample_count.max(1) as f32;
        let mut rgba = Vec::with_capacity(pixels.len() * 4);

        for pixel in pixels {
            rgba.extend_from_slice(&[
                tonemap(pixel[0], sample_count),
                tonemap(pixel[1], sample_count),
                tonemap(pixel[2], sample_count),
                255,
            ]);
        }

        Ok(rgba)
    }
}

// ---------------------------------------------------------------------------
// Per-pass root layouts and the camera basis the trace kernel consumes. Roots
// are transient: bump-allocated per frame slot, never persistent.
// ---------------------------------------------------------------------------

gpu_struct! {
    struct TraceRoot {
        cam_pos: Vec4 as "float4",
        cam_right: Vec4 as "float4",
        cam_up: Vec4 as "float4",
        cam_forward: Vec4 as "float4",
        lens: Vec4 as "float4",
        accum: GpuAddress as "float4*",
        verts: GpuAddress as "Vertex*",
        triangle_materials: GpuAddress as "uint*",
        materials: GpuAddress as "GpuMaterial*",
        light_triangles: GpuAddress as "uint*",
        spectrum: GpuAddress as "float4*", // baked light spectrum table
        dims0: UVec4 as "uint4", // width, height, sample_index, max_spp
        dims1: UVec4 as "uint4", // tri_count, light_count, samples_per_frame, spectrum_len
    }
}

gpu_struct! {
    struct DisplayRoot {
        dims: UVec4 as "uint4", // width, height, sample_count, target_is_srgb
        accum: GpuAddress as "float4*",
    }
}

/// Camera basis in the shape the trace kernel consumes.
struct CameraGpu {
    pos: Vec4,
    right: Vec4,
    up: Vec4,
    forward: Vec4,
    lens: Vec4,
}

impl CameraGpu {
    fn from_scene(scene: &Scene, extent: UVec2) -> Self {
        // The world matrix is column-vector glam; its x/y/z columns are the
        // camera's right/up/back axes, w its position.
        let world = &scene.camera.world;
        let aspect = extent.x as f32 / extent.y.max(1) as f32;
        let tan_half_fovy = (scene.camera.usd.vertical_fov_rad() * 0.5).tan();
        let basis = |axis: glam::DVec4| axis.truncate().normalize_or(DVec3::Z).as_vec3().extend(0.0);

        Self {
            pos: world.w_axis.as_vec4(),
            right: basis(world.x_axis),
            up: basis(world.y_axis),
            forward: basis(-world.z_axis),
            lens: Vec4::new(aspect, tan_half_fovy, 0.0, 0.0),
        }
    }

    /// Fold every accumulated-sample-invalidating camera input into a film key.
    fn film_key(&self, seed: u64) -> u64 {
        let mut key = seed;
        for vec in [self.pos, self.right, self.up, self.forward, self.lens] {
            for component in vec.to_array() {
            // FNV-1a over the raw bits: cheap, stable, and exact-equality
            // semantics (any camera change at all restarts accumulation).
                key ^= component.to_bits() as u64;
                key = key.wrapping_mul(0x100000001b3);
            }
        }
        key
    }
}
