//! Progressive spectral GPU path tracer.
//!
//! [`PathTracer`] owns only pipelines and per-frame state — the scene lives in
//! [`crate::scene::gpu::GpuScene`] and is handed in each frame, so swapping
//! scenes never rebuilds a PSO. Per frame it records one compute accumulation
//! pass into the [`Film`] and a fullscreen blit of the running average.
//!
//! This is a *realtime* path tracer in the sense of Peters' reference
//! renderer: every frame traces `samples_per_frame` (default 1) paths for
//! every film pixel and presents the running average — the per-present GPU
//! cost is one low-spp pass by construction, with no scheduling layer. While
//! the camera is still the film accumulates at the display rate; any change
//! restarts it, so motion shows the live 1-spp image. `render_scale` divides
//! the film resolution (the blit upscales) when full-resolution paths exceed
//! the frame budget the display imposes — per-pass overhead is negligible
//! (measured: 64×1 spp ≈ 4×16 spp wall time), so a lower `samples_per_frame`
//! costs no convergence throughput.
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
mod roots;
mod sampler;

use glam::{UVec2, UVec4, Vec3, Vec4};
use kiln_rhi::{
    BlendState, BufferDesc, BumpAllocator, ColorTarget, CommandBuffer, CompareOp, ComputePso,
    ComputePsoDesc, DepthFlags, DepthStencilState, Device, Format, GpuAddress, GpuAllocation,
    GraphicsPso, GraphicsPsoDesc, MAX_FRAMES_IN_FLIGHT, MemoryType, SampleCount, ShaderStage,
    StageFlags, Topology,
};

use kiln_app::FrameCtx;

use crate::scene::Scene;
use crate::scene::gpu::{GpuMaterial, GpuScene};
use crate::scene::{Vertex, spectral};
use roots::{CLEAR_SOURCE, CLEAR_THREADS, CameraGpu, ClearRoot, DisplayRoot, TraceRoot};

pub const DEFAULT_TARGET_SPP: u32 = 1024;
/// One pass per present — the realtime loop. Raising this trades present rate
/// for nothing: per-pass overhead is noise next to per-sample cost.
pub const DEFAULT_SAMPLES_PER_FRAME: u32 = 1;

/// Number of f32 per film pixel: one band-radiance accumulator per spectral
/// bin, plus a trailing per-pixel sample count.
const FILM_STRIDE: u32 = spectral::SPECTRAL_BINS as u32 + 1;

/// Hero wavelengths per path, split between the two MIS sampling strategies:
/// `N_LIGHT_LANES` drawn from the light-importance CDF (good for the display
/// image and spiky illuminants), `N_UNIFORM_LANES` drawn uniformly in
/// wavelength (so the deep spectral tails get bounded-variance coverage). Their
/// sum is the float4 wavelength width the kernel carries — keep it at 4.
const N_LIGHT_LANES: u32 = 2;
const N_UNIFORM_LANES: u32 = 2;

/// Bytes of transient root data each frame slot may allocate.
const FRAME_ARENA_SIZE: u64 = 4096;


pub struct PathTracer {
    trace_pso: ComputePso,
    clear_pso: ComputePso,
    display_pso: GraphicsPso,
    /// One transient-argument arena per frame in flight, reset when its slot records.
    frame_arenas: [BumpAllocator; MAX_FRAMES_IN_FLIGHT],
    film: Film,
    target_spp: u32,
    samples_per_frame: u32,
    /// Film resolution divisor: trace at `display_extent / render_scale`, blit
    /// upscales. 1 = native.
    render_scale: u32,
    /// Per-bin linear-sRGB colour-matching response (the sensor), resident for
    /// the display blit. The CPU copy backs PNG/probe readback.
    cmf_bins: GpuAllocation,
    cmf_bins_cpu: Vec<Vec3>,
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
        render_scale: u32,
    ) -> anyhow::Result<Self> {
        let trace_src = format!(
            "{}{}{}{}{}",
            Vertex::SLANG,
            GpuMaterial::SLANG,
            TraceRoot::SLANG,
            sampler::source(),
            integrator::source()
        );
        let trace_shader = kiln_rhi::compiler::compile_with_caps(
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
                label: Some("spectral-trace".into()),
            },
            &trace_shader,
        )?;

        let clear_src = format!("{}{}", ClearRoot::SLANG, CLEAR_SOURCE);
        let clear_shader =
            kiln_rhi::compiler::compile(device, &clear_src, "clearMain", ShaderStage::Compute);
        let clear_pso = device.create_compute_pso(
            &ComputePsoDesc {
                root_constant_size: std::mem::size_of::<GpuAddress>() as u32,
                threads_per_threadgroup: [CLEAR_THREADS, 1, 1],
                label: Some("spectral-film-clear".into()),
            },
            &clear_shader,
        )?;

        let display_src = format!("{}{}", DisplayRoot::SLANG, display::SOURCE);
        let display_vs =
            kiln_rhi::compiler::compile(device, &display_src, "displayVs", ShaderStage::Vertex);
        let display_fs =
            kiln_rhi::compiler::compile(device, &display_src, "displayFs", ShaderStage::Pixel);
        let display_pso = device.create_graphics_pso(
            &GraphicsPsoDesc {
                topology: Topology::TriangleList,
                color_targets: vec![ColorTarget::new(color_format)],
                depth_format: Some(Format::D32Float),
                sample_count: SampleCount::S1,
                root_constant_size: 16,
                cull: kiln_rhi::Cull::None,
                blendstate: Some(BlendState::default()),
                label: Some("spectral-display".into()),
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
                        label: Some(format!("spectral-frame-arena-{slot}")),
                    })
                    .expect("create frame arena"),
            )
        });

        // The sensor: per-bin linear-sRGB colour-matching response. Uploaded
        // once (padded to float4) for the display blit; kept on the CPU for
        // PNG/probe readback. Light-independent, so it never changes.
        let cmf_bins_cpu = spectral::cmf_bins_linear_srgb();
        let cmf_padded: Vec<Vec4> = cmf_bins_cpu.iter().map(|c| c.extend(0.0)).collect();
        let cmf_bins = device.malloc(
            std::mem::size_of_val(cmf_padded.as_slice()) as u64,
            MemoryType::Default,
        )?;
        cmf_bins.upload_slice(&cmf_padded)?;

        Ok(Self {
            trace_pso,
            clear_pso,
            display_pso,
            frame_arenas,
            film: Film::new(),
            target_spp,
            samples_per_frame,
            render_scale: render_scale.max(1),
            cmf_bins,
            cmf_bins_cpu,
            display_target_is_srgb: display::format_is_srgb(color_format),
        })
    }

    /// The film resolution for a given display extent.
    fn film_extent(&self, display: UVec2) -> UVec2 {
        UVec2::new(
            display.x.div_ceil(self.render_scale).max(1),
            display.y.div_ceil(self.render_scale).max(1),
        )
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

        let film_extent = self.film_extent(ctx.extent);
        let camera = CameraGpu::from_scene(scene, film_extent);
        let film_key = camera.film_key(self.target_spp as u64);
        // Camera motion resets the film every frame it changes; only the
        // startup/resize resets are worth a log line.
        let resized = self.film.extent() != film_extent;
        let needs_clear = self.film.prepare(ctx.device, film_extent, film_key);
        let will_trace =
            self.film.sample_count() < self.target_spp && gpu_scene.light_count > 0;
        if !needs_clear && !will_trace {
            return;
        }

        // Order this frame's film writes after the previous in-flight frame's
        // trace writes and display reads. The barrier is recorded outside any
        // encoder, so it lands queue-scoped at the head of the next compute
        // encoder — which is why it must only be emitted when compute work
        // follows. Without it, the film clear recorded on every camera move
        // races the prior frame's display pass still reading the buffer.
        cmd.barrier(
            StageFlags::COMPUTE | StageFlags::PIXEL_SHADER,
            StageFlags::COMPUTE,
        );

        if needs_clear {
            self.record_film_clear(ctx, cmd);
            if resized {
                eprintln!(
                    "spectral path tracer reset: {}x{} film ({}x{} display), target spp={}, samples/frame={}, materials={}, lights={}",
                    film_extent.x,
                    film_extent.y,
                    ctx.extent.x,
                    ctx.extent.y,
                    self.target_spp,
                    self.samples_per_frame,
                    gpu_scene.material_count,
                    gpu_scene.light_count
                );
            }
        }
        if !will_trace {
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
            film: accum.gpu(),
            verts: gpu_scene.vertex_buffer.gpu(),
            triangle_materials: gpu_scene.triangle_material_buffer.gpu(),
            materials: gpu_scene.material_buffer.gpu(),
            light_triangles: gpu_scene.light_triangle_buffer.gpu(),
            spectrum: gpu_scene.spectrum_buffer.gpu(),
            lambda: gpu_scene.lambda_buffer.gpu(),
            tlas: accel.tlas.gpu(),
            dims0: UVec4::new(
                film_extent.x,
                film_extent.y,
                self.film.sample_count(),
                self.target_spp,
            ),
            dims1: UVec4::new(
                gpu_scene.triangle_count,
                gpu_scene.light_count,
                samples,
                gpu_scene.spectrum_len,
            ),
            dims2: UVec4::new(
                spectral::SPECTRAL_BINS as u32,
                N_LIGHT_LANES,
                N_UNIFORM_LANES,
                0,
            ),
        })
        .expect("upload trace root");

        cmd.set_compute_pipeline(&self.trace_pso);
        cmd.dispatch(
            root.gpu,
            film_extent.x.div_ceil(integrator::THREADS_X),
            film_extent.y.div_ceil(integrator::THREADS_Y),
            1,
        );
        self.film.add_samples(samples);
        self.log_progress();
        cmd.barrier(StageFlags::COMPUTE, StageFlags::PIXEL_SHADER);
    }

    /// Blit the running average to the bound render target, upscaling when the
    /// film renders below display resolution.
    pub fn render(&mut self, ctx: &FrameCtx, cmd: &mut CommandBuffer) {
        let Some(accum) = self.film.accum() else {
            return;
        };
        if self.film.extent() != self.film_extent(ctx.extent) {
            return;
        }

        let root = self.frame_arenas[ctx.slot]
            .alloc(std::mem::size_of::<DisplayRoot>() as u64, 16)
            .expect("display root from frame arena");
        let film = self.film.extent();
        root.upload(&DisplayRoot {
            dims: UVec4::new(
                ctx.extent.x,
                ctx.extent.y,
                spectral::SPECTRAL_BINS as u32,
                self.display_target_is_srgb as u32,
            ),
            film_dims: UVec4::new(film.x, film.y, FILM_STRIDE, 0),
            film: accum.gpu(),
            cmf: self.cmf_bins.gpu(),
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
        Ok(display::film_to_rgba8(
            self.film.rows()?,
            FILM_STRIDE as usize,
            &self.cmf_bins_cpu,
        ))
    }

    /// Per-pixel band-integrated spectral radiance, row-major
    /// `[height][width][SPECTRAL_BINS]` — the raw spectral capture.
    pub fn spectral_bands(&self) -> anyhow::Result<Vec<f32>> {
        Ok(display::film_to_bands(self.film.rows()?, FILM_STRIDE as usize))
    }

    /// Centre wavelength (nm) of each spectral bin, for labelling exports.
    pub fn spectral_bin_centers() -> Vec<f32> {
        (0..spectral::SPECTRAL_BINS)
            .map(spectral::spectral_bin_center)
            .collect()
    }

    /// Record the GPU zero of the spectral film (every bin and count).
    fn record_film_clear(&mut self, ctx: &FrameCtx, cmd: &mut CommandBuffer) {
        let accum = self.film.accum().expect("film prepared");
        let film = self.film.extent();
        let float_count = film.x * film.y * FILM_STRIDE;
        let root = self.frame_arenas[ctx.slot]
            .alloc(std::mem::size_of::<ClearRoot>() as u64, 16)
            .expect("clear root from frame arena");
        root.upload(&ClearRoot {
            film: accum.gpu(),
            count: float_count,
            _pad: 0,
        })
        .expect("upload clear root");

        cmd.set_compute_pipeline(&self.clear_pso);
        cmd.dispatch(root.gpu, float_count.div_ceil(CLEAR_THREADS), 1, 1);
        // The trace dispatch lands in a *different* compute encoder (every
        // set_compute_pipeline opens a fresh one), and a COMPUTE→COMPUTE
        // barrier inside this encoder is encoder-scoped — it would not order
        // the trace after the clear at all. Including PIXEL_SHADER in the
        // destination forces the queue-scoped form, ordering everything that
        // follows (trace and display) after the clear.
        cmd.barrier(
            StageFlags::COMPUTE,
            StageFlags::COMPUTE | StageFlags::PIXEL_SHADER,
        );
    }

    fn log_progress(&self) {
        let sample_count = self.film.sample_count();
        if sample_count == self.target_spp
            || (sample_count >= 64 && sample_count.is_power_of_two())
        {
            eprintln!(
                "spectral path tracer progress: {}/{} spp",
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
        let film_bytes = pixels * FILM_STRIDE as u64 * std::mem::size_of::<f32>() as u64;
        let accum = match self.accum.take() {
            Some(existing) if self.extent == extent => existing,
            stale => {
                // Freeing must be explicit (dropping a GpuAllocation leaks it,
                // residency included). Safe here: extent changes only follow
                // the harness's wait_idle on resize, so no frame in flight
                // still reads the old buffer.
                if let Some(stale) = stale {
                    device.free(stale);
                }
                device
                    .malloc(film_bytes, MemoryType::Default)
                    .expect("alloc spectral film")
            }
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

    /// The film as `[bins.., count]` rows, one per pixel (row-major). Resolving
    /// these to RGB or per-band radiance is [`display`]'s job.
    fn rows(&self) -> anyhow::Result<&[f32]> {
        let accum = self
            .accum
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("film has no accumulation buffer"))?;
        Ok(accum.as_slice::<f32>()?)
    }
}

