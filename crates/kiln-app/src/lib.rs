//! Shared windowing harness for the Kiln demos.
//!
//! The headless integration tests render into an offscreen texture and read it back; these
//! examples render the same shaders into a real window via the RHI's surface + swapchain. The
//! harness owns the winit event loop, the device, and the per-frame present loop; each example
//! just builds its pipelines and records its draws by implementing [`Example`].
//!
//! # egui debug overlay
//!
//! The overlay is gated entirely by the `egui` build feature (on by default), not a runtime flag:
//! build with it and every demo gets an egui overlay on top of its render; build
//! `--no-default-features` and the harness is plain egui-free windowing (the RHI core never depends
//! on egui either way). When active, the overlay paints whatever the example injects by overriding
//! [`Example::ui`], in a second color-load pass so it composites over depth-using examples without
//! a dynamic-rendering format mismatch.

#![allow(dead_code)]

use std::sync::LazyLock;
use std::time::Instant;

use clap::Parser;
use glam::UVec2;

use kiln_rhi::{
    Allocation, ColorAttachment, CommandBuffer, DepthAttachment, Device, DeviceDesc, Format,
    LoadOp, MAX_FRAMES_IN_FLIGHT, MemoryType, RenderPassDesc, RenderTarget, SampleCount, StoreOp,
    Surface, SurfaceDesc, Swapchain, SwapchainDesc, Texture, TextureDesc, TextureDimension,
    TextureUsage,
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

/// Read once: `var_os` allocates and takes the environment lock, which the frame loop should not
/// pay — least of all in the path that exists to measure it.
static FRAME_BENCH: LazyLock<bool> =
    LazyLock::new(|| std::env::var_os("KILN_FRAME_BENCH").is_some());

/// Per-frame recording context: the device, the target extent, and which of the
/// [`MAX_FRAMES_IN_FLIGHT`] slots this frame occupies.
///
/// `slot` is the realtime invariant: when this frame records, up to `MAX_FRAMES_IN_FLIGHT - 1`
/// earlier frames may still execute on the GPU, so any CPU-written transient (root structs, bump
/// arenas) must be keyed by `slot` — the harness only guarantees that *this slot's* previous frame
/// has retired.
pub struct FrameCtx<'a> {
    pub device: &'a Device,
    pub extent: UVec2,
    pub slot: usize,
}

/// Harness data available while an example builds its egui overlay.
#[cfg(feature = "egui")]
#[derive(Clone, Copy)]
pub struct PerformanceStats {
    /// CPU time spent recording and submitting the current frame, excluding egui construction and
    /// the frame-slot acquisition wait.
    pub cpu_ms: f64,
    /// GPU queue time bracketed by the frame timestamps, including pre-render compute and present.
    pub gpu_ms: f64,
    /// Time spent acquiring the next frame slot/image. This is commonly compositor/vsync or a
    /// frames-in-flight fence wait, not renderer CPU work.
    pub wait_ms: f64,
    pub backend: &'static str,
}

/// What each windowed example implements: build its pipelines once, then record draw commands into
/// the per-frame render pass.
pub trait Example {
    /// Build pipelines/resources. `color_format` is the swapchain's colour format — any PSO colour
    /// target must match it.
    fn new(device: &Device, color_format: Format) -> Self
    where
        Self: Sized;

    /// Opt into a depth buffer. When `Some(format)`, the harness creates a swapchain-sized depth
    /// texture (recreated on resize) and binds it cleared to 1.0 for every render pass.
    fn depth_format(&self) -> Option<Format> {
        None
    }

    /// Observe window events (keyboard, mouse) ahead of the harness's own handling. The harness
    /// still owns close/Esc/resize; examples use this for interaction such as camera controls.
    /// When the egui overlay is active it consumes input first, so events it handled (typing in a
    /// text field, dragging a window) still arrive here but should generally be ignored.
    fn window_event(&mut self, _event: &WindowEvent) {}

    /// Record work that must happen before the swapchain render pass, such as compute accumulation
    /// for progressive renderers. Default examples do nothing here.
    fn pre_render(&mut self, _ctx: &FrameCtx, _cmd: &mut CommandBuffer) {}

    /// Record draws for one frame. The render pass is already begun on the acquired swapchain image
    /// (cleared) with viewport + scissor set to the full extent. When [`Self::depth_format`] is
    /// `Some`, a cleared depth attachment is bound too.
    fn render(&mut self, ctx: &FrameCtx, cmd: &mut CommandBuffer);

    /// Whether the harness should keep requesting redraws without an external event. Progressive
    /// examples return false once they have converged; input events still request a redraw.
    fn wants_continuous_redraw(&self) -> bool {
        true
    }

    /// Emit the example's egui overlay UI for this frame. Only present when the crate is built with
    /// the `egui` feature; with it the overlay is always active. Override to inject UI; the default
    /// emits nothing. Per-frame timing and backend information are supplied to the example UI.
    #[cfg(feature = "egui")]
    fn ui(&mut self, _ui: &mut egui::Ui, _stats: PerformanceStats) {}

    /// Release resources whose RHI ownership is explicit rather than RAII.
    fn destroy(self, _device: &Device)
    where
        Self: Sized,
    {
    }
}

/// Harness-level command-line options, parsed with clap. Examples with no CLI of their own get
/// these for free via [`run`]; examples that own their CLI flatten this into their clap struct
/// (`#[command(flatten)]`) and hand it to [`run_with`].
///
/// The egui overlay is *not* a runtime option: it is gated entirely by the `egui` build feature
/// (on by default). Build with `--no-default-features` for an egui-free harness.
#[derive(clap::Parser, Debug, Default, Clone)]
pub struct HarnessOpts {
    /// Enable RHI validation layers (also honoured via the `KILN_VALIDATION` env var).
    #[arg(long)]
    pub validation: bool,

    /// Present without display sync.
    ///
    /// This does not uncap the frame rate: the swapchain still only recycles images as the
    /// compositor releases them, so acquisition blocks for what vsync would have waited anyway,
    /// just unevenly. Uncapped rendering needs the headless path.
    #[arg(long)]
    pub no_vsync: bool,

    /// Swapchain images (Vulkan) / drawables (Metal).
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(2..=3))]
    pub image_count: u32,
}

impl HarnessOpts {
    fn present(&self) -> PresentOpts {
        PresentOpts {
            vsync: !self.no_vsync,
            image_count: self.image_count,
        }
    }
}

/// Applied on every (re)creation of the swapchain.
#[derive(Clone, Copy)]
struct PresentOpts {
    vsync: bool,
    image_count: u32,
}

/// Run `E` in an 800×600 window titled `title`, clearing each frame to `clear`, parsing
/// [`HarnessOpts`] from argv. Blocks until the window is closed (or Esc is pressed).
pub fn run<E: Example + 'static>(
    title: &str,
    clear: [f32; 4],
) -> Result<(), Box<dyn std::error::Error>> {
    run_with::<E>(title, clear, HarnessOpts::parse())
}

/// Like [`run`], but with already-parsed [`HarnessOpts`] — for examples that own their CLI and
/// parse argv themselves (a second clap parse inside `run` would reject the example's own flags).
pub fn run_with<E: Example + 'static>(
    title: &str,
    clear: [f32; 4],
    opts: HarnessOpts,
) -> Result<(), Box<dyn std::error::Error>> {
    let validation = opts.validation || std::env::var_os("KILN_VALIDATION").is_some();
    if validation {
        install_stderr_logger();
    }
    let device = Device::new(&DeviceDesc {
        validation,
        label: Some(title.into()),
        ..Default::default()
    })?;

    let event_loop = EventLoop::new()?;
    // Continuous present loop (also drives egui animations).
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::<E> {
        title: title.into(),
        clear,
        present: opts.present(),
        device,
        window: None,
        surface: None,
        swapchain: None,
        depth: None,
        example: None,
        frame_index: 0,
        benchmark_frame_count: 0,
        surface_size: (0, 0),
        #[cfg(feature = "egui")]
        egui: None,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

/// Create a swapchain-sized depth texture in its own GPU-only allocation. The caller keeps both
/// alive together; the texture borrows the allocation's storage.
fn make_depth(device: &Device, format: Format, w: u32, h: u32) -> (Texture, Allocation) {
    let desc = TextureDesc {
        width: w,
        height: h,
        depth: 1,
        mip_levels: 1,
        array_layers: 1,
        format,
        dimension: TextureDimension::D2,
        sample_count: SampleCount::S1,
        usage: TextureUsage::DEPTH_STENCIL_ATTACHMENT,
        label: Some("depth".into()),
    };
    let sa = device.texture_size_align(&desc).expect("depth size_align");
    let mem = device
        .allocate_aligned(sa.size, sa.align, MemoryType::GpuOnly)
        .expect("depth mem");
    let texture = device
        .create_texture(&desc, mem.gpu())
        .expect("create depth texture");
    (texture, mem)
}

struct App<E: Example> {
    title: String,
    clear: [f32; 4],
    // Drop order = field order: device children must drop while `device` is alive, so it is last;
    // surface before window (on Metal it hangs off the window's view), swapchain before surface.
    swapchain: Option<Swapchain>,
    surface: Option<Surface>,
    window: Option<Window>,
    depth: Option<(Texture, Allocation)>,
    example: Option<E>,
    frame_index: usize,
    benchmark_frame_count: u32,
    present: PresentOpts,
    // Tracked to skip the duplicate Resized events macOS streams during a live resize.
    surface_size: (u32, u32),
    #[cfg(feature = "egui")]
    egui: Option<Egui>,
    device: Device,
}

impl<E: Example> App<E> {
    /// Rebuild the swapchain (and depth buffer) at `w`×`h` after draining the GPU.
    fn recreate_surface_sized(&mut self, w: u32, h: u32) {
        let Some(swapchain) = self.swapchain.as_mut() else {
            return;
        };
        self.device.wait_idle();
        self.device
            .recreate_swapchain(
                swapchain,
                &SwapchainDesc {
                    width: w,
                    height: h,
                    vsync: self.present.vsync,
                    image_count: self.present.image_count,
                    ..Default::default()
                },
            )
            .expect("recreate_swapchain");
        self.surface_size = (w, h);
        // Destruction is explicit: dropping a Texture leaves it resident, leaking one per resize.
        let depth_format = self
            .example
            .as_ref()
            .and_then(|example| example.depth_format());
        if let Some((tex, mem)) = self.depth.take() {
            self.device.destroy_texture(tex);
            self.device.free(mem);
        }
        self.depth = depth_format.map(|fmt| make_depth(&self.device, fmt, w, h));
    }

    /// Acquire → record → present one frame for the current `frame_index` slot. The
    /// [`kiln_rhi::frame_scope`] releases the backend's temporaries at the frame boundary.
    fn render_frame(&mut self) {
        kiln_rhi::frame_scope(|| self.render_frame_inner());
    }

    fn render_frame_inner(&mut self) {
        // Build the overlay UI first: its texture deltas must upload before the render pass.
        #[cfg(feature = "egui")]
        let egui_frame = self.build_egui_frame();

        let frame_index = self.frame_index;
        let Some(swapchain) = self.swapchain.as_ref() else {
            return;
        };
        let Some(example) = self.example.as_mut() else {
            return;
        };
        let queue = self.device.queue();

        // `acquire_image` waits on this slot's fence, so the slot's resources are free.
        let acquire_start = Instant::now();
        let image = match queue.acquire_image(swapchain, frame_index) {
            Ok(image) => image,
            Err(e) => {
                // A wedged drawable pool won't recover on its own; drain and rebuild.
                eprintln!("acquire_image failed: {e}; rebuilding swapchain");
                let (w, h) = self.surface_size;
                self.recreate_surface_sized(w.max(1), h.max(1));
                return;
            }
        };
        #[cfg(feature = "egui")]
        if let Some(egui) = self.egui.as_mut() {
            egui.wait_ms = ema(egui.wait_ms, acquire_start.elapsed().as_secs_f64() * 1.0e3);
        }
        let extent = UVec2::new(image.width, image.height);
        let ctx = FrameCtx {
            device: &self.device,
            extent,
            slot: frame_index,
        };
        let cpu_start = Instant::now();

        // Read this slot's prior timestamps (its fence was waited at acquire) before recording over them.
        #[cfg(feature = "egui")]
        if let Some(egui) = self.egui.as_mut()
            && let Ok(Some(ms)) = self
                .device
                .gpu_elapsed_ms(&egui.query_pools[frame_index], 0, 1)
        {
            egui.gpu_ms = ema(egui.gpu_ms, ms);
        }

        let mut cmd = self
            .device
            .create_command_buffer_for_swapchain(swapchain, frame_index)
            .expect("create_command_buffer_for_swapchain");

        // Bracket the frame's GPU work. The pool reset must be outside any render pass.
        #[cfg(feature = "egui")]
        if let Some(egui) = self.egui.as_ref() {
            cmd.reset_queries(&egui.query_pools[frame_index]);
            cmd.write_timestamp(&egui.query_pools[frame_index], 0);
        }

        example.pre_render(&ctx, &mut cmd);
        cmd.begin_render_pass(&RenderPassDesc {
            color_attachments: vec![ColorAttachment {
                target: RenderTarget::SwapchainImage(image.index),
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                clear_color: self.clear,
            }],
            depth_attachment: self.depth.as_ref().map(|(tex, _)| DepthAttachment {
                target: RenderTarget::Texture(tex.id()),
                load_op: LoadOp::Clear,
                store_op: StoreOp::DontCare, // depth is transient; never read back
                clear_depth: 1.0,
                clear_stencil: 0,
            }),
            render_area: [0, 0, extent.x, extent.y],
            label: Some("example"),
        });
        cmd.set_viewport(0.0, 0.0, extent.x as f32, extent.y as f32, 0.0, 1.0);
        cmd.set_scissor(0, 0, extent.x, extent.y);

        example.render(&ctx, &mut cmd);

        cmd.end_render_pass();

        // Overlay in a second color-load pass (no depth) so it composites over a depth example
        // without a dynamic-rendering format mismatch.
        #[cfg(feature = "egui")]
        if let (Some(egui), Some(frame)) = (self.egui.as_mut(), egui_frame.as_ref()) {
            cmd.begin_render_pass(&RenderPassDesc {
                color_attachments: vec![ColorAttachment {
                    target: RenderTarget::SwapchainImage(image.index),
                    load_op: LoadOp::Load,
                    store_op: StoreOp::Store,
                    clear_color: [0.0; 4],
                }],
                depth_attachment: None,
                render_area: [0, 0, extent.x, extent.y],
                label: Some("egui overlay"),
            });
            egui.renderer
                .paint(
                    &self.device,
                    &mut cmd,
                    frame_index,
                    frame.ppp,
                    [extent.x, extent.y],
                    &frame.primitives,
                )
                .expect("egui paint");
            cmd.end_render_pass();
        }

        #[cfg(feature = "egui")]
        if let Some(egui) = self.egui.as_ref() {
            cmd.write_timestamp(&egui.query_pools[frame_index], 1);
        }

        cmd.transition_to_present(image.index);
        cmd.end();

        queue
            .submit_frame(cmd, swapchain, frame_index, image.index)
            .expect("submit_frame");
        self.frame_index = (frame_index + 1) % MAX_FRAMES_IN_FLIGHT;

        #[cfg(feature = "egui")]
        if let Some(egui) = self.egui.as_mut() {
            if let Some(frame) = egui_frame {
                egui.renderer.free_textures(&self.device, &frame.free);
            }
            egui.cpu_ms = ema(egui.cpu_ms, cpu_start.elapsed().as_secs_f64() * 1.0e3);
            if *FRAME_BENCH {
                egui.debug_frame_count += 1;
                if egui.debug_frame_count.is_multiple_of(20) {
                    eprintln!(
                        "kiln frame benchmark: cpu_record={:.3}ms acquire_wait={:.3}ms gpu={:.3}ms",
                        egui.cpu_ms, egui.wait_ms, egui.gpu_ms
                    );
                }
            }
        }

        if *FRAME_BENCH {
            self.benchmark_frame_count += 1;
            if self.benchmark_frame_count.is_multiple_of(20) {
                eprintln!(
                    "kiln frame benchmark: frame={} cpu={:.3}ms",
                    self.benchmark_frame_count,
                    cpu_start.elapsed().as_secs_f64() * 1.0e3
                );
            }
        }
    }

    /// Build this frame's egui UI (the example's overlay), apply its texture deltas,
    /// and return the tessellated geometry to paint. `None` when the overlay is inactive.
    #[cfg(feature = "egui")]
    fn build_egui_frame(&mut self) -> Option<EguiFrame> {
        let egui = self.egui.as_mut()?;
        let window = self.window.as_ref()?;
        let example = self.example.as_mut()?;
        let device = &self.device;

        let raw_input = egui.state.take_egui_input(window);
        // `run_ui`, not the deprecated `run`: only `run_ui` runs egui's plugins, and the
        // text-selection plugin's `on_end_pass` is what releases drag state on pointer-up. Bare
        // `run` skips it, so selection sticks to the cursor after release.
        let ctx = egui.ctx.clone();
        let stats = PerformanceStats {
            cpu_ms: egui.cpu_ms,
            gpu_ms: egui.gpu_ms,
            wait_ms: egui.wait_ms,
            backend: device.backend_name(),
        };
        let out = ctx.run_ui(raw_input, |ui| {
            example.ui(ui, stats);
        });
        egui.state
            .handle_platform_output(window, out.platform_output);
        egui.repaint_at = out
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .map(|viewport| viewport.repaint_delay)
            .filter(|delay| *delay < std::time::Duration::MAX)
            .and_then(|delay| std::time::Instant::now().checked_add(delay));
        let ppp = out.pixels_per_point;
        let primitives = egui.ctx.tessellate(out.shapes, ppp);
        egui.renderer
            .update_textures(device, &out.textures_delta)
            .expect("egui update_textures");
        Some(EguiFrame {
            primitives,
            ppp,
            free: out.textures_delta.free,
        })
    }
}

impl<E: Example> ApplicationHandler for App<E> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return; // already initialized (resumed can fire more than once)
        }

        let mut window_attributes = Window::default_attributes()
            .with_title(&self.title)
            .with_inner_size(LogicalSize::new(800.0, 600.0));
        if *FRAME_BENCH {
            window_attributes = window_attributes.with_resizable(false);
        }
        let window = event_loop
            .create_window(window_attributes)
            .expect("create window");
        let size = window.inner_size();
        let (w, h) = (size.width.max(1), size.height.max(1));

        let window_handle = window.window_handle().expect("window handle").as_raw();
        let display_handle = window.display_handle().expect("display handle").as_raw();

        let surface = self
            .device
            .create_surface(&SurfaceDesc {
                display_handle,
                window_handle,
            })
            .expect("create_surface");
        let swapchain = self
            .device
            .create_swapchain(
                &surface,
                &SwapchainDesc {
                    width: w,
                    height: h,
                    vsync: self.present.vsync,
                    image_count: self.present.image_count,
                    ..Default::default()
                },
            )
            .expect("create_swapchain");
        let example = E::new(&self.device, swapchain.format());
        let depth = example
            .depth_format()
            .map(|fmt| make_depth(&self.device, fmt, w, h));

        // The overlay is gated by the build feature alone: with `egui` on, it is always active.
        #[cfg(feature = "egui")]
        {
            self.egui = Some(Egui::new(&self.device, &window, swapchain.format()));
        }

        self.window = Some(window);
        self.surface = Some(surface);
        self.swapchain = Some(swapchain);
        self.depth = depth;
        self.example = Some(example);
        self.surface_size = (w, h);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Let egui consume the event first (text fields, drags, etc.).
        #[cfg(feature = "egui")]
        if let (Some(egui), Some(window)) = (self.egui.as_mut(), self.window.as_ref()) {
            let _ = egui.state.on_window_event(window, &event);
        }
        if let Some(example) = self.example.as_mut() {
            example.window_event(&event);
        }
        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        logical_key: Key::Named(NamedKey::Escape),
                        ..
                    },
                ..
            } => {
                self.device.wait_idle();
                // Release device-owned storage explicitly (drops are no-ops for RHI handles).
                if let Some((tex, mem)) = self.depth.take() {
                    self.device.destroy_texture(tex);
                    self.device.free(mem);
                }
                if let Some(example) = self.example.take() {
                    example.destroy(&self.device);
                }
                #[cfg(feature = "egui")]
                if let Some(egui) = self.egui.take() {
                    egui.destroy(&self.device);
                }
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                let (w, h) = (size.width.max(1), size.height.max(1));
                if (w, h) != self.surface_size {
                    self.recreate_surface_sized(w, h);
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => self.render_frame(),
            _ => {
                // When the example is idle, ControlFlow::Wait prevents a redraw storm. Every
                // user event still gets one frame so egui edits and camera input remain responsive.
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let continuous = self
            .example
            .as_ref()
            .is_some_and(|example| example.wants_continuous_redraw());
        #[cfg(feature = "egui")]
        let overlay_repaint = self.egui.as_ref().and_then(|egui| egui.repaint_at);
        #[cfg(not(feature = "egui"))]
        let overlay_repaint: Option<std::time::Instant> = None;

        // Redraw now when the example is animating or egui's next pass is already due; otherwise
        // sleep exactly until that pass, so an idle window costs nothing but a pending fade or
        // menu still lands on time.
        let due = overlay_repaint.is_some_and(|at| at <= std::time::Instant::now());
        event_loop.set_control_flow(match overlay_repaint {
            _ if continuous || due => ControlFlow::Poll,
            Some(at) => ControlFlow::WaitUntil(at),
            None => ControlFlow::Wait,
        });
        if (continuous || due)
            && let Some(window) = &self.window
        {
            window.request_redraw();
        }
    }
}

// ---------------------------------------------------------------------------
// egui overlay state. Entirely behind the `egui` feature so the RHI core and egui-free demo builds
// never pull egui/egui-winit/kiln-egui.
// ---------------------------------------------------------------------------

/// Tessellated geometry for one frame's overlay, produced before the render pass and consumed in
/// the overlay's color-load pass.
#[cfg(feature = "egui")]
struct EguiFrame {
    primitives: Vec<egui::ClippedPrimitive>,
    ppp: f32,
    free: Vec<egui::TextureId>,
}

/// Overlay resources: egui context + winit input glue, the Kiln painter, per-slot timestamp pools,
/// and the smoothed frame times supplied to the example UI.
#[cfg(feature = "egui")]
struct Egui {
    ctx: egui::Context,
    state: egui_winit::State,
    renderer: kiln_egui::EguiRenderer,
    /// One 2-slot timestamp pool per frame-in-flight (frame start + end).
    query_pools: Vec<kiln_rhi::QueryPool>,
    /// When egui asked to be run again, or `None` when it is settled.
    ///
    /// egui drives a good deal of its interaction across passes: menus, tooltips, hover fades, and
    /// any two-pass layout resolve on the pass *after* the click that started them. Without
    /// honouring this the loop sleeps in `ControlFlow::Wait` until some unrelated OS event arrives,
    /// and the interface reads as running a frame or more behind the pointer.
    repaint_at: Option<std::time::Instant>,
    cpu_ms: f64,
    gpu_ms: f64,
    wait_ms: f64,
    debug_frame_count: u32,
}

#[cfg(feature = "egui")]
impl Egui {
    fn new(device: &Device, window: &Window, color_format: Format) -> Self {
        let ctx = egui::Context::default();
        let state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let renderer =
            kiln_egui::EguiRenderer::new(device, color_format).expect("EguiRenderer::new");
        let query_pools = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| device.create_query_pool(2).expect("create_query_pool"))
            .collect();
        Self {
            ctx,
            state,
            renderer,
            query_pools,
            repaint_at: None,
            cpu_ms: 0.0,
            gpu_ms: 0.0,
            wait_ms: 0.0,
            debug_frame_count: 0,
        }
    }

    fn destroy(self, device: &Device) {
        self.renderer.destroy(device);
        for pool in self.query_pools {
            device.destroy_query_pool(pool);
        }
    }
}

/// Exponential moving average so the displayed frame times don't flicker. Seeds with the first
/// sample (when the running value is still zero).
#[cfg(feature = "egui")]
fn ema(current: f64, sample: f64) -> f64 {
    if current == 0.0 {
        sample
    } else {
        current * 0.9 + sample * 0.1
    }
}

/// Minimal stderr logger so the RHI's Vulkan validation callback (routed via `log`) is visible.
fn install_stderr_logger() {
    struct StderrLogger;
    impl log::Log for StderrLogger {
        fn enabled(&self, _: &log::Metadata) -> bool {
            true
        }
        fn log(&self, record: &log::Record) {
            eprintln!("[{}] {}", record.level(), record.args());
        }
        fn flush(&self) {}
    }
    static LOGGER: StderrLogger = StderrLogger;
    let _ = log::set_logger(&LOGGER).map(|()| log::set_max_level(log::LevelFilter::Trace));
}
