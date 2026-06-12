//! Shared windowing harness for the Kiln examples.
//!
//! The headless integration tests render into an offscreen texture and read it back;
//! these examples render the same shaders into a real window via the RHI's surface +
//! swapchain. The harness owns the winit event loop, the device, and the per-frame
//! present loop; each example just builds its pipelines and records its draws.
//!
//! Each example pulls in this whole module but uses only part of it, so silence the
//! per-binary "unused helper" warnings here rather than at every call site.
#![allow(dead_code)]

use std::process::Command;

use glam::UVec2;

use kiln_rhi::{
    ColorAttachment, CommandBuffer, DepthAttachment, Device, DeviceDesc, Format, GpuAllocation,
    LoadOp, MAX_FRAMES_IN_FLIGHT, MemoryType, RenderPassDesc, RenderTarget, SampleCount,
    ShaderModule, ShaderModuleDesc, ShaderStage, StoreOp, Surface, SurfaceDesc, Swapchain,
    SwapchainDesc, Texture, TextureDesc, TextureDimension, TextureUsage,
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

/// Per-frame recording context: the device, the target extent, and which of the
/// [`MAX_FRAMES_IN_FLIGHT`] slots this frame occupies.
///
/// `slot` is the realtime invariant: when this frame records, up to
/// `MAX_FRAMES_IN_FLIGHT - 1` earlier frames may still execute on the GPU, so any
/// CPU-written transient (root structs, bump arenas) must be keyed by `slot` —
/// the harness only guarantees that *this slot's* previous frame has retired.
pub struct FrameCtx<'a> {
    pub device: &'a Device,
    pub extent: UVec2,
    pub slot: usize,
}

/// What each windowed example implements: build its pipelines once, then record
/// draw commands into the per-frame render pass.
pub trait Example {
    /// Build pipelines/resources. `color_format` is the swapchain's colour format —
    /// any PSO colour target must match it.
    fn new(device: &Device, color_format: Format) -> Self
    where
        Self: Sized;

    /// Opt into a depth buffer. When `Some(format)`, the harness creates a
    /// swapchain-sized depth texture (recreated on resize) and binds it cleared to 1.0
    /// for every render pass. Default `None` keeps the simple no-depth path.
    fn depth_format() -> Option<Format>
    where
        Self: Sized,
    {
        None
    }

    /// Record work that must happen before the swapchain render pass, such as compute
    /// accumulation for progressive renderers. Default examples do nothing here.
    fn pre_render(&mut self, _ctx: &FrameCtx, _cmd: &mut CommandBuffer) {}

    /// Record draws for one frame. The render pass is already begun on the acquired
    /// swapchain image (cleared) with viewport + scissor set to the full extent. When
    /// [`Self::depth_format`] is `Some`, a cleared depth attachment is bound too.
    fn render(&mut self, ctx: &FrameCtx, cmd: &mut CommandBuffer);
}

/// Create a swapchain-sized depth texture in its own GPU-only allocation. The caller
/// keeps both alive together; the texture borrows the allocation's storage.
fn make_depth(device: &Device, format: Format, w: u32, h: u32) -> (Texture, GpuAllocation) {
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
        .malloc_aligned(sa.size, sa.align, MemoryType::GpuOnly)
        .expect("depth mem");
    let texture = device
        .create_texture(&desc, mem.gpu())
        .expect("create depth texture");
    (texture, mem)
}

/// Run `E` in an 800×600 window titled `title`, clearing each frame to `clear`.
/// Blocks until the window is closed (or Esc is pressed).
pub fn run<E: Example + 'static>(
    title: &str,
    clear: [f32; 4],
) -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new(&DeviceDesc {
        validation: false,
        label: Some(title.into()),
        ..Default::default()
    })?;

    let event_loop = EventLoop::new()?;
    // Poll + request_redraw drives a continuous present loop.
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::<E> {
        title: title.into(),
        clear,
        device,
        window: None,
        surface: None,
        swapchain: None,
        depth: None,
        example: None,
        frame_index: 0,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

struct App<E: Example> {
    title: String,
    clear: [f32; 4],
    device: Device,
    // `window` must outlive `surface`: on Metal the surface is a CAMetalLayer hung off
    // the window's view. Declared first only for readability — drop order is by the
    // explicit teardown in `wait_idle` on exit.
    window: Option<Window>,
    surface: Option<Surface>,
    swapchain: Option<Swapchain>,
    // Harness-owned depth buffer (texture + its backing allocation), present only when
    // the example opts in via `E::depth_format()`. Recreated on resize.
    depth: Option<(Texture, GpuAllocation)>,
    example: Option<E>,
    frame_index: usize,
}

impl<E: Example> App<E> {
    /// Acquire → record → present one frame for the current `frame_index` slot.
    fn render_frame(&mut self) {
        let frame_index = self.frame_index;
        let Some(swapchain) = self.swapchain.as_ref() else {
            return;
        };
        let Some(example) = self.example.as_mut() else {
            return;
        };
        let queue = self.device.queue();

        // `acquire_image` waits on this slot's fence, so the slot's resources are free.
        let image = match queue.acquire_image(swapchain, frame_index) {
            Ok(image) => image,
            Err(e) => {
                eprintln!("acquire_image failed: {e}");
                return;
            }
        };
        let extent = UVec2::new(image.width, image.height);
        let ctx = FrameCtx {
            device: &self.device,
            extent,
            slot: frame_index,
        };

        let mut cmd = self
            .device
            .create_command_buffer_for_swapchain(swapchain)
            .expect("create_command_buffer_for_swapchain");
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
        });
        cmd.set_viewport(0.0, 0.0, extent.x as f32, extent.y as f32, 0.0, 1.0);
        cmd.set_scissor(0, 0, extent.x, extent.y);

        example.render(&ctx, &mut cmd);

        cmd.end_render_pass();
        cmd.transition_to_present(image.index);
        cmd.end();

        queue
            .submit_frame(cmd, swapchain, frame_index, image.index)
            .expect("submit_frame");
        self.frame_index = (frame_index + 1) % MAX_FRAMES_IN_FLIGHT;
    }
}

impl<E: Example> ApplicationHandler for App<E> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return; // already initialized (resumed can fire more than once)
        }

        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title(&self.title)
                    .with_inner_size(LogicalSize::new(800.0, 600.0)),
            )
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
                    ..Default::default()
                },
            )
            .expect("create_swapchain");
        let example = E::new(&self.device, swapchain.format());
        let depth = E::depth_format().map(|fmt| make_depth(&self.device, fmt, w, h));

        self.window = Some(window);
        self.surface = Some(surface);
        self.swapchain = Some(swapchain);
        self.depth = depth;
        self.example = Some(example);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
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
                // Release the depth allocation explicitly (the texture only borrows it).
                if let Some((tex, mem)) = self.depth.take() {
                    drop(tex);
                    self.device.free(mem);
                }
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(swapchain) = self.swapchain.as_mut() {
                    let (w, h) = (size.width.max(1), size.height.max(1));
                    self.device.wait_idle();
                    self.device
                        .recreate_swapchain(
                            swapchain,
                            &SwapchainDesc {
                                width: w,
                                height: h,
                                ..Default::default()
                            },
                        )
                        .expect("recreate_swapchain");
                    // Depth buffer must track the swapchain size: drop the old one and
                    // build a fresh match (the example opted in, so the slot stays Some).
                    if let Some((tex, mem)) = self.depth.take() {
                        drop(tex);
                        self.device.free(mem);
                        self.depth =
                            Some(make_depth(&self.device, E::depth_format().unwrap(), w, h));
                    }
                }
            }
            WindowEvent::RedrawRequested => self.render_frame(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

// ---------------------------------------------------------------------------
// Slang compilation (mirrors tests/common): one Slang source per example, lowered
// to the active backend's format and registered as a module. Examples are run
// interactively, so a missing `slangc` or a compile error is a hard error here
// (the headless tests *skip* instead).
// ---------------------------------------------------------------------------

/// Compile a Slang entry point to the active backend's shader format and register it.
/// Panics with a clear message if `slangc` is missing or compilation fails.
pub fn compile(device: &Device, slang_src: &str, entry: &str, stage: ShaderStage) -> ShaderModule {
    compile_with_caps(device, slang_src, entry, stage, &[])
}

/// Compile a Slang entry point with explicit Slang capabilities.
pub fn compile_with_caps(
    device: &Device,
    slang_src: &str,
    entry: &str,
    stage: ShaderStage,
    capabilities: &[&str],
) -> ShaderModule {
    let (target, ext) = match device.backend_name() {
        "Vulkan" => ("spirv", "spv"),
        "Metal" => ("metallib", "metallib"),
        other => panic!("compile: unsupported backend {other}"),
    };
    let slang_stage = match stage {
        ShaderStage::Compute => "compute",
        ShaderStage::Vertex => "vertex",
        ShaderStage::Pixel => "fragment",
        ShaderStage::Mesh => "mesh",
    };

    let dir = std::env::temp_dir();
    let pid = std::process::id();
    let src_path = dir.join(format!("kiln_example_{pid}_{entry}.slang"));
    let out_path = dir.join(format!("kiln_example_{pid}_{entry}.{ext}"));
    std::fs::write(&src_path, slang_src).expect("write slang source");

    let mut cmd = Command::new("slangc");
    cmd.arg(&src_path)
        .args(["-target", target, "-entry", entry, "-stage", slang_stage]);
    for cap in capabilities {
        cmd.args(["-capability", cap]);
    }
    let output = cmd
        .arg("-o")
        .arg(&out_path)
        .output()
        .expect("failed to run slangc — is it on your PATH?");
    if !output.status.success() {
        let _ = std::fs::remove_file(&src_path);
        panic!(
            "slangc failed compiling entry `{entry}` for {target}:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let code = std::fs::read(&out_path).expect("read compiled shader");
    let _ = std::fs::remove_file(&src_path);
    let _ = std::fs::remove_file(&out_path);

    device
        .create_shader_module(&ShaderModuleDesc {
            code: &code,
            entry_point: entry,
            stage,
            label: Some("slang"),
        })
        .expect("create_shader_module")
}
