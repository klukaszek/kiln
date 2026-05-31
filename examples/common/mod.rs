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

use kiln_rhi::{
    ColorAttachment, CommandBuffer, Device, DeviceDesc, Format, LoadOp, RenderPassDesc,
    RenderTarget, ShaderModule, ShaderModuleDesc, ShaderStage, StoreOp, Surface, SurfaceDesc,
    Swapchain, SwapchainDesc, MAX_FRAMES_IN_FLIGHT,
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

/// What each windowed example implements: build its pipelines once, then record
/// draw commands into the per-frame render pass.
pub trait Example {
    /// Build pipelines/resources. `color_format` is the swapchain's colour format —
    /// any PSO colour target must match it.
    fn new(device: &Device, color_format: Format) -> Self
    where
        Self: Sized;

    /// Record draws for one frame. The render pass is already begun on the acquired
    /// swapchain image (cleared) with viewport + scissor set to the full `extent`.
    fn render(&mut self, cmd: &mut CommandBuffer, extent: [u32; 2]);
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
    // Poll + request_redraw drives a continuous present loop; vsync throttles it.
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::<E> {
        title: title.into(),
        clear,
        device,
        window: None,
        surface: None,
        swapchain: None,
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
        let extent = [image.width, image.height];

        let mut cmd = self
            .device
            .create_command_buffer_for_swapchain(swapchain)
            .expect("create_command_buffer_for_swapchain");
        cmd.begin_render_pass(&RenderPassDesc {
            color_attachments: vec![ColorAttachment {
                target: RenderTarget::SwapchainImage(image.index),
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                clear_color: self.clear,
            }],
            depth_attachment: None,
            render_area: [0, 0, extent[0], extent[1]],
        });
        cmd.set_viewport(0.0, 0.0, extent[0] as f32, extent[1] as f32, 0.0, 1.0);
        cmd.set_scissor(0, 0, extent[0], extent[1]);

        example.render(&mut cmd, extent);

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

        let window_handle = window
            .window_handle()
            .expect("window handle")
            .as_raw();
        let display_handle = window
            .display_handle()
            .expect("display handle")
            .as_raw();

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
                    vsync: true,
                    ..Default::default()
                },
            )
            .expect("create_swapchain");
        let example = E::new(&self.device, swapchain.format());

        self.window = Some(window);
        self.surface = Some(surface);
        self.swapchain = Some(swapchain);
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
                                vsync: true,
                                ..Default::default()
                            },
                        )
                        .expect("recreate_swapchain");
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
pub fn compile(
    device: &Device,
    slang_src: &str,
    entry: &str,
    stage: ShaderStage,
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

    let output = Command::new("slangc")
        .arg(&src_path)
        .args(["-target", target, "-entry", entry, "-stage", slang_stage])
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
