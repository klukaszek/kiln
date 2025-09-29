//! Unified Kiln example: opens a window and draws a Metal 4 triangle.
//!
//! - On macOS with objc2 AppKit (default), it uses MTKView.
//! - With `--features winit`, it uses winit + CAMetalLayer.

#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

// Use the kiln library's modules instead of redefining them here
// (no direct imports needed here)

#[cfg(all(feature = "winit", target_os = "macos"))]
mod winit_main {
    use kiln::kiln::events;
    use kiln::kiln::windowing::{apply_swapchain_to_metal_layer, request_app_exit};
    use kiln::shared_renderer::{Renderer as SharedRenderer, RenderSurface as SharedRenderSurface, SwapchainConfig};
    use core::cell::RefCell;
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2::{msg_send, ClassType};
    use objc2_core_foundation::CGSize;
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice, MTLPixelFormat, MTLLoadAction, MTLStoreAction};
    use objc2_quartz_core::{CAMetalLayer, CALayer, CAMetalDrawable as _};
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use std::time::Instant;
    use winit::application::ApplicationHandler;
    use winit::event::WindowEvent;
    use winit::event_loop::{ActiveEventLoop, EventLoop};
    use winit::window::{Window, WindowAttributes};

    #[derive(Debug)]
    struct Surface {
        layer: Retained<CAMetalLayer>,
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        pending_drawable: RefCell<Option<Retained<ProtocolObject<dyn objc2_quartz_core::CAMetalDrawable>>>>,
    }

    impl SharedRenderSurface for Surface {
        fn current_mtl4_render_pass_descriptor(&self) -> Option<Retained<objc2_metal::MTL4RenderPassDescriptor>> {
            let Some(drawable) = (unsafe { self.layer.nextDrawable() }) else { return None; };
            self.pending_drawable.replace(Some(drawable.clone()));
            let rp = unsafe { objc2_metal::MTL4RenderPassDescriptor::new() };
            unsafe {
                let ca0 = rp.colorAttachments().objectAtIndexedSubscript(0);
                let tex = drawable.texture();
                ca0.setTexture(Some(&tex));
                ca0.setLoadAction(MTLLoadAction::Clear);
                ca0.setStoreAction(MTLStoreAction::Store);
                ca0.setClearColor(objc2_metal::MTLClearColor { red: 0.1, green: 0.1, blue: 0.12, alpha: 1.0 });
            }
            Some(rp)
        }
        fn current_drawable(&self) -> Option<Retained<ProtocolObject<dyn objc2_quartz_core::CAMetalDrawable>>> {
            self.pending_drawable.borrow_mut().take()
        }
        fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> { self.device.clone() }
        fn color_pixel_format(&self) -> MTLPixelFormat { unsafe { self.layer.pixelFormat() } }
    }

    fn attach_layer_to_nsview(ns_view: *mut objc2::runtime::AnyObject, layer: &CAMetalLayer) {
        unsafe { let _: () = msg_send![ns_view, setWantsLayer: true]; }
        let ca_layer: &CALayer = (&*layer).as_super();
        unsafe { let _: () = msg_send![ns_view, setLayer: Some(ca_layer)]; }
    }
    fn detach_layer_from_nsview(ns_view: *mut objc2::runtime::AnyObject) {
        unsafe { let _: () = msg_send![ns_view, setLayer: Option::<&CALayer>::None]; }
        unsafe { let _: () = msg_send![ns_view, setWantsLayer: false]; }
    }

    struct App {
        window: Option<Window>,
        ns_view: Option<*mut objc2::runtime::AnyObject>,
        surface: Option<Surface>,
        renderer: Option<SharedRenderer>,
        start: Option<Instant>,
        swapchain: SwapchainConfig,
    }

    impl Default for App {
        fn default() -> Self {
            Self { window: None, ns_view: None, surface: None, renderer: None, start: None, swapchain: SwapchainConfig::default() }
        }
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.window.is_some() { return; }
            let attrs = WindowAttributes::default().with_title("kiln triangle (winit)");
            let window = event_loop.create_window(attrs).expect("create window");

            // Resolve NSView from raw handle
            let ns_view: *mut objc2::runtime::AnyObject = {
                let wh = window.window_handle().unwrap();
                let raw = wh.as_raw();
                match raw { RawWindowHandle::AppKit(h) => h.ns_view.as_ptr().cast(), _ => core::ptr::null_mut() }
            };
            assert!(!ns_view.is_null(), "Expected AppKit NSView");

            // Configure CAMetalLayer
            let device = MTLCreateSystemDefaultDevice().expect("no system device");
            let layer = unsafe { CAMetalLayer::layer() };
            let sc = self.swapchain;
            unsafe {
                layer.setDevice(Some(&device));
                let size = window.inner_size();
                apply_swapchain_to_metal_layer(&layer, size.width as f64, size.height as f64, &sc);
            }
            attach_layer_to_nsview(ns_view, &layer);

            let surface = Surface { layer: layer.clone(), device: device.clone(), pending_drawable: RefCell::new(None) };
            let renderer = SharedRenderer::new(&surface, sc);

            self.start = Some(Instant::now());
            self.renderer = Some(renderer);
            self.surface = Some(surface);
            self.ns_view = Some(ns_view);
            self.window = Some(window);
        }

        fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: winit::window::WindowId, event: WindowEvent) {
            let Some(window) = self.window.as_ref() else { return; };
            if window.id() != window_id { return; }
            match event {
                WindowEvent::CloseRequested => {
                    if let Some(view) = self.ns_view.take() { detach_layer_from_nsview(view); }
                    request_app_exit(event_loop);
                }
                WindowEvent::Resized(size) => {
                    if let Some(surface) = self.surface.as_ref() {
                        unsafe { surface.layer.setDrawableSize(CGSize { width: size.width as f64, height: size.height as f64 }); }
                    }
                    window.request_redraw();
                }
                WindowEvent::RedrawRequested => {
                    if let (Some(renderer), Some(surface), Some(start)) = (self.renderer.as_ref(), self.surface.as_ref(), self.start.as_ref()) {
                        let t = -start.elapsed().as_secs_f32();
                        renderer.draw_frame(surface, t);
                    }
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    let mapped = events::from_winit_mouse_input(state, button);
                    if let events::AppEvent::MouseInput { button, state, .. } = mapped {
                        println!("mouse click: {:?} {:?}", button, state);
                    }
                }
                _ => {}
            }
        }

        fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
            if let Some(window) = self.window.as_ref() { window.request_redraw(); }
        }
    }

    pub fn run() {
        let event_loop = EventLoop::new().expect("create event loop");
        let mut app = App::default();
        let _ = event_loop.run_app(&mut app);
    }
}

#[cfg(all(feature = "winit", target_os = "macos"))]
fn main() { winit_main::run(); }

// ---------------- ObjC2 (AppKit + MTKView) path ----------------

#[cfg(all(not(feature = "winit"), target_os = "macos"))]
mod objc2_main {
    // Use no Kiln window helpers here; inline the exact AppKit flow
    use objc2::{define_class, msg_send, DefinedClass, MainThreadMarker, MainThreadOnly, ClassType};
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::{ns_string, NSDate, NSNotification, NSObject, NSObjectProtocol, NSPoint, NSRect, NSSize};
    use objc2_metal::MTL4RenderCommandEncoder as _; // trait methods
    use objc2_metal::MTL4CommandEncoder; // for endEncoding
    use objc2_metal::{
        MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4BlendState, MTL4CommandAllocator,
        MTL4CommandBuffer, MTL4CommandQueue, MTL4Compiler, MTL4CompilerDescriptor,
        MTL4FunctionDescriptor, MTL4LibraryDescriptor, MTL4LibraryFunctionDescriptor,
        MTL4RenderPipelineDescriptor,
        MTLBuffer, MTLCreateSystemDefaultDevice, MTLDevice as _, MTLDrawable, MTLLibrary,
        MTLPrimitiveType, MTLRenderPipelineState, MTLRenderStages, MTLResourceOptions,
        MTLClearColor, MTLLoadAction,
    };
    use objc2_app_kit::{NSApplication, NSApplicationActivationPolicy, NSApplicationDelegate, NSBackingStoreType, NSWindow, NSWindowStyleMask};
    use objc2_metal_kit::{MTKView, MTKViewDelegate};
    use core::cell::OnceCell;
    use core::cell::RefCell;

    // Kiln unified event types (use library path from example)
    use kiln::kiln::events::{AppEvent, EventQueue, ElementState, MouseButton};

    // A minimal MTKView subclass that records AppKit mouse events
    // into an EventQueue for parity with the winit backend.
    struct ViewIvars { queue: RefCell<EventQueue> }

    define_class!(
        #[unsafe(super(MTKView))]
        #[thread_kind = MainThreadOnly]
        #[ivars = ViewIvars]
        struct InputView;
        impl InputView {
            // Left button down
            #[unsafe(method(mouseDown:))]
            fn mouseDown(&self, event: &objc2_app_kit::NSEvent) {
                let loc_win = unsafe { event.locationInWindow() };
                let loc = (&*self).as_super().convertPoint_fromView(loc_win, None);
                let mut q = self.ivars().queue.borrow_mut();
                q.push(AppEvent::MouseInput {
                    button: MouseButton::Left,
                    state: ElementState::Pressed,
                    x: loc.x as f64,
                    y: loc.y as f64,
                });
            }

            // Left button up
            #[unsafe(method(mouseUp:))]
            fn mouseUp(&self, event: &objc2_app_kit::NSEvent) {
                let loc_win = unsafe { event.locationInWindow() };
                let loc = (&*self).as_super().convertPoint_fromView(loc_win, None);
                let mut q = self.ivars().queue.borrow_mut();
                q.push(AppEvent::MouseInput {
                    button: MouseButton::Left,
                    state: ElementState::Released,
                    x: loc.x as f64,
                    y: loc.y as f64,
                });
            }

            // Right button down
            #[unsafe(method(rightMouseDown:))]
            fn rightMouseDown(&self, event: &objc2_app_kit::NSEvent) {
                let loc_win = unsafe { event.locationInWindow() };
                let loc = (&*self).as_super().convertPoint_fromView(loc_win, None);
                let mut q = self.ivars().queue.borrow_mut();
                q.push(AppEvent::MouseInput {
                    button: MouseButton::Right,
                    state: ElementState::Pressed,
                    x: loc.x as f64,
                    y: loc.y as f64,
                });
            }

            // Right button up
            #[unsafe(method(rightMouseUp:))]
            fn rightMouseUp(&self, event: &objc2_app_kit::NSEvent) {
                let loc_win = unsafe { event.locationInWindow() };
                let loc = (&*self).as_super().convertPoint_fromView(loc_win, None);
                let mut q = self.ivars().queue.borrow_mut();
                q.push(AppEvent::MouseInput {
                    button: MouseButton::Right,
                    state: ElementState::Released,
                    x: loc.x as f64,
                    y: loc.y as f64,
                });
            }

            // Other button down (e.g. middle/extra buttons)
            #[unsafe(method(otherMouseDown:))]
            fn otherMouseDown(&self, event: &objc2_app_kit::NSEvent) {
                let loc_win = unsafe { event.locationInWindow() };
                let loc = (&*self).as_super().convertPoint_fromView(loc_win, None);
                let btn = unsafe { event.buttonNumber() } as u16;
                let mapped = if btn == 2 { MouseButton::Middle } else { MouseButton::Other(btn) };
                let mut q = self.ivars().queue.borrow_mut();
                q.push(AppEvent::MouseInput {
                    button: mapped,
                    state: ElementState::Pressed,
                    x: loc.x as f64,
                    y: loc.y as f64,
                });
            }

            // Other button up (e.g. middle/extra buttons)
            #[unsafe(method(otherMouseUp:))]
            fn otherMouseUp(&self, event: &objc2_app_kit::NSEvent) {
                let loc_win = unsafe { event.locationInWindow() };
                let loc = (&*self).as_super().convertPoint_fromView(loc_win, None);
                let btn = unsafe { event.buttonNumber() } as u16;
                let mapped = if btn == 2 { MouseButton::Middle } else { MouseButton::Other(btn) };
                let mut q = self.ivars().queue.borrow_mut();
                q.push(AppEvent::MouseInput {
                    button: mapped,
                    state: ElementState::Released,
                    x: loc.x as f64,
                    y: loc.y as f64,
                });
            }
        }
    );

    // AppKit click forwarding handled by InputView to match winit path.

    // Zerocopy-friendly CPU-side types matching MSL layout (exactly like metal4_triangle)
    #[derive(Copy, Clone)]
    #[repr(C)]
    struct PackedFloat3 { x: f32, y: f32, z: f32 }
    impl PackedFloat3 { const fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } } }

    #[derive(Copy, Clone)]
    #[repr(C)]
    struct SceneProperties { time: f32 }

    #[derive(Copy, Clone)]
    #[repr(C)]
    struct VertexInput { position: PackedFloat3, color: PackedFloat3 }
    struct Ivars {
        start_date: Retained<NSDate>,
        device: OnceCell<Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>>,
        command_queue: OnceCell<Retained<ProtocolObject<dyn MTL4CommandQueue>>>,
        command_allocator: OnceCell<Retained<ProtocolObject<dyn MTL4CommandAllocator>>>,
        pipeline_state: OnceCell<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
        argument_table: OnceCell<Retained<ProtocolObject<dyn MTL4ArgumentTable>>>,
        scene_buffer: OnceCell<Retained<ProtocolObject<dyn MTLBuffer>>>,
        vertex_buffer: OnceCell<Retained<ProtocolObject<dyn MTLBuffer>>>,
        window: OnceCell<Retained<NSWindow>>,
        view: OnceCell<Retained<InputView>>,
    }

    macro_rules! idcell_set { ($name:ident, $this:expr, $value:expr) => {{ let _ = $this.ivars().$name.set($value); }} }

    define_class!(
        #[unsafe(super(NSObject))]
        #[thread_kind = MainThreadOnly]
        #[ivars = Ivars]
        struct Delegate;

        unsafe impl NSObjectProtocol for Delegate {}
        unsafe impl NSApplicationDelegate for Delegate {
            #[unsafe(method(applicationDidFinishLaunching:))]
            unsafe fn applicationDidFinishLaunching(&self, _n: &NSNotification) {
                let mtm = self.mtm();
                // Create window (exactly like metal4_triangle)
            let window = {
                let content_rect = NSRect::new(NSPoint::new(0., 0.), NSSize::new(768., 768.));
                let style = NSWindowStyleMask::Closable | NSWindowStyleMask::Resizable | NSWindowStyleMask::Titled;
                let backing_store_type = NSBackingStoreType::Buffered;
                let flag = false;
                unsafe { NSWindow::initWithContentRect_styleMask_backing_defer(NSWindow::alloc(mtm), content_rect, style, backing_store_type, flag) }
            };

                // Device and queue
                let device = MTLCreateSystemDefaultDevice().expect("failed to get default system device");
                let command_queue = unsafe { device.newMTL4CommandQueue().expect("create queue") };

                // MTKView (subclassed for input capturing)
                let view: Retained<InputView> = {
                    let this = InputView::alloc(mtm);
                    let this = this.set_ivars(ViewIvars { queue: RefCell::new(EventQueue::new()) });
                    // Call MTKView's designated initializer on super
                    unsafe { msg_send![super(this), initWithFrame: window.frame(), device: Some(&*device)] }
                };
                // Match winit path clear color
                unsafe { (&*view).as_super().setClearColor(MTLClearColor { red: 0.1, green: 0.1, blue: 0.12, alpha: 1.0 }); }

                // Compiler and library from inline MSL
                let compiler_desc = unsafe { MTL4CompilerDescriptor::new() };
                let compiler = unsafe { device.newCompilerWithDescriptor_error(&compiler_desc).expect("create compiler") };
                let lib_desc = unsafe { MTL4LibraryDescriptor::new() };
                unsafe {
                    // Use local copy of shader in examples directory to avoid external path dependency
                    lib_desc.setSource(Some(ns_string!(include_str!("metal4_triangle.metal"))));
                    lib_desc.setName(Some(ns_string!("kiln_triangle_lib")));
                }
                let library: Retained<ProtocolObject<dyn MTLLibrary>> = unsafe { compiler.newLibraryWithDescriptor_error(&lib_desc).expect("Failed to create MTLLibrary with MTL4Compiler") };

                // Function descriptors
                let vfd = unsafe { MTL4LibraryFunctionDescriptor::new() };
                unsafe { vfd.setName(Some(ns_string!("vertex_main"))); vfd.setLibrary(Some(&library)); }
                let ffd = unsafe { MTL4LibraryFunctionDescriptor::new() };
                unsafe { ffd.setName(Some(ns_string!("fragment_main"))); ffd.setLibrary(Some(&library)); }

                // Render pipeline descriptor
                let rp_desc = unsafe { MTL4RenderPipelineDescriptor::new() };
                let vfd_base: &MTL4FunctionDescriptor = (&*vfd).as_super();
                let ffd_base: &MTL4FunctionDescriptor = (&*ffd).as_super();
                unsafe {
                    rp_desc.setVertexFunctionDescriptor(Some(vfd_base));
                    rp_desc.setFragmentFunctionDescriptor(Some(ffd_base));
                    let ca0 = rp_desc.colorAttachments().objectAtIndexedSubscript(0);
                    ca0.setPixelFormat(view.colorPixelFormat());
                    ca0.setBlendingState(MTL4BlendState::Enabled);
                }
                let pipeline_state = unsafe { compiler.newRenderPipelineStateWithDescriptor_compilerTaskOptions_error(&rp_desc, None).expect("create pipeline") };

                // Argument table
                let at_desc = unsafe { MTL4ArgumentTableDescriptor::new() };
                unsafe { at_desc.setMaxBufferBindCount(2); at_desc.setMaxTextureBindCount(0); }
                let argument_table = unsafe { device.newArgumentTableWithDescriptor_error(&at_desc).expect("create arg table") };

                // Buffers
                let scene_buf_len = core::mem::size_of::<SceneProperties>();
                let scene_buffer = device.newBufferWithLength_options(scene_buf_len, MTLResourceOptions::CPUCacheModeDefaultCache).expect("scene buf");

                let verts: [VertexInput; 3] = [
                    VertexInput { position: PackedFloat3::new(-f32::sqrt(3.0) / 4.0, -0.25, 0.0), color: PackedFloat3::new(1.0, 0.0, 0.0) },
                    VertexInput { position: PackedFloat3::new( f32::sqrt(3.0) / 4.0, -0.25, 0.0), color: PackedFloat3::new(0.0, 1.0, 0.0) },
                    VertexInput { position: PackedFloat3::new( 0.0, 0.5, 0.0),                    color: PackedFloat3::new(0.0, 0.0, 1.0) },
                ];
                let verts_len = core::mem::size_of_val(&verts);
                let verts_ptr = core::ptr::NonNull::new(verts.as_ptr() as *mut core::ffi::c_void).unwrap();
                let vertex_buffer = unsafe { device.newBufferWithBytes_length_options(verts_ptr, verts_len, MTLResourceOptions::CPUCacheModeDefaultCache).expect("vbuf") };

                // Command allocator
                let command_allocator = unsafe { device.newCommandAllocator().expect("allocator") };

                // Wire up the MTKView
                let obj = ProtocolObject::from_ref(self);
                unsafe { view.setDelegate(Some(obj)); }
                // NSWindow expects an NSView; convert InputView -> MTKView -> NSView
                window.setContentView(Some((&*view).as_super().as_super()));
                window.center();
                window.setTitle(ns_string!("kiln triangle (objc2)"));
                window.makeKeyAndOrderFront(None);

            // Bring app to foreground (avoid minimized/background start)
            let app = NSApplication::sharedApplication(mtm);
            app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
            #[allow(deprecated)]
            app.activateIgnoringOtherApps(true);

                // Initialize ivars
                idcell_set!(device, self, device);
                idcell_set!(command_queue, self, command_queue);
                idcell_set!(command_allocator, self, command_allocator);
                idcell_set!(pipeline_state, self, pipeline_state);
                idcell_set!(argument_table, self, argument_table);
                idcell_set!(scene_buffer, self, scene_buffer);
                idcell_set!(vertex_buffer, self, vertex_buffer);
                idcell_set!(window, self, window);
                idcell_set!(view, self, view);
        }
        // Ensure closing the window quits the app (like winit)
        #[unsafe(method(applicationShouldTerminateAfterLastWindowClosed:))]
        unsafe fn applicationShouldTerminateAfterLastWindowClosed(&self, _app: &NSApplication) -> bool { true }
    }

    // No NSWindowDelegate/termination hooks: mirror metal4_triangle lifecycle exactly.

        unsafe impl MTKViewDelegate for Delegate {
            #[unsafe(method(drawInMTKView:))]
            unsafe fn drawInMTKView(&self, view: &MTKView) {
                // Drain any queued AppKit mouse events for parity with winit's WindowEvent::MouseInput path
                if let Some(v) = self.ivars().view.get() {
                    v.ivars().queue.borrow_mut().drain(|ev| {
                        if let AppEvent::MouseInput { button, state, .. } = ev {
                            println!("mouse click: {:?} {:?}", button, state);
                        }
                    });
                }

                let command_queue = self.ivars().command_queue.get().unwrap();
                let command_allocator = self.ivars().command_allocator.get().unwrap();
                let pipeline_state = self.ivars().pipeline_state.get().unwrap();
                let argument_table = self.ivars().argument_table.get().unwrap();
                let scene_buffer = self.ivars().scene_buffer.get().unwrap();
                let vertex_buffer = self.ivars().vertex_buffer.get().unwrap();

                // Update scene properties
                let scene = SceneProperties { time: self.ivars().start_date.timeIntervalSinceNow() as f32 };
                let dst = scene_buffer.contents();
                let src_ptr = &scene as *const SceneProperties as *const u8;
                unsafe { core::ptr::copy_nonoverlapping(src_ptr, dst.as_ptr().cast::<u8>(), core::mem::size_of::<SceneProperties>()); }

                // Bind argument table
                unsafe { argument_table.setAddress_atIndex(scene_buffer.gpuAddress(), 0); }
                unsafe { argument_table.setAddress_atIndex(vertex_buffer.gpuAddress(), 1); }

                // Acquire drawable and pass descriptors
                let Some(drawable) = (unsafe { view.currentDrawable() }) else { return; };
                let Some(rp) = (unsafe { view.currentMTL4RenderPassDescriptor() }) else { return; };
                // Ensure load action + clear color match the winit path
                unsafe {
                    let ca0 = rp.colorAttachments().objectAtIndexedSubscript(0);
                    ca0.setLoadAction(MTLLoadAction::Clear);
                    ca0.setClearColor(MTLClearColor { red: 0.1, green: 0.1, blue: 0.12, alpha: 1.0 });
                }

                // Prepare allocator and command buffer
                unsafe { command_allocator.reset(); }
                let device = self.ivars().device.get().unwrap();
                let Some(cmd) = (unsafe { device.newCommandBuffer() }) else { return; };
                unsafe { cmd.beginCommandBufferWithAllocator(&command_allocator); }

                // Encode render pass
                let Some(enc) = (unsafe { cmd.renderCommandEncoderWithDescriptor(&rp) }) else { return; };
                unsafe { enc.setRenderPipelineState(pipeline_state); }
                unsafe { enc.setArgumentTable_atStages(argument_table, MTLRenderStages::Vertex); }
                unsafe { enc.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3); }
                unsafe { enc.endEncoding(); }
                unsafe { cmd.endCommandBuffer(); }

                // Submit via Metal 4 queue flow
                unsafe { command_queue.waitForDrawable(ProtocolObject::from_ref(&*drawable)); }
                let mut arr = [core::ptr::NonNull::from(&*cmd)];
                let ptr = unsafe { core::ptr::NonNull::new_unchecked(arr.as_mut_ptr()) };
                unsafe { command_queue.commit_count(ptr, 1); }
                unsafe { command_queue.signalDrawable(ProtocolObject::from_ref(&*drawable)); }
                drawable.present();
            }

            #[unsafe(method(mtkView:drawableSizeWillChange:))]
            unsafe fn mtkView_drawableSizeWillChange(&self, _view: &MTKView, _size: NSSize) {
                // no-op, parity with metal4_triangle
            }
        }

        // If we want AppKit mouse events, add a NSView subclass or install a local monitor.
    );

    pub fn run() {
        let mtm = MainThreadMarker::new().unwrap();
        let app = NSApplication::sharedApplication(mtm);
        app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
        let delegate: Retained<Delegate> = {
            let this = Delegate::alloc(mtm);
            let this = this.set_ivars(Ivars {
                start_date: NSDate::now(),
                device: OnceCell::default(),
                command_queue: OnceCell::default(),
                command_allocator: OnceCell::default(),
                pipeline_state: OnceCell::default(),
                argument_table: OnceCell::default(),
                scene_buffer: OnceCell::default(),
                vertex_buffer: OnceCell::default(),
                window: OnceCell::default(),
                view: OnceCell::default(),
            });
            unsafe { msg_send![super(this), init] }
        };
        let object = ProtocolObject::from_ref(&*delegate);
        app.setDelegate(Some(object));
        app.run();
    }
}

#[cfg(all(not(feature = "winit"), target_os = "macos"))]
fn main() { objc2_main::run(); }

#[cfg(not(target_os = "macos"))]
fn main() { println!("Kiln example currently supported on macOS only."); }
