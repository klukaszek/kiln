//! Runtime-facing API surface only. Example code lives in examples/.
#![allow(clippy::too_many_arguments)]

// Re-export types needed to implement a runtime outside the library.
pub use crate::kiln::events::{
    self, AppEvent, ElementState, EventQueue, Modifiers, MouseButton, TouchPhase,
};
pub use crate::kiln::renderer::{self, Renderer};
pub use crate::kiln::swapchain::{self, ColorSpace, PresentMode, RenderSurface, SwapchainConfig};
pub use crate::kiln::windowing;

use crate::kiln;

pub struct RunConfig<'a> {
    pub title: &'a str,
}
impl<'a> RunConfig<'a> {
    pub fn new(title: &'a str) -> Self {
        Self { title }
    }
}

type DrawFn = Box<dyn FnMut(&dyn kiln::swapchain::RenderSurface, f32) + 'static>;

// High-level lifecycle hooks for a stateful example
pub trait KilnApp {
    fn title(&self) -> &str {
        "kiln"
    }
    fn init(&mut self, _surface: &dyn RenderSurface) {}
    fn update(&mut self, _dt: f32) {}
    fn draw(&mut self, surface: &dyn RenderSurface, t: f32);
    fn quit(&mut self) {}
}

// ---------------- Winit backend (macOS) ----------------
#[cfg(all(feature = "winit", target_os = "macos"))]
pub fn run(config: RunConfig, draw: DrawFn) {
    use core::cell::RefCell;
    use kiln::windowing::{apply_swapchain_to_metal_layer, request_app_exit};
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2::{msg_send, ClassType};
    use objc2_core_foundation::CGSize;
    use objc2_metal::{
        MTLCreateSystemDefaultDevice, MTLDevice, MTLLoadAction, MTLPixelFormat, MTLStoreAction,
    };
    use objc2_quartz_core::{CALayer, CAMetalDrawable as _, CAMetalLayer};
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
        pending_drawable:
            RefCell<Option<Retained<ProtocolObject<dyn objc2_quartz_core::CAMetalDrawable>>>>,
    }
    impl kiln::renderer::RenderSurface for Surface {
        fn current_mtl4_render_pass_descriptor(
            &self,
        ) -> Option<Retained<objc2_metal::MTL4RenderPassDescriptor>> {
            let Some(drawable) = (unsafe { self.layer.nextDrawable() }) else {
                return None;
            };
            self.pending_drawable.replace(Some(drawable.clone()));
            let rp = unsafe { objc2_metal::MTL4RenderPassDescriptor::new() };
            unsafe {
                let ca0 = rp.colorAttachments().objectAtIndexedSubscript(0);
                let tex = drawable.texture();
                ca0.setTexture(Some(&tex));
            }
            Some(rp)
        }
        fn current_drawable(
            &self,
        ) -> Option<Retained<ProtocolObject<dyn objc2_quartz_core::CAMetalDrawable>>> {
            self.pending_drawable.borrow_mut().take()
        }
        fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
            self.device.clone()
        }
        fn color_pixel_format(&self) -> MTLPixelFormat {
            unsafe { self.layer.pixelFormat() }
        }
    }

    fn attach_layer_to_nsview(ns_view: *mut objc2::runtime::AnyObject, layer: &CAMetalLayer) {
        unsafe {
            let _: () = msg_send![ns_view, setWantsLayer: true];
        }
        let ca_layer: &CALayer = (&*layer).as_super();
        unsafe {
            let _: () = msg_send![ns_view, setLayer: Some(ca_layer)];
        }
    }
    fn detach_layer_from_nsview(ns_view: *mut objc2::runtime::AnyObject) {
        unsafe {
            let _: () = msg_send![ns_view, setLayer: Option::<&CALayer>::None];
        }
        unsafe {
            let _: () = msg_send![ns_view, setWantsLayer: false];
        }
    }

    struct App {
        window: Option<Window>,
        ns_view: Option<*mut objc2::runtime::AnyObject>,
        surface: Option<Surface>,
        start: Option<Instant>,
        swapchain: kiln::swapchain::SwapchainConfig,
        translator: kiln::events::WinitEventTranslator,
        draw: DrawFn,
        title: String,
    }
    impl App {
        fn new(draw: DrawFn, title: &str) -> Self {
            Self {
                window: None,
                ns_view: None,
                surface: None,
                start: None,
                swapchain: kiln::swapchain::SwapchainConfig::default(),
                translator: kiln::events::WinitEventTranslator::new(),
                draw,
                title: title.to_string(),
            }
        }
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.window.is_some() {
                return;
            }
            let attrs = WindowAttributes::default().with_title(self.title.as_str());
            let window = event_loop.create_window(attrs).expect("create window");
            let ns_view: *mut objc2::runtime::AnyObject = {
                let wh = window.window_handle().unwrap();
                let raw = wh.as_raw();
                match raw {
                    RawWindowHandle::AppKit(h) => h.ns_view.as_ptr().cast(),
                    _ => core::ptr::null_mut(),
                }
            };
            assert!(!ns_view.is_null(), "Expected AppKit NSView");
            let device = MTLCreateSystemDefaultDevice().expect("no system device");
            let layer = unsafe { CAMetalLayer::layer() };
            let sc = self.swapchain;
            unsafe {
                layer.setDevice(Some(&device));
                let size = window.inner_size();
                apply_swapchain_to_metal_layer(&layer, size.width as f64, size.height as f64, &sc);
            }
            attach_layer_to_nsview(ns_view, &layer);
            let surface = Surface {
                layer: layer.clone(),
                device: device.clone(),
                pending_drawable: RefCell::new(None),
            };
            self.start = Some(Instant::now());
            self.surface = Some(surface);
            self.ns_view = Some(ns_view);
            self.window = Some(window);
        }
        fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            window_id: winit::window::WindowId,
            event: WindowEvent,
        ) {
            let Some(window) = self.window.as_ref() else {
                return;
            };
            if window.id() != window_id {
                return;
            }
            match event {
                WindowEvent::CloseRequested => {
                    if let Some(view) = self.ns_view.take() {
                        detach_layer_from_nsview(view);
                    }
                    request_app_exit(event_loop);
                }
                WindowEvent::ModifiersChanged(m) => {
                    self.translator.update_modifiers(m);
                }
                WindowEvent::MouseInput { .. }
                | WindowEvent::CursorMoved { .. }
                | WindowEvent::MouseWheel { .. }
                | WindowEvent::KeyboardInput { .. }
                | WindowEvent::Touch(_) => {
                    if let Some(mapped) = self.translator.process(&event) {
                        match mapped {
                            kiln::events::AppEvent::MouseInput { button, state, .. } => {
                                println!("mouse click: {:?} {:?}", button, state)
                            }
                            kiln::events::AppEvent::CursorMoved { x, y, .. } => {
                                println!("cursor moved: {x:.1},{y:.1}")
                            }
                            kiln::events::AppEvent::MouseWheel {
                                delta_x, delta_y, ..
                            } => println!("wheel: {delta_x:.2},{delta_y:.2}"),
                            kiln::events::AppEvent::Key {
                                state,
                                key_code,
                                repeat,
                                text,
                                ..
                            } => println!(
                                "key: code={key_code} {:?} repeat={repeat} text={:?}",
                                state, text
                            ),
                            kiln::events::AppEvent::Touch {
                                id,
                                phase,
                                x,
                                y,
                                force,
                            } => println!(
                                "touch: id={id} {:?} {x:.1},{y:.1} force={:?}",
                                phase, force
                            ),
                            _ => {}
                        }
                    }
                }
                WindowEvent::Resized(size) => {
                    if let Some(surface) = self.surface.as_ref() {
                        unsafe {
                            surface.layer.setDrawableSize(CGSize {
                                width: size.width as f64,
                                height: size.height as f64,
                            });
                        }
                    }
                    window.request_redraw();
                }
                WindowEvent::RedrawRequested => {
                    if let (Some(surface), Some(start)) =
                        (self.surface.as_ref(), self.start.as_ref())
                    {
                        let t = -start.elapsed().as_secs_f32();
                        (self.draw)(surface, t);
                    }
                }
                _ => {}
            }
        }
        fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
            if let Some(window) = self.window.as_ref() {
                window.request_redraw();
            }
        }
    }

    let event_loop = EventLoop::new().expect("create event loop");
    let mut app = App::new(draw, config.title);
    let _ = event_loop.run_app(&mut app);
}

// High-level API for AppKit: run ExampleApp with hooks (init/update/draw/quit)
#[cfg(all(not(feature = "winit"), target_os = "macos"))]
pub fn run_app<A: KilnApp + 'static>(app_obj: A) {
    use core::cell::{OnceCell, RefCell};
    use kiln::events::{AppEvent, EventQueue, MouseButton};
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2::{
        define_class, msg_send, ClassType, DefinedClass, MainThreadMarker, MainThreadOnly,
    };
    use objc2_app_kit::{
        NSApplication, NSApplicationActivationPolicy, NSApplicationDelegate, NSBackingStoreType,
        NSWindow, NSWindowStyleMask,
    };
    use objc2_foundation::{
        NSDate, NSNotification, NSObject, NSObjectProtocol, NSPoint, NSRect, NSSize, NSString,
    };
    use objc2_metal_kit::{MTKView, MTKViewDelegate};

    struct ViewIvars {
        queue: RefCell<EventQueue>,
    }
    define_class!(
        #[unsafe(super(MTKView))]
        #[thread_kind = MainThreadOnly]
        #[ivars = ViewIvars]
        struct InputView;
        impl InputView {
            #[unsafe(method(acceptsFirstResponder))]
            fn acceptsFirstResponder(&self) -> bool { true }
            #[unsafe(method(mouseDown:))]
            fn mouseDown(&self, event: &objc2_app_kit::NSEvent) {
                let loc_win = unsafe { event.locationInWindow() };
                let loc = (&*self).as_super().convertPoint_fromView(loc_win, None);
                let mut q = self.ivars().queue.borrow_mut();
                let flags = unsafe { event.modifierFlags() };
                q.push(kiln::events::appkit_mouse_input(MouseButton::Left, ElementState::Pressed, loc.x as f64, loc.y as f64, flags));
            }
            #[unsafe(method(mouseUp:))]
            fn mouseUp(&self, event: &objc2_app_kit::NSEvent) {
                let loc_win = unsafe { event.locationInWindow() };
                let loc = (&*self).as_super().convertPoint_fromView(loc_win, None);
                let mut q = self.ivars().queue.borrow_mut();
                let flags = unsafe { event.modifierFlags() };
                q.push(kiln::events::appkit_mouse_input(MouseButton::Left, ElementState::Released, loc.x as f64, loc.y as f64, flags));
            }
            #[unsafe(method(rightMouseDown:))]
            fn rightMouseDown(&self, event: &objc2_app_kit::NSEvent) {
                let loc_win = unsafe { event.locationInWindow() };
                let loc = (&*self).as_super().convertPoint_fromView(loc_win, None);
                let mut q = self.ivars().queue.borrow_mut();
                let flags = unsafe { event.modifierFlags() };
                q.push(kiln::events::appkit_mouse_input(MouseButton::Right, ElementState::Pressed, loc.x as f64, loc.y as f64, flags));
            }
            #[unsafe(method(rightMouseUp:))]
            fn rightMouseUp(&self, event: &objc2_app_kit::NSEvent) {
                let loc_win = unsafe { event.locationInWindow() };
                let loc = (&*self).as_super().convertPoint_fromView(loc_win, None);
                let mut q = self.ivars().queue.borrow_mut();
                let flags = unsafe { event.modifierFlags() };
                q.push(kiln::events::appkit_mouse_input(MouseButton::Right, ElementState::Released, loc.x as f64, loc.y as f64, flags));
            }
            #[unsafe(method(otherMouseDown:))]
            fn otherMouseDown(&self, event: &objc2_app_kit::NSEvent) {
                let loc_win = unsafe { event.locationInWindow() };
                let loc = (&*self).as_super().convertPoint_fromView(loc_win, None);
                let btn = unsafe { event.buttonNumber() } as u16;
                let mapped = if btn == 2 { MouseButton::Middle } else { MouseButton::Other(btn) };
                let mut q = self.ivars().queue.borrow_mut();
                let flags = unsafe { event.modifierFlags() };
                q.push(kiln::events::appkit_mouse_input(mapped, ElementState::Pressed, loc.x as f64, loc.y as f64, flags));
            }
            #[unsafe(method(otherMouseUp:))]
            fn otherMouseUp(&self, event: &objc2_app_kit::NSEvent) {
                let loc_win = unsafe { event.locationInWindow() };
                let loc = (&*self).as_super().convertPoint_fromView(loc_win, None);
                let btn = unsafe { event.buttonNumber() } as u16;
                let mapped = if btn == 2 { MouseButton::Middle } else { MouseButton::Other(btn) };
                let mut q = self.ivars().queue.borrow_mut();
                let flags = unsafe { event.modifierFlags() };
                q.push(kiln::events::appkit_mouse_input(mapped, ElementState::Released, loc.x as f64, loc.y as f64, flags));
            }
            #[unsafe(method(scrollWheel:))]
            fn scrollWheel(&self, event: &objc2_app_kit::NSEvent) {
                let dx = unsafe { event.deltaX() } as f64;
                let dy = unsafe { event.deltaY() } as f64;
                let precise = unsafe { event.hasPreciseScrollingDeltas() };
                let flags = unsafe { event.modifierFlags() };
                let mut q = self.ivars().queue.borrow_mut();
                q.push(kiln::events::appkit_mouse_wheel(dx, dy, precise, flags));
            }
            #[unsafe(method(keyDown:))]
            fn keyDown(&self, event: &objc2_app_kit::NSEvent) {
                let key = unsafe { event.keyCode() } as u16;
                let rep = unsafe { event.isARepeat() };
                let chars = unsafe { event.characters() };
                let flags = unsafe { event.modifierFlags() };
                let mut q = self.ivars().queue.borrow_mut();
                q.push(kiln::events::appkit_key(ElementState::Pressed, rep, key, chars.as_deref(), flags));
            }
            #[unsafe(method(keyUp:))]
            fn keyUp(&self, event: &objc2_app_kit::NSEvent) {
                let key = unsafe { event.keyCode() } as u16;
                let chars = unsafe { event.characters() };
                let flags = unsafe { event.modifierFlags() };
                let mut q = self.ivars().queue.borrow_mut();
                q.push(kiln::events::appkit_key(ElementState::Released, false, key, chars.as_deref(), flags));
            }
            #[unsafe(method(mouseMoved:))]
            fn mouseMoved(&self, event: &objc2_app_kit::NSEvent) { self.handle_mouse_moved_rust(event); }
            #[unsafe(method(mouseDragged:))]
            fn mouseDragged(&self, event: &objc2_app_kit::NSEvent) { self.handle_mouse_moved_rust(event); }
            #[unsafe(method(rightMouseDragged:))]
            fn rightMouseDragged(&self, event: &objc2_app_kit::NSEvent) { self.handle_mouse_moved_rust(event); }
            #[unsafe(method(otherMouseDragged:))]
            fn otherMouseDragged(&self, event: &objc2_app_kit::NSEvent) { self.handle_mouse_moved_rust(event); }
        }
    );
    impl InputView {
        fn handle_mouse_moved_rust(&self, event: &objc2_app_kit::NSEvent) {
            let loc_win = unsafe { event.locationInWindow() };
            let loc = (&*self).as_super().convertPoint_fromView(loc_win, None);
            let flags = unsafe { event.modifierFlags() };
            let mut q = self.ivars().queue.borrow_mut();
            q.push(kiln::events::appkit_cursor_moved(
                loc.x as f64,
                loc.y as f64,
                flags,
            ));
        }
    }

    struct Ivars {
        start_instant: RefCell<std::time::Instant>,
        last_instant: RefCell<std::time::Instant>,
        device: OnceCell<Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>>,
        window: OnceCell<Retained<NSWindow>>,
        view: OnceCell<Retained<InputView>>,
        title: OnceCell<Retained<NSString>>,
        app: OnceCell<RefCell<Box<dyn KilnApp>>>,
    }
    macro_rules! idcell_set {
        ($name:ident, $this:expr, $value:expr) => {{
            let _ = $this.ivars().$name.set($value);
        }};
    }

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
                let window = {
                    let content_rect = NSRect::new(NSPoint::new(0., 0.), NSSize::new(768., 768.));
                    let style = NSWindowStyleMask::Closable
                        | NSWindowStyleMask::Resizable
                        | NSWindowStyleMask::Titled;
                    let backing_store_type = NSBackingStoreType::Buffered;
                    let flag = false;
                    unsafe {
                        NSWindow::initWithContentRect_styleMask_backing_defer(
                            NSWindow::alloc(mtm),
                            content_rect,
                            style,
                            backing_store_type,
                            flag,
                        )
                    }
                };
                if let Some(t) = self.ivars().title.get() {
                    window.setTitle(t);
                }
                window.setAcceptsMouseMovedEvents(true);

                let device = objc2_metal::MTLCreateSystemDefaultDevice()
                    .expect("failed to get default system device");
                let view: Retained<InputView> = {
                    let this = InputView::alloc(mtm);
                    let this = this.set_ivars(ViewIvars {
                        queue: RefCell::new(EventQueue::new()),
                    });
                    unsafe {
                        msg_send![super(this), initWithFrame: window.frame(), device: Some(&*device)]
                    }
                };
                unsafe {
                    (&*view).as_super().setPaused(false);
                    (&*view).as_super().setPreferredFramesPerSecond(60);
                }

                let app = NSApplication::sharedApplication(mtm);
                app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
                window.setContentView(Some((&*view).as_super().as_super()));
                window.center();
                window.makeKeyAndOrderFront(None);
                #[allow(deprecated)]
                app.activateIgnoringOtherApps(true);

                idcell_set!(device, self, device);
                idcell_set!(window, self, window.clone());
                idcell_set!(view, self, view);
                let v = self.ivars().view.get().unwrap();
                let del = ProtocolObject::from_ref(self);
                unsafe {
                    (&*v).as_super().setDelegate(Some(del));
                }
                window.makeFirstResponder(Some((&*v).as_super().as_super()));

                // Call app.init once we have a surface adapter
                struct ViewSurface<'a> {
                    v: &'a MTKView,
                    d: Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>,
                }
                impl<'a> kiln::swapchain::RenderSurface for ViewSurface<'a> {
                    fn current_mtl4_render_pass_descriptor(
                        &self,
                    ) -> Option<Retained<objc2_metal::MTL4RenderPassDescriptor>>
                    {
                        unsafe { self.v.currentMTL4RenderPassDescriptor() }
                    }
                    fn current_drawable(
                        &self,
                    ) -> Option<Retained<ProtocolObject<dyn objc2_quartz_core::CAMetalDrawable>>>
                    {
                        unsafe { self.v.currentDrawable() }
                    }
                    fn device(&self) -> Retained<ProtocolObject<dyn objc2_metal::MTLDevice>> {
                        self.d.clone()
                    }
                    fn color_pixel_format(&self) -> objc2_metal::MTLPixelFormat {
                        unsafe { self.v.colorPixelFormat() }
                    }
                }
                let device = self.ivars().device.get().unwrap();
                let surf = ViewSurface {
                    v: (&*v).as_super(),
                    d: device.clone(),
                };
                if let Some(appcell) = self.ivars().app.get() {
                    appcell.borrow_mut().init(&surf);
                }
            }
            #[unsafe(method(applicationShouldTerminateAfterLastWindowClosed:))]
            unsafe fn applicationShouldTerminateAfterLastWindowClosed(
                &self,
                _app: &NSApplication,
            ) -> bool {
                if let Some(appcell) = self.ivars().app.get() {
                    appcell.borrow_mut().quit();
                }
                true
            }
        }
        unsafe impl MTKViewDelegate for Delegate {
            #[unsafe(method(drawInMTKView:))]
            unsafe fn drawInMTKView(&self, view: &MTKView) {
                // Log queued AppKit events (parity with Winit logging)
                if let Some(v) = self.ivars().view.get() {
                    v.ivars().queue.borrow_mut().drain(|ev| match ev {
                        AppEvent::MouseInput { button, state, .. } => {
                            println!("mouse click: {:?} {:?}", button, state)
                        }
                        AppEvent::CursorMoved { x, y, .. } => {
                            println!("cursor moved: {x:.1},{y:.1}")
                        }
                        AppEvent::MouseWheel {
                            delta_x, delta_y, ..
                        } => println!("wheel: {delta_x:.2},{delta_y:.2}"),
                        AppEvent::Key {
                            state,
                            key_code,
                            repeat,
                            text,
                            ..
                        } => println!(
                            "key: code={key_code} {:?} repeat={repeat} text={:?}",
                            state, text
                        ),
                        AppEvent::Touch {
                            id,
                            phase,
                            x,
                            y,
                            force,
                        } => println!("touch: id={id} {:?} {x:.1},{y:.1} force={:?}", phase, force),
                        AppEvent::RedrawRequested | AppEvent::CloseRequested => {}
                    });
                }

                // Call update/draw hooks with monotonic dt/t
                let now = std::time::Instant::now();
                let start = *self.ivars().start_instant.borrow();
                let last = *self.ivars().last_instant.borrow();
                let t = now.duration_since(start).as_secs_f32();
                let dt = now.duration_since(last).as_secs_f32();
                *self.ivars().last_instant.borrow_mut() = now;

                struct ViewSurface<'a> {
                    v: &'a MTKView,
                    d: Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>,
                }
                impl<'a> kiln::swapchain::RenderSurface for ViewSurface<'a> {
                    fn current_mtl4_render_pass_descriptor(
                        &self,
                    ) -> Option<Retained<objc2_metal::MTL4RenderPassDescriptor>>
                    {
                        unsafe { self.v.currentMTL4RenderPassDescriptor() }
                    }
                    fn current_drawable(
                        &self,
                    ) -> Option<Retained<ProtocolObject<dyn objc2_quartz_core::CAMetalDrawable>>>
                    {
                        unsafe { self.v.currentDrawable() }
                    }
                    fn device(&self) -> Retained<ProtocolObject<dyn objc2_metal::MTLDevice>> {
                        self.d.clone()
                    }
                    fn color_pixel_format(&self) -> objc2_metal::MTLPixelFormat {
                        unsafe { self.v.colorPixelFormat() }
                    }
                }
                let device = self.ivars().device.get().unwrap();
                let surf = ViewSurface {
                    v: view,
                    d: device.clone(),
                };
                if let Some(appcell) = self.ivars().app.get() {
                    let mut app = appcell.borrow_mut();
                    app.update(dt);
                    app.draw(&surf, t);
                }
            }
            #[unsafe(method(mtkView:drawableSizeWillChange:))]
            unsafe fn mtkView_drawableSizeWillChange(&self, _view: &MTKView, _size: NSSize) {}
        }
    );

    let mtm = MainThreadMarker::new().unwrap();
    let app = NSApplication::sharedApplication(mtm);
    app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
    let delegate: Retained<Delegate> = {
        let this = Delegate::alloc(mtm);
        let now = std::time::Instant::now();
        let this = this.set_ivars(Ivars {
            start_instant: RefCell::new(now),
            last_instant: RefCell::new(now),
            device: OnceCell::default(),
            window: OnceCell::default(),
            view: OnceCell::default(),
            title: OnceCell::default(),
            app: OnceCell::default(),
        });
        unsafe { msg_send![super(this), init] }
    };
    let _ = delegate
        .ivars()
        .title
        .set(NSString::from_str(app_obj.title()));
    let _ = delegate.ivars().app.set(RefCell::new(Box::new(app_obj)));
    let object = ProtocolObject::from_ref(&*delegate);
    app.setDelegate(Some(object));
    app.run();
}

// High-level: run a stateful ExampleApp (Winit backend)
#[cfg(all(feature = "winit", target_os = "macos"))]
pub fn run_app<A: KilnApp + 'static>(mut example: A) {
    use core::cell::RefCell;
    use kiln::windowing::{apply_swapchain_to_metal_layer, request_app_exit};
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2::{msg_send, ClassType};
    use objc2_core_foundation::CGSize;
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice, MTLPixelFormat};
    use objc2_quartz_core::{CALayer, CAMetalDrawable as _, CAMetalLayer};
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
        pending_drawable:
            RefCell<Option<Retained<ProtocolObject<dyn objc2_quartz_core::CAMetalDrawable>>>>,
    }
    impl kiln::renderer::RenderSurface for Surface {
        fn current_mtl4_render_pass_descriptor(
            &self,
        ) -> Option<Retained<objc2_metal::MTL4RenderPassDescriptor>> {
            let Some(drawable) = (unsafe { self.layer.nextDrawable() }) else {
                return None;
            };
            self.pending_drawable.replace(Some(drawable.clone()));
            let rp = unsafe { objc2_metal::MTL4RenderPassDescriptor::new() };
            unsafe {
                let ca0 = rp.colorAttachments().objectAtIndexedSubscript(0);
                let tex = drawable.texture();
                ca0.setTexture(Some(&tex));
            }
            Some(rp)
        }
        fn current_drawable(
            &self,
        ) -> Option<Retained<ProtocolObject<dyn objc2_quartz_core::CAMetalDrawable>>> {
            self.pending_drawable.borrow_mut().take()
        }
        fn device(&self) -> Retained<ProtocolObject<dyn MTLDevice>> {
            self.device.clone()
        }
        fn color_pixel_format(&self) -> MTLPixelFormat {
            unsafe { self.layer.pixelFormat() }
        }
    }
    fn attach_layer_to_nsview(ns_view: *mut objc2::runtime::AnyObject, layer: &CAMetalLayer) {
        unsafe {
            let _: () = msg_send![ns_view, setWantsLayer: true];
        }
        let ca_layer: &CALayer = (&*layer).as_super();
        unsafe {
            let _: () = msg_send![ns_view, setLayer: Some(ca_layer)];
        }
    }
    fn detach_layer_from_nsview(ns_view: *mut objc2::runtime::AnyObject) {
        unsafe {
            let _: () = msg_send![ns_view, setLayer: Option::<&CALayer>::None];
            let _: () = msg_send![ns_view, setWantsLayer: false];
        }
    }

    struct Handler {
        window: Option<Window>,
        ns_view: Option<*mut objc2::runtime::AnyObject>,
        surface: Option<Surface>,
        start: Option<Instant>,
        last: Option<Instant>,
        swapchain: kiln::swapchain::SwapchainConfig,
        translator: kiln::events::WinitEventTranslator,
        app: Box<dyn KilnApp>,
    }
    impl Handler {
        fn new(app: Box<dyn KilnApp>) -> Self {
            Self {
                window: None,
                ns_view: None,
                surface: None,
                start: None,
                last: None,
                swapchain: kiln::swapchain::SwapchainConfig::default(),
                translator: kiln::events::WinitEventTranslator::new(),
                app,
            }
        }
    }
    impl ApplicationHandler for Handler {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.window.is_some() {
                return;
            }
            let attrs = WindowAttributes::default().with_title(self.app.title());
            let window = event_loop.create_window(attrs).expect("create window");
            let ns_view: *mut objc2::runtime::AnyObject = {
                let wh = window.window_handle().unwrap();
                let raw = wh.as_raw();
                match raw {
                    RawWindowHandle::AppKit(h) => h.ns_view.as_ptr().cast(),
                    _ => core::ptr::null_mut(),
                }
            };
            assert!(!ns_view.is_null(), "Expected AppKit NSView");
            let device = MTLCreateSystemDefaultDevice().expect("no system device");
            let layer = unsafe { CAMetalLayer::layer() };
            let sc = self.swapchain;
            unsafe {
                layer.setDevice(Some(&device));
                let size = window.inner_size();
                apply_swapchain_to_metal_layer(&layer, size.width as f64, size.height as f64, &sc);
            }
            attach_layer_to_nsview(ns_view, &layer);
            let surface = Surface {
                layer: layer.clone(),
                device: device.clone(),
                pending_drawable: RefCell::new(None),
            };
            self.app.init(&surface);
            self.start = Some(Instant::now());
            self.surface = Some(surface);
            self.ns_view = Some(ns_view);
            self.window = Some(window);
        }
        fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            window_id: winit::window::WindowId,
            event: WindowEvent,
        ) {
            let Some(window) = self.window.as_ref() else {
                return;
            };
            if window.id() != window_id {
                return;
            }
            match event {
                WindowEvent::CloseRequested => {
                    self.app.quit();
                    if let Some(view) = self.ns_view.take() {
                        detach_layer_from_nsview(view);
                    }
                    request_app_exit(event_loop);
                }
                WindowEvent::ModifiersChanged(m) => {
                    self.translator.update_modifiers(m);
                }
                WindowEvent::MouseInput { .. }
                | WindowEvent::CursorMoved { .. }
                | WindowEvent::MouseWheel { .. }
                | WindowEvent::KeyboardInput { .. }
                | WindowEvent::Touch(_) => {
                    if let Some(mapped) = self.translator.process(&event) {
                        match mapped {
                            kiln::events::AppEvent::MouseInput { button, state, .. } => {
                                println!("mouse click: {:?} {:?}", button, state)
                            }
                            kiln::events::AppEvent::CursorMoved { x, y, .. } => {
                                println!("cursor moved: {x:.1},{y:.1}")
                            }
                            kiln::events::AppEvent::MouseWheel {
                                delta_x, delta_y, ..
                            } => println!("wheel: {delta_x:.2},{delta_y:.2}"),
                            kiln::events::AppEvent::Key {
                                state,
                                key_code,
                                repeat,
                                text,
                                ..
                            } => println!(
                                "key: code={key_code} {:?} repeat={repeat} text={:?}",
                                state, text
                            ),
                            kiln::events::AppEvent::Touch {
                                id,
                                phase,
                                x,
                                y,
                                force,
                            } => println!(
                                "touch: id={id} {:?} {x:.1},{y:.1} force={:?}",
                                phase, force
                            ),
                            _ => {}
                        }
                    }
                }
                WindowEvent::Resized(size) => {
                    if let Some(surface) = self.surface.as_ref() {
                        unsafe {
                            surface.layer.setDrawableSize(CGSize {
                                width: size.width as f64,
                                height: size.height as f64,
                            });
                        }
                    }
                    window.request_redraw();
                }
                WindowEvent::RedrawRequested => {
                    if let (Some(surface), Some(start)) =
                        (self.surface.as_ref(), self.start.as_ref())
                    {
                        let now = Instant::now();
                        let dt = self
                            .last
                            .map(|lf| now.duration_since(lf).as_secs_f32())
                            .unwrap_or(0.0);
                        self.last = Some(now);
                        self.app.update(dt);
                        let t = -start.elapsed().as_secs_f32();
                        self.app.draw(surface, t);
                    }
                }
                _ => {}
            }
        }
        fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
            if let Some(window) = self.window.as_ref() {
                window.request_redraw();
            }
        }
    }
    let event_loop = EventLoop::new().expect("create event loop");
    let mut handler = Handler::new(Box::new(example));
    let _ = event_loop.run_app(&mut handler);
}

// ---------------- AppKit-only backend (macOS) ----------------
#[cfg(all(not(feature = "winit"), target_os = "macos"))]
pub fn run(config: RunConfig, draw: DrawFn) {
    use core::cell::{OnceCell, RefCell};
    use kiln::events::{AppEvent, EventQueue, MouseButton};
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2::{
        define_class, msg_send, ClassType, DefinedClass, MainThreadMarker, MainThreadOnly,
    };
    use objc2_app_kit::{
        NSApplication, NSApplicationActivationPolicy, NSApplicationDelegate, NSBackingStoreType,
        NSWindow, NSWindowStyleMask,
    };
    use objc2_foundation::{
        NSDate, NSNotification, NSObject, NSObjectProtocol, NSPoint, NSRect, NSSize, NSString,
    };
    use objc2_metal_kit::{MTKView, MTKViewDelegate};

    struct ViewIvars {
        queue: RefCell<EventQueue>,
    }
    define_class!(
        #[unsafe(super(MTKView))]
        #[thread_kind = MainThreadOnly]
        #[ivars = ViewIvars]
        struct InputViewDraw;
        impl InputViewDraw {
            #[unsafe(method(acceptsFirstResponder))]
            fn acceptsFirstResponder(&self) -> bool { true }
            #[unsafe(method(mouseDown:))]
            fn mouseDown(&self, event: &objc2_app_kit::NSEvent) {
                let loc_win = unsafe { event.locationInWindow() };
                let loc = (&*self).as_super().convertPoint_fromView(loc_win, None);
                let mut q = self.ivars().queue.borrow_mut();
                let flags = unsafe { event.modifierFlags() };
                q.push(kiln::events::appkit_mouse_input(MouseButton::Left, ElementState::Pressed, loc.x as f64, loc.y as f64, flags));
            }
            #[unsafe(method(mouseUp:))]
            fn mouseUp(&self, event: &objc2_app_kit::NSEvent) {
                let loc_win = unsafe { event.locationInWindow() };
                let loc = (&*self).as_super().convertPoint_fromView(loc_win, None);
                let mut q = self.ivars().queue.borrow_mut();
                let flags = unsafe { event.modifierFlags() };
                q.push(kiln::events::appkit_mouse_input(MouseButton::Left, ElementState::Released, loc.x as f64, loc.y as f64, flags));
            }
            #[unsafe(method(rightMouseDown:))]
            fn rightMouseDown(&self, event: &objc2_app_kit::NSEvent) {
                let loc_win = unsafe { event.locationInWindow() };
                let loc = (&*self).as_super().convertPoint_fromView(loc_win, None);
                let mut q = self.ivars().queue.borrow_mut();
                let flags = unsafe { event.modifierFlags() };
                q.push(kiln::events::appkit_mouse_input(MouseButton::Right, ElementState::Pressed, loc.x as f64, loc.y as f64, flags));
            }
            #[unsafe(method(rightMouseUp:))]
            fn rightMouseUp(&self, event: &objc2_app_kit::NSEvent) {
                let loc_win = unsafe { event.locationInWindow() };
                let loc = (&*self).as_super().convertPoint_fromView(loc_win, None);
                let mut q = self.ivars().queue.borrow_mut();
                let flags = unsafe { event.modifierFlags() };
                q.push(kiln::events::appkit_mouse_input(MouseButton::Right, ElementState::Released, loc.x as f64, loc.y as f64, flags));
            }
            #[unsafe(method(otherMouseDown:))]
            fn otherMouseDown(&self, event: &objc2_app_kit::NSEvent) {
                let loc_win = unsafe { event.locationInWindow() };
                let loc = (&*self).as_super().convertPoint_fromView(loc_win, None);
                let btn = unsafe { event.buttonNumber() } as u16;
                let mapped = if btn == 2 { MouseButton::Middle } else { MouseButton::Other(btn) };
                let mut q = self.ivars().queue.borrow_mut();
                let flags = unsafe { event.modifierFlags() };
                q.push(kiln::events::appkit_mouse_input(mapped, ElementState::Pressed, loc.x as f64, loc.y as f64, flags));
            }
            #[unsafe(method(otherMouseUp:))]
            fn otherMouseUp(&self, event: &objc2_app_kit::NSEvent) {
                let loc_win = unsafe { event.locationInWindow() };
                let loc = (&*self).as_super().convertPoint_fromView(loc_win, None);
                let btn = unsafe { event.buttonNumber() } as u16;
                let mapped = if btn == 2 { MouseButton::Middle } else { MouseButton::Other(btn) };
                let mut q = self.ivars().queue.borrow_mut();
                let flags = unsafe { event.modifierFlags() };
                q.push(kiln::events::appkit_mouse_input(mapped, ElementState::Released, loc.x as f64, loc.y as f64, flags));
            }
            #[unsafe(method(scrollWheel:))]
            fn scrollWheel(&self, event: &objc2_app_kit::NSEvent) {
                let dx = unsafe { event.deltaX() } as f64;
                let dy = unsafe { event.deltaY() } as f64;
                let precise = unsafe { event.hasPreciseScrollingDeltas() };
                let flags = unsafe { event.modifierFlags() };
                let mut q = self.ivars().queue.borrow_mut();
                q.push(kiln::events::appkit_mouse_wheel(dx, dy, precise, flags));
            }
            #[unsafe(method(keyDown:))]
            fn keyDown(&self, event: &objc2_app_kit::NSEvent) {
                let key = unsafe { event.keyCode() } as u16;
                let rep = unsafe { event.isARepeat() };
                let chars = unsafe { event.characters() };
                let flags = unsafe { event.modifierFlags() };
                let mut q = self.ivars().queue.borrow_mut();
                q.push(kiln::events::appkit_key(ElementState::Pressed, rep, key, chars.as_deref(), flags));
            }
            #[unsafe(method(keyUp:))]
            fn keyUp(&self, event: &objc2_app_kit::NSEvent) {
                let key = unsafe { event.keyCode() } as u16;
                let chars = unsafe { event.characters() };
                let flags = unsafe { event.modifierFlags() };
                let mut q = self.ivars().queue.borrow_mut();
                q.push(kiln::events::appkit_key(ElementState::Released, false, key, chars.as_deref(), flags));
            }
            #[unsafe(method(mouseMoved:))]
            fn mouseMoved(&self, event: &objc2_app_kit::NSEvent) { self.handle_mouse_moved_rust(event); }
            #[unsafe(method(mouseDragged:))]
            fn mouseDragged(&self, event: &objc2_app_kit::NSEvent) { self.handle_mouse_moved_rust(event); }
            #[unsafe(method(rightMouseDragged:))]
            fn rightMouseDragged(&self, event: &objc2_app_kit::NSEvent) { self.handle_mouse_moved_rust(event); }
            #[unsafe(method(otherMouseDragged:))]
            fn otherMouseDragged(&self, event: &objc2_app_kit::NSEvent) { self.handle_mouse_moved_rust(event); }
        }
    );
    impl InputViewDraw {
        fn handle_mouse_moved_rust(&self, event: &objc2_app_kit::NSEvent) {
            let loc_win = unsafe { event.locationInWindow() };
            let loc = (&*self).as_super().convertPoint_fromView(loc_win, None);
            let flags = unsafe { event.modifierFlags() };
            let mut q = self.ivars().queue.borrow_mut();
            q.push(kiln::events::appkit_cursor_moved(
                loc.x as f64,
                loc.y as f64,
                flags,
            ));
        }
    }

    struct Ivars {
        start_instant: RefCell<std::time::Instant>,
        device: OnceCell<Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>>,
        window: OnceCell<Retained<NSWindow>>,
        view: OnceCell<Retained<InputViewDraw>>,
        title: OnceCell<Retained<NSString>>,
        draw: OnceCell<RefCell<DrawFn>>,
    }
    macro_rules! idcell_set {
        ($name:ident, $this:expr, $value:expr) => {{
            let _ = $this.ivars().$name.set($value);
        }};
    }

    define_class!(
        #[unsafe(super(NSObject))]
        #[thread_kind = MainThreadOnly]
        #[ivars = Ivars]
        struct DelegateDraw;
        unsafe impl NSObjectProtocol for DelegateDraw {}
        unsafe impl NSApplicationDelegate for DelegateDraw {
            #[unsafe(method(applicationDidFinishLaunching:))]
            unsafe fn applicationDidFinishLaunching(&self, _n: &NSNotification) {
                let mtm = self.mtm();
                let window = {
                    let content_rect = NSRect::new(NSPoint::new(0., 0.), NSSize::new(768., 768.));
                    let style = NSWindowStyleMask::Closable
                        | NSWindowStyleMask::Resizable
                        | NSWindowStyleMask::Titled;
                    let backing_store_type = NSBackingStoreType::Buffered;
                    let flag = false;
                    unsafe {
                        NSWindow::initWithContentRect_styleMask_backing_defer(
                            NSWindow::alloc(mtm),
                            content_rect,
                            style,
                            backing_store_type,
                            flag,
                        )
                    }
                };
                if let Some(t) = self.ivars().title.get() {
                    window.setTitle(t);
                }
                window.setAcceptsMouseMovedEvents(true);

                let device = objc2_metal::MTLCreateSystemDefaultDevice()
                    .expect("failed to get default system device");

                let view: Retained<InputViewDraw> = {
                    let this = InputViewDraw::alloc(mtm);
                    let this = this.set_ivars(ViewIvars {
                        queue: RefCell::new(EventQueue::new()),
                    });
                    unsafe {
                        msg_send![super(this), initWithFrame: window.frame(), device: Some(&*device)]
                    }
                };
                unsafe {
                    (&*view).as_super().setPaused(false);
                    (&*view).as_super().setPreferredFramesPerSecond(60);
                }

                let app = NSApplication::sharedApplication(mtm);
                app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
                window.setContentView(Some((&*view).as_super().as_super()));
                window.center();
                window.makeKeyAndOrderFront(None);
                #[allow(deprecated)]
                app.activateIgnoringOtherApps(true);

                idcell_set!(device, self, device);
                idcell_set!(window, self, window.clone());
                idcell_set!(view, self, view);
                let v = self.ivars().view.get().unwrap();
                let del = ProtocolObject::from_ref(self);
                unsafe {
                    (&*v).as_super().setDelegate(Some(del));
                }
                window.makeFirstResponder(Some((&*v).as_super().as_super()));
            }
            #[unsafe(method(applicationShouldTerminateAfterLastWindowClosed:))]
            unsafe fn applicationShouldTerminateAfterLastWindowClosed(
                &self,
                _app: &NSApplication,
            ) -> bool {
                true
            }
        }

        unsafe impl MTKViewDelegate for DelegateDraw {
            #[unsafe(method(drawInMTKView:))]
            unsafe fn drawInMTKView(&self, view: &MTKView) {
                if let Some(v) = self.ivars().view.get() {
                    v.ivars().queue.borrow_mut().drain(|ev| match ev {
                        AppEvent::MouseInput { button, state, .. } => {
                            println!("mouse click: {:?} {:?}", button, state)
                        }
                        AppEvent::CursorMoved { x, y, .. } => {
                            println!("cursor moved: {x:.1},{y:.1}")
                        }
                        AppEvent::MouseWheel {
                            delta_x, delta_y, ..
                        } => println!("wheel: {delta_x:.2},{delta_y:.2}"),
                        AppEvent::Key {
                            state,
                            key_code,
                            repeat,
                            text,
                            ..
                        } => println!(
                            "key: code={key_code} {:?} repeat={repeat} text={:?}",
                            state, text
                        ),
                        AppEvent::Touch {
                            id,
                            phase,
                            x,
                            y,
                            force,
                        } => println!("touch: id={id} {:?} {x:.1},{y:.1} force={:?}", phase, force),
                        AppEvent::RedrawRequested | AppEvent::CloseRequested => {}
                    });
                }
                if let Some(draw_cell) = self.ivars().draw.get() {
                    struct ViewSurface<'a> {
                        v: &'a MTKView,
                        d: Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>,
                    }
                    impl<'a> kiln::swapchain::RenderSurface for ViewSurface<'a> {
                        fn current_mtl4_render_pass_descriptor(
                            &self,
                        ) -> Option<Retained<objc2_metal::MTL4RenderPassDescriptor>>
                        {
                            unsafe { self.v.currentMTL4RenderPassDescriptor() }
                        }
                        fn current_drawable(
                            &self,
                        ) -> Option<Retained<ProtocolObject<dyn objc2_quartz_core::CAMetalDrawable>>>
                        {
                            unsafe { self.v.currentDrawable() }
                        }
                        fn device(&self) -> Retained<ProtocolObject<dyn objc2_metal::MTLDevice>> {
                            self.d.clone()
                        }
                        fn color_pixel_format(&self) -> objc2_metal::MTLPixelFormat {
                            unsafe { self.v.colorPixelFormat() }
                        }
                    }
                    let device = self.ivars().device.get().unwrap();
                    let surface = ViewSurface {
                        v: view,
                        d: device.clone(),
                    };
                    let now = std::time::Instant::now();
                    let start = *self.ivars().start_instant.borrow();
                    (draw_cell.borrow_mut())(&surface, now.duration_since(start).as_secs_f32());
                }
            }
            #[unsafe(method(mtkView:drawableSizeWillChange:))]
            unsafe fn mtkView_drawableSizeWillChange(&self, _view: &MTKView, _size: NSSize) {}
        }
    );

    let mtm = MainThreadMarker::new().unwrap();
    let app = NSApplication::sharedApplication(mtm);
    app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
    let delegate: Retained<DelegateDraw> = {
        let this = DelegateDraw::alloc(mtm);
        let now = std::time::Instant::now();
        let this = this.set_ivars(Ivars {
            start_instant: RefCell::new(now),
            device: OnceCell::default(),
            window: OnceCell::default(),
            view: OnceCell::default(),
            title: OnceCell::default(),
            draw: OnceCell::default(),
        });
        unsafe { msg_send![super(this), init] }
    };
    let _ = delegate.ivars().title.set(NSString::from_str(config.title));
    let _ = delegate.ivars().draw.set(RefCell::new(draw));
    let object = ProtocolObject::from_ref(&*delegate);
    app.setDelegate(Some(object));
    app.run();
}

// No demo runner here; see examples/kiln_example.rs
