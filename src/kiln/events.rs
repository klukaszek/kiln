#[derive(Debug, Copy, Clone)]
pub enum MouseButton { Left, Right, Middle, Other(u16) }

#[derive(Debug, Copy, Clone)]
pub enum ElementState { Pressed, Released }

#[derive(Debug, Copy, Clone)]
pub struct Modifiers { pub shift: bool, pub control: bool, pub alt: bool, pub command: bool, pub caps_lock: bool }
impl Modifiers { pub const fn none() -> Self { Self { shift: false, control: false, alt: false, command: false, caps_lock: false } } }

#[derive(Debug, Copy, Clone)]
pub enum TouchPhase { Began, Moved, Ended, Cancelled }

#[derive(Debug, Clone)]
pub enum AppEvent {
    RedrawRequested,
    CloseRequested,
    // Mouse
    MouseInput { button: MouseButton, state: ElementState, x: f64, y: f64, modifiers: Modifiers },
    CursorMoved { x: f64, y: f64, modifiers: Modifiers },
    MouseWheel { delta_x: f64, delta_y: f64, precise: bool, modifiers: Modifiers },
    // Keyboard
    Key { state: ElementState, repeat: bool, key_code: u32, text: Option<String>, modifiers: Modifiers },
    // Touch
    Touch { id: u64, phase: TouchPhase, x: f64, y: f64, force: Option<f64> },
}

impl AppEvent { pub fn redraw() -> Self { AppEvent::RedrawRequested } pub fn close() -> Self { AppEvent::CloseRequested } }

pub struct EventQueue { buf: std::vec::Vec<AppEvent> }
impl EventQueue {
    pub fn new() -> Self { Self { buf: std::vec::Vec::new() } }
    pub fn push(&mut self, ev: AppEvent) { self.buf.push(ev); }
    pub fn drain<F: FnMut(AppEvent)>(&mut self, mut f: F) { for ev in self.buf.drain(..) { f(ev) } }
    pub fn is_empty(&self) -> bool { self.buf.is_empty() }
}

// ---------------- AppKit (macOS) helpers ----------------
#[cfg(target_os = "macos")]
use objc2_app_kit::NSEventModifierFlags;

#[cfg(target_os = "macos")]
fn map_appkit_modifiers(flags: NSEventModifierFlags) -> Modifiers {
    Modifiers {
        shift: (flags.0 & NSEventModifierFlags::Shift.0) != 0,
        control: (flags.0 & NSEventModifierFlags::Control.0) != 0,
        alt: (flags.0 & NSEventModifierFlags::Option.0) != 0,
        command: (flags.0 & NSEventModifierFlags::Command.0) != 0,
        caps_lock: (flags.0 & NSEventModifierFlags::CapsLock.0) != 0,
    }
}

#[cfg(target_os = "macos")]
pub fn appkit_mouse_input(button: MouseButton, state: ElementState, x: f64, y: f64, flags: NSEventModifierFlags) -> AppEvent {
    AppEvent::MouseInput { button, state, x, y, modifiers: map_appkit_modifiers(flags) }
}

#[cfg(target_os = "macos")]
pub fn appkit_cursor_moved(x: f64, y: f64, flags: NSEventModifierFlags) -> AppEvent {
    AppEvent::CursorMoved { x, y, modifiers: map_appkit_modifiers(flags) }
}

#[cfg(target_os = "macos")]
pub fn appkit_mouse_wheel(delta_x: f64, delta_y: f64, precise: bool, flags: NSEventModifierFlags) -> AppEvent {
    AppEvent::MouseWheel { delta_x, delta_y, precise, modifiers: map_appkit_modifiers(flags) }
}

#[cfg(target_os = "macos")]
pub fn appkit_key(state: ElementState, repeat: bool, key_code: u16, text: Option<&objc2_foundation::NSString>, flags: NSEventModifierFlags) -> AppEvent {
    let text_str = text.map(|s| s.to_string());
    AppEvent::Key { state, repeat, key_code: key_code as u32, text: text_str, modifiers: map_appkit_modifiers(flags) }
}

#[cfg(target_os = "macos")]
pub fn appkit_touch(id: u64, phase: TouchPhase, x: f64, y: f64, force: Option<f64>) -> AppEvent {
    AppEvent::Touch { id, phase, x, y, force }
}

#[cfg(feature = "winit")]
fn map_modifiers(m: winit::event::Modifiers) -> Modifiers {
    use winit::keyboard::ModifiersState;
    Modifiers {
        shift: m.state().contains(ModifiersState::SHIFT),
        control: m.state().contains(ModifiersState::CONTROL),
        alt: m.state().contains(ModifiersState::ALT),
        command: m.state().contains(ModifiersState::SUPER),
        caps_lock: false,
    }
}

#[cfg(feature = "winit")]
pub fn from_winit_mouse_input(state: winit::event::ElementState, button: winit::event::MouseButton) -> AppEvent {
    let b = match button {
        winit::event::MouseButton::Left => MouseButton::Left,
        winit::event::MouseButton::Right => MouseButton::Right,
        winit::event::MouseButton::Middle => MouseButton::Middle,
        winit::event::MouseButton::Back => MouseButton::Other(4),
        winit::event::MouseButton::Forward => MouseButton::Other(5),
        winit::event::MouseButton::Other(x) => MouseButton::Other(x),
    };
    let s = match state { winit::event::ElementState::Pressed => ElementState::Pressed, winit::event::ElementState::Released => ElementState::Released };
    AppEvent::MouseInput { button: b, state: s, x: 0.0, y: 0.0, modifiers: Modifiers::none() }
}

#[cfg(feature = "winit")]
pub fn from_winit_window_event(event: &winit::event::WindowEvent) -> Option<AppEvent> {
    use winit::event::{WindowEvent, ElementState as WElementState, MouseScrollDelta, TouchPhase as WTouchPhase};
    match event {
        WindowEvent::KeyboardInput { event, .. } => {
            use winit::keyboard::PhysicalKey;
            let state = match event.state { WElementState::Pressed => ElementState::Pressed, WElementState::Released => ElementState::Released };
            let key_code = match &event.physical_key { PhysicalKey::Code(code) => *code as u32, PhysicalKey::Unidentified(_) => u32::MAX };
            let text = event.text.as_ref().map(|s| s.to_string());
            Some(AppEvent::Key { state, repeat: event.repeat, key_code, text, modifiers: Modifiers::none() })
        }
        WindowEvent::CursorMoved { position, .. } => Some(AppEvent::CursorMoved { x: position.x, y: position.y, modifiers: Modifiers::none() }),
        WindowEvent::MouseWheel { delta, .. } => {
            let (dx, dy, precise) = match delta {
                MouseScrollDelta::LineDelta(x, y) => (*x as f64, *y as f64, false),
                MouseScrollDelta::PixelDelta(p) => (p.x, p.y, true),
            };
            Some(AppEvent::MouseWheel { delta_x: dx, delta_y: dy, precise, modifiers: Modifiers::none() })
        }
        WindowEvent::MouseInput { state, button, .. } => Some(from_winit_mouse_input(*state, *button)),
        WindowEvent::Touch(t) => {
            let phase = match t.phase { WTouchPhase::Started => TouchPhase::Began, WTouchPhase::Moved => TouchPhase::Moved, WTouchPhase::Ended => TouchPhase::Ended, WTouchPhase::Cancelled => TouchPhase::Cancelled };
            Some(AppEvent::Touch { id: t.id, phase, x: t.location.x, y: t.location.y, force: t.force.map(|f| f.normalized()) })
        }
        _ => None,
    }
}

#[cfg(feature = "winit")]
pub fn from_winit_window_event_with_modifiers(event: &winit::event::WindowEvent, mods: winit::event::Modifiers) -> Option<AppEvent> {
    use winit::event::{WindowEvent, ElementState as WElementState, MouseScrollDelta, TouchPhase as WTouchPhase};
    let mapped_mods = map_modifiers(mods);
    match event {
        WindowEvent::KeyboardInput { event, .. } => {
            use winit::keyboard::PhysicalKey;
            let state = match event.state { WElementState::Pressed => ElementState::Pressed, WElementState::Released => ElementState::Released };
            let key_code = match &event.physical_key { PhysicalKey::Code(code) => *code as u32, PhysicalKey::Unidentified(_) => u32::MAX };
            let text = event.text.as_ref().map(|s| s.to_string());
            Some(AppEvent::Key { state, repeat: event.repeat, key_code, text, modifiers: mapped_mods })
        }
        WindowEvent::CursorMoved { position, .. } => Some(AppEvent::CursorMoved { x: position.x, y: position.y, modifiers: mapped_mods }),
        WindowEvent::MouseWheel { delta, .. } => {
            let (dx, dy, precise) = match delta {
                MouseScrollDelta::LineDelta(x, y) => (*x as f64, *y as f64, false),
                MouseScrollDelta::PixelDelta(p) => (p.x, p.y, true),
            };
            Some(AppEvent::MouseWheel { delta_x: dx, delta_y: dy, precise, modifiers: mapped_mods })
        }
        WindowEvent::MouseInput { state, button, .. } => {
            let mut ev = from_winit_mouse_input(*state, *button);
            if let AppEvent::MouseInput { ref mut modifiers, .. } = ev { *modifiers = mapped_mods; }
            Some(ev)
        }
        WindowEvent::Touch(t) => {
            let phase = match t.phase { WTouchPhase::Started => TouchPhase::Began, WTouchPhase::Moved => TouchPhase::Moved, WTouchPhase::Ended => TouchPhase::Ended, WTouchPhase::Cancelled => TouchPhase::Cancelled };
            Some(AppEvent::Touch { id: t.id, phase, x: t.location.x, y: t.location.y, force: t.force.map(|f| f.normalized()) })
        }
        _ => None,
    }
}

// Convenience translator that keeps Winit modifier state and maps events to AppEvent.
#[cfg(feature = "winit")]
#[derive(Default, Copy, Clone)]
pub struct WinitEventTranslator { mods: winit::event::Modifiers }

#[cfg(feature = "winit")]
impl WinitEventTranslator {
    pub fn new() -> Self { Self { mods: winit::event::Modifiers::default() } }
    pub fn update_modifiers(&mut self, mods: winit::event::Modifiers) { self.mods = mods; }
    pub fn process(&self, event: &winit::event::WindowEvent) -> Option<AppEvent> {
        from_winit_window_event_with_modifiers(event, self.mods)
    }
}
