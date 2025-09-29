#[derive(Debug, Copy, Clone)]
pub enum MouseButton { Left, Right, Middle, Other(u16) }

#[derive(Debug, Copy, Clone)]
pub enum ElementState { Pressed, Released }

#[derive(Debug, Copy, Clone)]
pub enum AppEvent { RedrawRequested, CloseRequested, MouseInput { button: MouseButton, state: ElementState, x: f64, y: f64 } }

impl AppEvent { pub fn redraw() -> Self { AppEvent::RedrawRequested } pub fn close() -> Self { AppEvent::CloseRequested } }

pub struct EventQueue { buf: std::vec::Vec<AppEvent> }
impl EventQueue { pub fn new() -> Self { Self { buf: std::vec::Vec::new() } } pub fn push(&mut self, ev: AppEvent) { self.buf.push(ev); } pub fn drain<F: FnMut(AppEvent)>(&mut self, mut f: F) { for ev in self.buf.drain(..) { f(ev) } } pub fn is_empty(&self) -> bool { self.buf.is_empty() } }

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
    let s = match state {
        winit::event::ElementState::Pressed => ElementState::Pressed,
        winit::event::ElementState::Released => ElementState::Released,
    };
    // Position is not available in MouseInput; use 0,0 for now.
    AppEvent::MouseInput { button: b, state: s, x: 0.0, y: 0.0 }
}
