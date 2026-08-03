//! Interactive fly camera for the windowed viewer.
//!
//! WASD moves in the view plane, Q/E descends/climbs along the scene's up axis,
//! Shift speeds up, left-drag looks around, and R returns to the authored USD
//! camera. The path tracer keys its film off the camera basis, so any motion
//! restarts progressive accumulation by itself.
//!
//! Yaw/pitch live in a "levelled" local frame whose Y is the *stage's* up axis
//! ([`spectra::scene::Scene::up`]) — a Z-up stage steered with Y-up controls yaws
//! around the view axis (i.e. rolls) and starts at the gimbal pole, where
//! decomposing the authored matrix turns numerical noise into a finite roll.
//! The controller also never writes the camera until the first actual input, so
//! the authored USD view survives loading bit-exact (and the film key stays
//! stable).

use std::collections::HashSet;
use std::time::Instant;

use glam::{DMat4, DQuat, DVec2, DVec3};
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

use spectra::scene::Scene;

/// Base fly speed in scene units/second (the Cornell box is ~5.5 units tall).
const FLY_SPEED: f64 = 2.5;
const FLY_SPEED_BOOST: f64 = 4.0;
/// Look sensitivity in radians per pixel of drag.
const LOOK_SPEED: f64 = 0.004;

/// Keys that take the camera over from the authored USD transform.
const MOVEMENT_KEYS: [KeyCode; 6] = [
    KeyCode::KeyW,
    KeyCode::KeyA,
    KeyCode::KeyS,
    KeyCode::KeyD,
    KeyCode::KeyQ,
    KeyCode::KeyE,
];
pub struct CameraController {
    /// The authored camera world transform, restored by R.
    home: DMat4,
    /// Rotation taking the controller's Y-up local frame to world space.
    frame: DQuat,
    position: DVec3,
    yaw: f64,
    pitch: f64,
    /// False until the first movement/look input: while false, `update` leaves
    /// the camera untouched (the authored transform, roll and all).
    active: bool,
    reset_requested: bool,
    held: HashSet<KeyCode>,
    dragging: bool,
    cursor: Option<DVec2>,
    last_tick: Instant,
}

impl CameraController {
    pub fn new(world: DMat4, up: DVec3) -> Self {
        let frame = DQuat::from_rotation_arc(DVec3::Y, up.normalize_or(DVec3::Y));
        let (position, yaw, pitch) = Self::decompose(&world, frame);
        Self {
            home: world,
            frame,
            position,
            yaw,
            pitch,
            active: false,
            reset_requested: false,
            held: HashSet::new(),
            dragging: false,
            cursor: None,
            last_tick: Instant::now(),
        }
    }

    /// Position plus yaw/pitch of the camera's view axis in the levelled local
    /// frame. Any authored roll is dropped — the controller keeps the horizon
    /// level once it takes over.
    fn decompose(world: &DMat4, frame: DQuat) -> (DVec3, f64, f64) {
        let forward = frame.inverse() * (-world.z_axis.truncate()).normalize_or(DVec3::NEG_Z);
        (
            world.w_axis.truncate(),
            (-forward.x).atan2(-forward.z),
            forward.y.clamp(-1.0, 1.0).asin(),
        )
    }

    pub fn window_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                let PhysicalKey::Code(code) = event.physical_key else {
                    return;
                };
                match event.state {
                    ElementState::Pressed => {
                        if code == KeyCode::KeyR {
                            self.reset_requested = true;
                        }
                        self.held.insert(code);
                    }
                    ElementState::Released => {
                        self.held.remove(&code);
                    }
                }
            }
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state,
                ..
            } => {
                self.dragging = *state == ElementState::Pressed;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let position = DVec2::new(position.x, position.y);
                if self.dragging
                    && let Some(last) = self.cursor
                {
                    let delta = position - last;
                    if delta != DVec2::ZERO {
                        self.active = true;
                    }
                    self.yaw -= delta.x * LOOK_SPEED;
                    self.pitch = (self.pitch - delta.y * LOOK_SPEED).clamp(
                        -std::f64::consts::FRAC_PI_2 + 0.01,
                        std::f64::consts::FRAC_PI_2 - 0.01,
                    );
                }
                self.cursor = Some(position);
            }
            WindowEvent::Focused(false) => {
                self.held.clear();
                self.dragging = false;
            }
            _ => {}
        }
    }

    /// Integrate held keys and return a new camera transform after user input.
    pub fn update(&mut self) -> Option<DMat4> {
        let dt = self.last_tick.elapsed().as_secs_f64().min(0.1);
        self.last_tick = Instant::now();

        if self.reset_requested {
            self.reset_requested = false;
            self.active = false;
            (self.position, self.yaw, self.pitch) = Self::decompose(&self.home, self.frame);
            return Some(self.home);
        }
        if !self.active {
            // Key takeover is derived from held state here (not from the press
            // event) so movement keys still held across an R reset resume
            // flying immediately instead of waiting for an OS key repeat.
            self.active = MOVEMENT_KEYS.iter().any(|key| self.held.contains(key));
            if !self.active {
                return None;
            }
        }

        let rotation =
            self.frame * DQuat::from_rotation_y(self.yaw) * DQuat::from_rotation_x(self.pitch);
        let mut wish = DVec3::ZERO;
        let held = |code| f64::from(u8::from(self.held.contains(&code)));
        wish += (rotation * DVec3::NEG_Z) * (held(KeyCode::KeyW) - held(KeyCode::KeyS));
        wish += (rotation * DVec3::X) * (held(KeyCode::KeyD) - held(KeyCode::KeyA));
        wish += (self.frame * DVec3::Y) * (held(KeyCode::KeyE) - held(KeyCode::KeyQ));
        if wish != DVec3::ZERO {
            let boost = if self.held.contains(&KeyCode::ShiftLeft)
                || self.held.contains(&KeyCode::ShiftRight)
            {
                FLY_SPEED_BOOST
            } else {
                1.0
            };
            self.position += wish.normalize() * (FLY_SPEED * boost * dt);
        }

        Some(DMat4::from_rotation_translation(rotation, self.position))
    }
}

/// `SPECTRAL_DEBUG_CAMERA=1`: print the authored camera world matrix next to
/// the controller's takeover rebuild, to validate the decompose math per scene.
pub fn debug_camera_roundtrip(scene: &Scene) {
    let authored = scene.camera.world;
    let mut controls = CameraController::new(authored, scene.up);
    controls.active = true;
    let rebuilt = controls.update().unwrap_or(authored);
    eprintln!("up axis:  {:?}", scene.up);
    eprintln!("authored: {authored:.6}");
    eprintln!("rebuilt:  {rebuilt:.6}");
    let drift = (rebuilt - authored)
        .abs()
        .to_cols_array()
        .into_iter()
        .fold(0.0, f64::max);
    eprintln!("max abs drift: {drift:.2e}");
}
