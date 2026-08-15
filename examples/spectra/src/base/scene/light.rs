//! Renderer-neutral analytic and emissive light descriptions.

use glam::{DMat4, Vec3};

/// Shapes understood by the spectral light sampler.
#[derive(Clone, Debug, PartialEq)]
pub enum LightKind {
    Point,
    Directional { angle_deg: f32 },
    Rect { width: f32, height: f32 },
    Disk { radius: f32 },
    Sphere { radius: f32 },
    Dome,
}

impl LightKind {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Point => "Point",
            Self::Directional { .. } => "Directional",
            Self::Rect { .. } => "Rect area",
            Self::Disk { .. } => "Disk area",
            Self::Sphere { .. } => "Sphere",
            Self::Dome => "Dome",
        }
    }
}

/// A light that can be edited by the scene inspector.
#[derive(Clone, Debug)]
pub struct Light {
    pub name: String,
    pub path: String,
    pub kind: LightKind,
    pub transform: DMat4,
    /// Named SPD used to shape this light's emitted power.
    pub spectrum: String,
    pub color: Vec3,
    pub intensity: f32,
    pub enabled: bool,
}

impl Default for Light {
    fn default() -> Self {
        Self {
            name: "Light".into(),
            path: "/Light".into(),
            kind: LightKind::Point,
            transform: DMat4::IDENTITY,
            spectrum: "A".into(),
            color: Vec3::ONE,
            intensity: 100.0,
            enabled: true,
        }
    }
}

impl Light {
    pub fn point(name: impl Into<String>, transform: DMat4) -> Self {
        let name = name.into();
        Self {
            path: format!("/{name}"),
            name,
            transform,
            ..Self::default()
        }
    }
}
