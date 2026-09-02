//! Renderer-neutral analytic light descriptions.

use glam::{DMat4, DVec3, Vec3};

use super::{EmitterExtent, Illuminant};

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
    pub const ALL: [Self; 6] = [
        Self::Point,
        Self::Directional { angle_deg: 0.53 },
        Self::Rect {
            width: 1.0,
            height: 1.0,
        },
        Self::Disk { radius: 0.5 },
        Self::Sphere { radius: 0.5 },
        Self::Dome,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            Self::Point => "POINT",
            Self::Directional { .. } => "DIRECTIONAL",
            Self::Rect { .. } => "RECT AREA",
            Self::Disk { .. } => "DISK AREA",
            Self::Sphere { .. } => "SPHERE",
            Self::Dome => "DOME",
        }
    }

    /// The geometry the light's total emitted power is spread over, in world space. `transform`
    /// scales the authored dimensions, so a scaled rect light reports the area it actually covers.
    pub fn extent(&self, transform: &DMat4) -> EmitterExtent {
        let axis = |axis: DVec3, length: f32| transform.transform_vector3(axis * f64::from(length));
        match *self {
            Self::Rect { width, height } => {
                let area = axis(DVec3::X, width.max(0.0))
                    .cross(axis(DVec3::Y, height.max(0.0)))
                    .length();
                EmitterExtent::Surface(area as f32)
            }
            Self::Disk { radius } => {
                let area = std::f64::consts::PI
                    * axis(DVec3::X, radius.max(0.0))
                        .cross(axis(DVec3::Y, radius.max(0.0)))
                        .length();
                EmitterExtent::Surface(area as f32)
            }
            Self::Sphere { radius } => {
                let scaled = axis(DVec3::X, radius.max(0.0)).length();
                EmitterExtent::Surface((4.0 * std::f64::consts::PI * scaled * scaled) as f32)
            }
            Self::Point => EmitterExtent::Punctual,
            Self::Directional { .. } | Self::Dome => EmitterExtent::Infinite,
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
    pub illuminant: Illuminant,
}

impl Default for Light {
    fn default() -> Self {
        Self {
            name: "Light".into(),
            path: "/Light".into(),
            kind: LightKind::Point,
            transform: DMat4::IDENTITY,
            illuminant: Illuminant::luminance(Vec3::ONE, 100.0),
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

    /// The geometry this light's total emitted power is spread over.
    pub fn extent(&self) -> EmitterExtent {
        self.kind.extent(&self.transform)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_extent_follows_the_light_transform() {
        let kind = LightKind::Rect {
            width: 2.0,
            height: 3.0,
        };
        let EmitterExtent::Surface(area) = kind.extent(&DMat4::IDENTITY) else {
            panic!("a rect light is a surface emitter");
        };
        assert!((area - 6.0).abs() < 1e-4);

        let scaled = kind.extent(&DMat4::from_scale(DVec3::splat(2.0)));
        let EmitterExtent::Surface(scaled) = scaled else {
            panic!("a rect light is a surface emitter");
        };
        assert!((scaled - 24.0).abs() < 1e-3, "{scaled}");
    }

    #[test]
    fn punctual_and_infinite_lights_have_no_area() {
        assert_eq!(
            LightKind::Point.extent(&DMat4::IDENTITY),
            EmitterExtent::Punctual
        );
        assert_eq!(
            LightKind::Dome.extent(&DMat4::IDENTITY),
            EmitterExtent::Infinite
        );
    }
}
