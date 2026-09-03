//! The light-kind tag shared between the light list and the integrator.

/// The light-kind tag stored in `GpuLight::normal.w`, which the shader switches on.
///
/// [`slang`](LightTag::slang) emits the shader-side constants from these same discriminants, so the
/// two sides cannot drift apart.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LightTag {
    Triangle = 0,
    Rect = 1,
    Disk = 2,
    Point = 3,
    Directional = 4,
    Dome = 5,
    Sphere = 6,
    Mesh = 7,
}

impl LightTag {
    const ALL: [(Self, &'static str); 8] = [
        (Self::Triangle, "LIGHT_TRIANGLE"),
        (Self::Rect, "LIGHT_RECT"),
        (Self::Disk, "LIGHT_DISK"),
        (Self::Point, "LIGHT_POINT"),
        (Self::Directional, "LIGHT_DIRECTIONAL"),
        (Self::Dome, "LIGHT_DOME"),
        (Self::Sphere, "LIGHT_SPHERE"),
        (Self::Mesh, "LIGHT_MESH"),
    ];

    /// The tag as it is written into the float field the shader reads.
    pub(crate) fn as_f32(self) -> f32 {
        self as u8 as f32
    }

    pub(crate) fn slang() -> String {
        Self::ALL
            .iter()
            .map(|(tag, name)| format!("static const uint {name} = {}u;\n", *tag as u8))
            .collect()
    }
}
