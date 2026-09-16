//! Analytic 2D scenes, authored in a normalised square and packed for the GPU.
//!
//! Scenes are a handful of signed-distance primitives, so the whole scene is rebuilt on the host
//! every frame and uploaded as a few hundred bytes. Tessellating it into BVH triangles and building
//! the broadphase are GPU passes, so per-frame host work does not grow with the primitive count.

use glam::{Vec2, Vec3, Vec4};
use std::f32::consts::PI;

/// Upper bound on primitives in any scene here, and the capacity every GPU-side scene structure is
/// sized for. The busiest scene uses 26.
pub const MAX_PRIMS: u32 = 64;

/// `float4` rows per packed primitive.
pub const ROWS_PER_PRIM: u32 = 4;

const KIND_CIRCLE: f32 = 0.0;
const KIND_BOX: f32 = 1.0;
const KIND_SEGMENT: f32 = 2.0;

const BLACK: Vec3 = Vec3::ZERO;

#[derive(Clone, Copy, Debug)]
pub struct Prim {
    kind: f32,
    /// Circle/box centre, or segment start.
    a: Vec2,
    /// Box half-extent, or segment end.
    b: Vec2,
    /// Circle radius, box corner radius, or segment thickness.
    r: f32,
    rot: f32,
    emission: Vec3,
    albedo: Vec3,
}

impl Prim {
    fn new(kind: f32, a: Vec2) -> Self {
        Self {
            kind,
            a,
            b: Vec2::ZERO,
            r: 0.0,
            rot: 0.0,
            emission: BLACK,
            albedo: BLACK,
        }
    }

    pub fn emission(mut self, emission: Vec3) -> Self {
        self.emission = emission;
        self
    }

    pub fn albedo(mut self, albedo: Vec3) -> Self {
        self.albedo = albedo;
        self
    }

    fn rot(mut self, radians: f32) -> Self {
        self.rot = radians;
        self
    }

    /// Box corner radius.
    fn round(mut self, round: f32) -> Self {
        self.r = round;
        self
    }
}

pub fn circle(cx: f32, cy: f32, r: f32) -> Prim {
    Prim {
        r,
        ..Prim::new(KIND_CIRCLE, Vec2::new(cx, cy))
    }
}

pub fn box_prim(cx: f32, cy: f32, hx: f32, hy: f32) -> Prim {
    Prim {
        b: Vec2::new(hx, hy),
        ..Prim::new(KIND_BOX, Vec2::new(cx, cy))
    }
}

pub fn segment(ax: f32, ay: f32, bx: f32, by: f32, r: f32) -> Prim {
    Prim {
        b: Vec2::new(bx, by),
        r,
        ..Prim::new(KIND_SEGMENT, Vec2::new(ax, ay))
    }
}

/// Pack primitives into `ROWS_PER_PRIM` `float4` rows each, in the pixel units of a `res`-sized
/// probe grid. Scenes are authored in the unit square, which is centred in the grid's shorter axis
/// so a non-square window widens the view rather than stretching it.
pub fn pack(prims: &[Prim], res: (u32, u32), out: &mut Vec<Vec4>) {
    let scale = res.0.min(res.1) as f32;
    let ox = 0.5 * (res.0 as f32 - scale);
    let oy = 0.5 * (res.1 as f32 - scale);

    out.clear();
    for p in prims {
        let centre = Vec2::new(ox, oy) + p.a * scale;
        out.push(Vec4::new(p.kind, centre.x, centre.y, p.r * scale));
        // A segment's second point is a position; every other kind's is an extent.
        let b = if p.kind == KIND_SEGMENT {
            Vec2::new(ox, oy) + p.b * scale
        } else {
            p.b * scale
        };
        out.push(Vec4::new(b.x, b.y, p.rot.cos(), p.rot.sin()));
        out.push(p.emission.extend(0.0));
        out.push(p.albedo.extend(0.0));
    }
}

// --- scenes -----------------------------------------------------------------

/// Builds a scene's primitives for an animation time in seconds.
pub type SceneBuilder = fn(f32) -> Vec<Prim>;

pub const SCENES: [(&str, SceneBuilder); 4] = [
    ("Three lights", three_lights),
    ("Penumbra", penumbra),
    ("Many lights", many_lights),
    ("Slit", slit),
];

fn walls(albedo: Vec3) -> Vec<Prim> {
    let t = 0.02;
    vec![
        box_prim(0.5, t, 0.5, t).albedo(albedo),
        box_prim(0.5, 1.0 - t, 0.5, t).albedo(albedo),
        box_prim(t, 0.5, t, 0.5).albedo(albedo),
        box_prim(1.0 - t, 0.5, t, 0.5).albedo(albedo),
    ]
}

/// Three coloured emitters orbiting a cluster of occluders.
fn three_lights(t: f32) -> Vec<Prim> {
    let mut prims = walls(Vec3::new(0.35, 0.35, 0.38));
    for (k, colour) in [
        Vec3::new(6.0, 1.2, 0.5),
        Vec3::new(0.5, 5.0, 1.8),
        Vec3::new(0.8, 1.6, 7.0),
    ]
    .into_iter()
    .enumerate()
    {
        let a = t * 0.35 + k as f32 * 2.0 * PI / 3.0;
        // 0.38, not 0.33: the rotated box reaches 0.331 from the centre and the circle 0.316, so a
        // 0.022-radius emitter orbiting any closer sweeps through both. This clears them by 0.027
        // and still leaves 0.058 to the walls.
        prims.push(circle(0.5 + 0.38 * a.cos(), 0.5 + 0.38 * a.sin(), 0.022).emission(colour));
    }
    prims.extend([
        box_prim(0.5, 0.5, 0.11, 0.035)
            .rot(t * 0.2)
            .round(0.01)
            .albedo(Vec3::splat(0.6)),
        box_prim(0.30, 0.66, 0.025, 0.12)
            .rot(-0.4)
            .albedo(Vec3::new(0.7, 0.5, 0.3)),
        circle(0.70, 0.34, 0.06).albedo(Vec3::new(0.3, 0.6, 0.7)),
    ]);
    prims
}

/// Diagnostic: one small emitter behind a long thin occluder. The hard-to-soft shadow transition is
/// the classic failure case for the cascade interpolation, so it is worth having on a key.
fn penumbra(t: f32) -> Vec<Prim> {
    let mut prims = walls(Vec3::splat(0.25));
    let r = 0.006 + 0.012 * (0.5 + 0.5 * (t * 0.6).sin());
    prims.extend([
        circle(0.5, 0.14, r).emission(Vec3::new(9.0, 8.4, 7.4)),
        segment(0.18, 0.42, 0.82, 0.42, 0.006).albedo(Vec3::splat(0.4)),
        box_prim(0.32, 0.72, 0.04, 0.04)
            .rot(0.6)
            .albedo(Vec3::splat(0.8)),
        circle(0.68, 0.74, 0.05).albedo(Vec3::splat(0.8)),
    ]);
    prims
}

/// Sixteen small emitters: tests angular resolution at distance.
fn many_lights(t: f32) -> Vec<Prim> {
    let mut prims = walls(Vec3::new(0.2, 0.2, 0.22));
    let n = 16;
    for k in 0..n {
        let a = t * 0.15 + k as f32 * 2.0 * PI / n as f32;
        let hue = k as f32 / n as f32;
        let channel = |shift: f32| 4.0 * (0.5 + 0.5 * (2.0 * PI * (hue + shift)).cos());
        let colour = Vec3::new(channel(0.0), channel(1.0 / 3.0), channel(2.0 / 3.0));
        prims.push(circle(0.5 + 0.42 * a.cos(), 0.5 + 0.42 * a.sin(), 0.008).emission(colour));
    }
    for k in 0..5 {
        let a = -t * 0.1 + k as f32 * 2.0 * PI / 5.0;
        prims.push(
            box_prim(0.5 + 0.16 * a.cos(), 0.5 + 0.16 * a.sin(), 0.07, 0.016)
                .rot(a)
                .albedo(Vec3::splat(0.6)),
        );
    }
    prims
}

/// Diagnostic: a bright source behind a narrow slit, plus nested occluders.
fn slit(t: f32) -> Vec<Prim> {
    let mut prims = walls(Vec3::splat(0.3));
    let gap = 0.01 + 0.05 * (0.5 + 0.5 * (t * 0.5).sin());
    prims.extend([
        box_prim(0.12, 0.5, 0.05, 0.16).emission(Vec3::new(14.0, 11.0, 6.0)),
        box_prim(0.30, 0.25 - gap * 0.5, 0.02, 0.23).albedo(Vec3::splat(0.5)),
        box_prim(0.30, 0.75 + gap * 0.5, 0.02, 0.23).albedo(Vec3::splat(0.5)),
        circle(0.62, 0.5, 0.07).albedo(Vec3::new(0.85, 0.8, 0.7)),
        circle(0.85, 0.32, 0.04).albedo(Vec3::new(0.85, 0.8, 0.7)),
        circle(0.85, 0.68, 0.04).albedo(Vec3::new(0.85, 0.8, 0.7)),
    ]);
    prims
}

/// The emitter that follows the pointer.
pub fn cursor_light(x: f32, y: f32) -> Prim {
    circle(x, y, 0.012).emission(Vec3::new(5.0, 4.6, 4.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_scene_fits_the_reserved_budget() {
        for (name, build) in SCENES {
            for step in 0..8 {
                // One extra for the cursor light, which the app appends.
                let count = build(step as f32).len() + 1;
                assert!(
                    count as u32 <= MAX_PRIMS,
                    "{name} builds {count} primitives, over the {MAX_PRIMS} reserved"
                );
            }
        }
    }

    #[test]
    fn packing_centres_the_unit_square_in_the_shorter_axis() {
        let mut packed = Vec::new();
        pack(&[circle(0.5, 0.5, 0.25)], (400, 200), &mut packed);
        assert_eq!(packed.len() as u32, ROWS_PER_PRIM);
        // 200px tall, so the square spans x in [100, 300] and its centre lands at (200, 100).
        assert_eq!(packed[0].y, 200.0);
        assert_eq!(packed[0].z, 100.0);
        assert_eq!(packed[0].w, 50.0);
    }
}
