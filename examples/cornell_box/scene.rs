//! USD → GPU triangle-soup loader for the Cornell box example.
//!
//! Pulls every `Mesh` prim out of the stage, bakes its vertices into world space (folding
//! in the asset's Z-up→Y-up root transform), triangulates the quads, and tags each vertex
//! with its material's colour. Also derives a row-vector `view_proj` from the scene's
//! authored `Camera` prim. The result is a flat, GPU-ready vertex array plus the matrix —
//! everything the mesh shader needs, with no USD types crossing into the render code.
//!
//! All matrix math follows openusd's convention: `[f64; 16]` row-major, **row-vector**
//! (`p' = p · M`, translation in indices 12..14).

use openusd::math::{mat4_inverse, mat4_mul, mat4_transform_point, mat4_transform_vec, IDENTITY_MAT4};
use openusd::schemas::geom::{find_geom_prims, read_camera, read_mesh, Orientation};
use openusd::sdf::{self, Value};
use openusd::usd::Stage;

// The render vertex is `crate::Vertex`, defined via `gpu_struct!` in `main.rs` so its host
// layout and the Slang declaration stay in lockstep. `float4` lanes throughout sidestep any
// `float3` pointer-packing ambiguity (the `.w` lanes are unused padding).
use crate::Vertex;

/// Everything the renderer needs from the scene.
pub struct Scene {
    pub vertices: Vec<Vertex>,
    /// Camera world transform (used to rebuild `view_proj` per-frame for the live aspect).
    pub camera_world: [f64; 16],
    /// Authored camera intrinsics (fov, clip range) for the projection.
    pub camera: openusd::schemas::geom::ReadCamera,
}

impl Scene {
    /// Number of triangles in the soup (three vertices each).
    pub fn triangle_count(&self) -> u32 {
        (self.vertices.len() / 3) as u32
    }

    /// Camera world-space position (translation row of the world transform).
    pub fn camera_pos(&self) -> [f32; 3] {
        let m = &self.camera_world;
        [m[12] as f32, m[13] as f32, m[14] as f32]
    }

    /// Build the row-vector `view_proj = view · proj` for the given viewport aspect.
    /// Right-handed (camera looks down -Z), Y-up, depth range [0, 1] — matching Kiln's
    /// normalized clip space. Returned as four `float4` rows for layout-free shader use.
    pub fn view_proj_rows(&self, aspect: f32) -> [[f32; 4]; 4] {
        let view = mat4_inverse(&self.camera_world).unwrap_or(IDENTITY_MAT4);
        let proj = perspective_rh_zo(
            self.camera.vertical_fov_rad(),
            aspect,
            self.camera.clipping_range[0].max(1e-3),
            self.camera.clipping_range[1],
        );
        let vp = mat4_mul(&view, &proj);
        [
            [vp[0] as f32, vp[1] as f32, vp[2] as f32, vp[3] as f32],
            [vp[4] as f32, vp[5] as f32, vp[6] as f32, vp[7] as f32],
            [vp[8] as f32, vp[9] as f32, vp[10] as f32, vp[11] as f32],
            [vp[12] as f32, vp[13] as f32, vp[14] as f32, vp[15] as f32],
        ]
    }
}

/// Load `path` (a `.usda`/`.usdc`/`.usdz`) into a [`Scene`].
pub fn load(path: &str) -> anyhow::Result<Scene> {
    let stage = Stage::open(path)?;
    let prims = find_geom_prims(&stage)?;

    // Resolve each material path's colour once.
    let mut material_cache: std::collections::HashMap<String, [f32; 3]> = Default::default();

    let mut vertices = Vec::new();
    for mesh_path in &prims.meshes {
        let path = sdf::path(mesh_path)?;
        let Some(mesh) = read_mesh(&stage, &path)? else {
            continue;
        };
        let world = world_xform(&stage, &path)?;

        // Material colour for this whole mesh (the asset binds one material per mesh).
        let color = match material_binding(&stage, mesh_path)? {
            Some(mat) => *material_cache
                .entry(mat.clone())
                .or_insert_with(|| material_color(&stage, &mat).unwrap_or([0.8, 0.8, 0.8])),
            None => [0.8, 0.8, 0.8],
        };

        // Walk the face list, fan-triangulating each polygon. `faceVertexIndices` is a flat
        // stream chunked by `faceVertexCounts`; `normals` are faceVarying (one per corner).
        let mut corner = 0usize; // running index into the faceVarying / index streams
        for &fvc in &mesh.face_vertex_counts {
            let n = fvc.max(0) as usize;
            if n >= 3 {
                // Per-corner world positions + normals for this face.
                let mut fp = Vec::with_capacity(n);
                let mut fn_ = Vec::with_capacity(n);
                for k in 0..n {
                    let vi = mesh.face_vertex_indices[corner + k] as usize;
                    let p = mesh.points[vi];
                    fp.push(mat4_transform_point(&world, p));
                    // faceVarying normal if present, else the face's geometric normal.
                    let nrm = mesh
                        .normals
                        .as_ref()
                        .and_then(|nr| nr.values.get(corner + k).copied());
                    fn_.push(nrm);
                }
                let geo_n = face_normal(&fp);
                // Fan triangulation: (0,1,2), (0,2,3), … Reverse for left-handed meshes so
                // front faces stay CCW in Kiln's Y-up clip space.
                let lh = mesh.orientation == Orientation::LeftHanded;
                for t in 1..n - 1 {
                    let tri = if lh { [0, t + 1, t] } else { [0, t, t + 1] };
                    for &k in &tri {
                        let wn = match fn_[k] {
                            Some(local_n) => norm3(mat4_transform_vec(&world, local_n)),
                            None => geo_n,
                        };
                        vertices.push(Vertex {
                            pos: [fp[k][0], fp[k][1], fp[k][2], 1.0],
                            normal: [wn[0], wn[1], wn[2], 0.0],
                            color: [color[0], color[1], color[2], 1.0],
                        });
                    }
                }
            }
            corner += n;
        }
    }

    // Authored camera: take the first one in the scene.
    let cam_path = prims
        .cameras
        .first()
        .ok_or_else(|| anyhow::anyhow!("scene has no Camera prim"))?;
    let cam_sdf = sdf::path(cam_path)?;
    let camera = read_camera(&stage, &cam_sdf)?
        .ok_or_else(|| anyhow::anyhow!("camera prim {cam_path} did not read as a Camera"))?;
    let camera_world = world_xform(&stage, &cam_sdf)?;

    Ok(Scene {
        vertices,
        camera_world,
        camera,
    })
}

// ── transforms ──────────────────────────────────────────────────────────────

/// Compose a prim's local-to-world transform by walking ancestors to the root. Row-vector
/// convention: `world = local · parent_local · … · root_local`, accumulated child-first.
fn world_xform(stage: &Stage, prim: &sdf::Path) -> anyhow::Result<[f64; 16]> {
    use openusd::schemas::geom::compute_local_to_parent_transform;
    let mut w = IDENTITY_MAT4;
    let mut cur = Some(prim.clone());
    while let Some(p) = cur {
        if p.name().is_none() {
            break; // reached the pseudo-root "/"
        }
        let local = compute_local_to_parent_transform(stage, &p, 0.0)?;
        w = mat4_mul(&w, &local);
        cur = p.parent();
    }
    Ok(w)
}

// ── materials ───────────────────────────────────────────────────────────────

/// The path of the material bound to `prim` via `rel material:binding`, if any. Composed
/// relationship targets arrive as a `PathListOp` (list-edited), so flatten it like openusd's
/// own `read_rel_first_target` rather than expecting a plain `PathVec`.
fn material_binding(stage: &Stage, prim_path: &str) -> anyhow::Result<Option<String>> {
    let rel = sdf::path(prim_path)?.append_property("material:binding")?;
    let paths = match stage.field::<Value>(rel, "targetPaths")? {
        Some(Value::PathListOp(op)) => op.flatten(),
        Some(Value::PathVec(v)) => v,
        _ => Vec::new(),
    };
    Ok(paths.first().map(|p| p.as_str().to_string()))
}

/// Resolve a Material prim's colour: its child surface shader's `inputs:diffuseColor`,
/// falling back to `inputs:emissiveColor` (so the emissive light reads bright).
fn material_color(stage: &Stage, material_path: &str) -> anyhow::Result<[f32; 3]> {
    let mat = sdf::path(material_path)?;
    for child in stage.prim_children(mat.clone())? {
        let shader = mat.append_path(child.as_str())?;
        for attr in ["inputs:diffuseColor", "inputs:emissiveColor"] {
            let prop = shader.append_property(attr)?;
            if let Some(Value::Vec3f(c)) = stage.field::<Value>(prop.clone(), "default")? {
                return Ok(c);
            }
            if let Some(Value::Vec3d(c)) = stage.field::<Value>(prop, "default")? {
                return Ok([c[0] as f32, c[1] as f32, c[2] as f32]);
            }
        }
    }
    Ok([0.8, 0.8, 0.8])
}

// ── small vector helpers ────────────────────────────────────────────────────

fn perspective_rh_zo(fovy_rad: f32, aspect: f32, near: f32, far: f32) -> [f64; 16] {
    let f = 1.0 / (fovy_rad as f64 / 2.0).tan();
    let (near, far) = (near as f64, far as f64);
    let mut m = [0.0f64; 16];
    m[0] = f / aspect as f64;
    m[5] = f;
    m[10] = far / (near - far);
    m[11] = -1.0;
    m[14] = far * near / (near - far);
    m
}

fn face_normal(p: &[[f32; 3]]) -> [f32; 3] {
    if p.len() < 3 {
        return [0.0, 1.0, 0.0];
    }
    let a = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
    let b = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
    norm3([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])
}

fn norm3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-8 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}
