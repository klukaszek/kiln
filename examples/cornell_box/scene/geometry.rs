use openusd::math::{mat4_transform_point, mat4_transform_vec};
use openusd::schemas::geom::{Orientation, read_mesh};
use openusd::sdf;
use openusd::usd::Stage;

use crate::Vertex;

use super::material::{Material, MaterialCache};
use super::math::{face_normal, norm3};
use super::transform::world_xform;

pub struct Geometry {
    pub vertices: Vec<Vertex>,
    pub triangle_materials: Vec<u32>,
    pub materials: Vec<Material>,
}

pub fn load_geometry(stage: &Stage, mesh_paths: &[String]) -> anyhow::Result<Geometry> {
    let mut material_cache = MaterialCache::default();
    let mut vertices = Vec::new();
    let mut triangle_materials = Vec::new();

    for mesh_path in mesh_paths {
        let path = sdf::path(mesh_path)?;
        let Some(mesh) = read_mesh(stage, &path)? else {
            continue;
        };

        let world = world_xform(stage, &path)?;
        let material_index = material_cache.material_index_for_prim(stage, mesh_path)?;
        let material = material_cache.material(material_index);
        let color = material.raster_color();

        // `faceVertexIndices` is a flat stream chunked by `faceVertexCounts`; normals in
        // this asset are faceVarying, so they share the same running corner index.
        let mut corner = 0usize;
        for &fvc in &mesh.face_vertex_counts {
            let n = fvc.max(0) as usize;
            if n >= 3 {
                let mut face_points = Vec::with_capacity(n);
                let mut face_normals = Vec::with_capacity(n);

                for k in 0..n {
                    let vi = mesh.face_vertex_indices[corner + k] as usize;
                    face_points.push(mat4_transform_point(&world, mesh.points[vi]));
                    face_normals.push(
                        mesh.normals
                            .as_ref()
                            .and_then(|nr| nr.values.get(corner + k).copied()),
                    );
                }

                let geo_normal = face_normal(&face_points);
                let left_handed = mesh.orientation == Orientation::LeftHanded;

                for t in 1..n - 1 {
                    let tri = if left_handed {
                        [0, t + 1, t]
                    } else {
                        [0, t, t + 1]
                    };
                    triangle_materials.push(material_index);
                    for &k in &tri {
                        let world_normal = match face_normals[k] {
                            Some(local_normal) => norm3(mat4_transform_vec(&world, local_normal)),
                            None => geo_normal,
                        };
                        vertices.push(Vertex {
                            pos: [face_points[k][0], face_points[k][1], face_points[k][2], 1.0],
                            normal: [world_normal[0], world_normal[1], world_normal[2], 0.0],
                            color: [color[0], color[1], color[2], 1.0],
                        });
                    }
                }
            }
            corner += n;
        }
    }

    Ok(Geometry {
        vertices,
        triangle_materials,
        materials: material_cache.into_materials(),
    })
}
