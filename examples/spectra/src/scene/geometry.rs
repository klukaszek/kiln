use anyhow::Context;
use glam::{DVec3, Vec3};
use openusd::schemas::geom::{Interpolation, Orientation, Primvar, ReadMesh, read_mesh};
use openusd::sdf;
use openusd::usd::Stage;

use super::Vertex;
use super::material::{Material, MaterialCache};
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
        let color = material_cache.material(material_index).raster_color();
        append_mesh(
            &mesh,
            world,
            color,
            material_index,
            &mut vertices,
            &mut triangle_materials,
        )
        .with_context(|| format!("reading mesh {mesh_path}"))?;
    }

    Ok(Geometry {
        vertices,
        triangle_materials,
        materials: material_cache.into_materials(),
    })
}

fn append_mesh(
    mesh: &ReadMesh,
    world: glam::DMat4,
    color: Vec3,
    material_index: u32,
    vertices: &mut Vec<Vertex>,
    triangle_materials: &mut Vec<u32>,
) -> anyhow::Result<()> {
    let mut corner_offset = 0usize;
    let normal_transform = world.inverse().transpose();

    for (face_index, &face_vertex_count) in mesh.face_vertex_counts.iter().enumerate() {
        let corner_count = usize::try_from(face_vertex_count)
            .context("faceVertexCounts contains a negative value")?;
        let face_end = corner_offset
            .checked_add(corner_count)
            .context("face vertex count overflow")?;
        anyhow::ensure!(
            face_end <= mesh.face_vertex_indices.len(),
            "faceVertexCounts exceeds faceVertexIndices"
        );

        if corner_count >= 3 {
            let mut points = Vec::with_capacity(corner_count);
            let mut normals = Vec::with_capacity(corner_count);
            for face_corner in 0..corner_count {
                let corner = corner_offset + face_corner;
                let point_index = usize::try_from(mesh.face_vertex_indices[corner])
                    .context("faceVertexIndices contains a negative value")?;
                let point = mesh
                    .points
                    .get(point_index)
                    .with_context(|| format!("point index {point_index} is out of bounds"))?;
                points.push(world.transform_point3(Vec3::from_array(*point).as_dvec3()));
                normals.push(mesh_normal(mesh, face_index, corner, point_index)?);
            }

            let geometric_normal = face_normal(&points);
            let left_handed = mesh.orientation == Orientation::LeftHanded;
            for triangle in 1..corner_count - 1 {
                let corners = if left_handed {
                    [0, triangle + 1, triangle]
                } else {
                    [0, triangle, triangle + 1]
                };
                triangle_materials.push(material_index);
                for corner in corners {
                    let normal = normals[corner]
                        .map(|normal| {
                            normal_transform
                                .transform_vector3(Vec3::from_array(normal).as_dvec3())
                                .normalize_or(DVec3::Y)
                                .as_vec3()
                        })
                        .unwrap_or(geometric_normal);
                    vertices.push(Vertex {
                        pos: points[corner].as_vec3().extend(1.0),
                        normal: normal.extend(0.0),
                        color: color.extend(1.0),
                    });
                }
            }
        }
        corner_offset = face_end;
    }

    anyhow::ensure!(
        corner_offset == mesh.face_vertex_indices.len(),
        "faceVertexIndices has {} trailing entries",
        mesh.face_vertex_indices.len() - corner_offset
    );
    Ok(())
}

fn mesh_normal(
    mesh: &ReadMesh,
    face: usize,
    corner: usize,
    point: usize,
) -> anyhow::Result<Option<[f32; 3]>> {
    let Some(normals) = &mesh.normals else {
        return Ok(None);
    };
    let element = match normals.interpolation {
        Interpolation::Constant => 0,
        Interpolation::Uniform => face,
        Interpolation::Varying | Interpolation::Vertex => point,
        Interpolation::FaceVarying => corner,
    };
    Ok(Some(primvar_value(normals, element, "normal")?))
}

fn primvar_value<T: Copy>(primvar: &Primvar<T>, element: usize, label: &str) -> anyhow::Result<T> {
    anyhow::ensure!(
        primvar.element_size == 1,
        "{label} primvar has unsupported elementSize {}",
        primvar.element_size
    );
    let value_index = if primvar.indices.is_empty() {
        element
    } else {
        usize::try_from(
            *primvar
                .indices
                .get(element)
                .with_context(|| format!("{label} primvar index {element} is missing"))?,
        )
        .with_context(|| format!("{label} primvar contains a negative index"))?
    };
    primvar
        .values
        .get(value_index)
        .copied()
        .with_context(|| format!("{label} primvar value {value_index} is missing"))
}

fn face_normal(points: &[DVec3]) -> Vec3 {
    debug_assert!(points.len() >= 3);
    (points[1] - points[0])
        .cross(points[2] - points[0])
        .normalize_or(DVec3::Y)
        .as_vec3()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_indexed_primvar_values() {
        let primvar = Primvar {
            values: vec![10, 20],
            indices: vec![1, 0],
            element_size: 1,
            ..Default::default()
        };

        assert_eq!(primvar_value(&primvar, 0, "test").unwrap(), 20);
        assert_eq!(primvar_value(&primvar, 1, "test").unwrap(), 10);
    }

    #[test]
    fn selects_normals_by_interpolation_mode() {
        let mesh = ReadMesh {
            normals: Some(Primvar {
                values: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                interpolation: Interpolation::Uniform,
                element_size: 1,
                ..Default::default()
            }),
            ..Default::default()
        };

        assert_eq!(
            mesh_normal(&mesh, 1, 99, 99).unwrap(),
            Some([0.0, 1.0, 0.0])
        );
    }
}
