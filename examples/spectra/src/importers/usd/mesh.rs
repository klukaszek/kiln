use glam::{Vec2, Vec3};
use openusd::schemas::geom::{Interpolation, Orientation, Primvar, ReadMesh, read_mesh};
use openusd::sdf;
use openusd::usd::Stage;
use std::collections::HashMap;

use crate::base::scene::{
    EmissiveComponent, Geometry, Instance, MaterialId, Mesh, MeshId, Primitive, Vertex,
};

use super::material::MaterialLibrary;
use super::transform::world_xform;
use super::{Error, Result};

pub(super) fn load(
    stage: &Stage,
    mesh_paths: &[String],
    materials: &mut MaterialLibrary,
) -> Result<Geometry> {
    let mut meshes = Vec::new();
    let mut instances = Vec::new();
    for mesh_path in mesh_paths {
        let path = sdf::path(mesh_path)?;
        let Some(mesh) = read_mesh(stage, &path)? else {
            continue;
        };
        let mesh_id = MeshId(meshes.len());
        let material_id = materials.id_for_prim(stage, mesh_path)?;
        let asset = decode_mesh(stage, &mesh, material_id, materials)
            .map_err(|error| Error::Invalid(format!("reading mesh {mesh_path}: {error}")))?;
        meshes.push(asset);
        instances.push(Instance {
            mesh: mesh_id,
            path: mesh_path.clone(),
            transform: world_xform(stage, &path)?,
        });
    }
    Ok(Geometry::new(meshes, instances))
}

fn decode_mesh(
    stage: &Stage,
    mesh: &ReadMesh,
    default_material: MaterialId,
    materials: &mut MaterialLibrary,
) -> Result<Mesh> {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut primitives = Vec::new();
    let mut vertex_map = HashMap::<ImportVertexKey, u32>::new();
    let mut corner_offset = 0usize;

    for (face, &count) in mesh.face_vertex_counts.iter().enumerate() {
        let count = usize::try_from(count)
            .map_err(|_| Error::Invalid("negative face vertex count".into()))?;
        let face_end = corner_offset + count;
        if face_end > mesh.face_vertex_indices.len() {
            return Err(Error::Invalid(
                "faceVertexCounts exceeds faceVertexIndices".into(),
            ));
        }
        if count >= 3 {
            let material_id = subset_material(stage, mesh, face, default_material, materials)?;
            let start = indices.len();
            for fan in 1..count - 1 {
                let corners = if mesh.orientation == Orientation::LeftHanded {
                    [0, fan + 1, fan]
                } else {
                    [0, fan, fan + 1]
                };
                for face_corner in corners {
                    let corner = corner_offset + face_corner;
                    let point_index = usize::try_from(mesh.face_vertex_indices[corner])
                        .map_err(|_| Error::Invalid("negative face vertex index".into()))?;
                    let position = *mesh.points.get(point_index).ok_or_else(|| {
                        Error::Invalid(format!("point index {point_index} is out of bounds"))
                    })?;
                    let normal = mesh_normal(mesh, face, corner, point_index)?;
                    let uv = mesh_uv(mesh, face, corner, point_index)?;
                    let key = ImportVertexKey {
                        point_index,
                        normal: normal.map(|value| value.to_array().map(f32::to_bits)),
                        uv: uv.map(|value| value.to_array().map(f32::to_bits)),
                    };
                    let index = if let Some(&index) = vertex_map.get(&key) {
                        index
                    } else {
                        let index = u32::try_from(vertices.len())?;
                        vertices.push(Vertex {
                            position: Vec3::from_array(position),
                            source_index: Some(u32::try_from(point_index)?),
                            normal,
                            uv,
                        });
                        vertex_map.insert(key, index);
                        index
                    };
                    indices.push(index);
                }
            }
            primitives.push(Primitive {
                index_start: start,
                index_count: indices.len() - start,
                material: material_id,
            });
        }
        corner_offset = face_end;
    }
    if corner_offset != mesh.face_vertex_indices.len() {
        return Err(Error::Invalid(
            "faceVertexIndices has trailing entries".into(),
        ));
    }
    let emissive_components =
        connected_emissive_components(&vertices, &indices, &primitives, materials);
    Ok(Mesh {
        vertices,
        indices,
        primitives,
        emissive_components,
    })
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ImportVertexKey {
    point_index: usize,
    normal: Option<[u32; 3]>,
    uv: Option<[u32; 2]>,
}

#[derive(Clone, Copy)]
struct EmissiveTriangle {
    material: MaterialId,
    topology: [u32; 3],
    indices: [u32; 3],
}

fn connected_emissive_components(
    vertices: &[Vertex],
    indices: &[u32],
    primitives: &[Primitive],
    materials: &MaterialLibrary,
) -> Vec<EmissiveComponent> {
    let mut triangles = Vec::new();
    for primitive in primitives {
        let Some(material) = materials.materials().get(primitive.material.0) else {
            continue;
        };
        if !material.is_emissive() {
            continue;
        }
        let range = primitive.index_start..primitive.index_start + primitive.index_count;
        for corners in indices[range].chunks_exact(3) {
            let triangle = [corners[0], corners[1], corners[2]];
            triangles.push(EmissiveTriangle {
                material: primitive.material,
                topology: triangle
                    .map(|index| vertices[index as usize].source_index.unwrap_or(index)),
                indices: triangle,
            });
        }
    }

    let mut sets = DisjointSet::new(triangles.len());
    let mut edges = HashMap::<(MaterialId, u32, u32), usize>::new();
    for (triangle_index, triangle) in triangles.iter().enumerate() {
        for edge in [
            [triangle.topology[0], triangle.topology[1]],
            [triangle.topology[1], triangle.topology[2]],
            [triangle.topology[2], triangle.topology[0]],
        ] {
            let (a, b) = if edge[0] < edge[1] {
                (edge[0], edge[1])
            } else {
                (edge[1], edge[0])
            };
            if let Some(other) = edges.insert((triangle.material, a, b), triangle_index) {
                sets.union(triangle_index, other);
            }
        }
    }

    let mut grouped = HashMap::<(usize, MaterialId), Vec<[u32; 3]>>::new();
    for (triangle_index, triangle) in triangles.into_iter().enumerate() {
        grouped
            .entry((sets.find(triangle_index), triangle.material))
            .or_default()
            .push(triangle.indices);
    }
    let mut components = grouped.into_iter().collect::<Vec<_>>();
    components.sort_by_key(|((root, _), _)| *root);
    components
        .into_iter()
        .map(|((_, material), triangles)| EmissiveComponent {
            material,
            triangles,
        })
        .collect()
}

struct DisjointSet {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl DisjointSet {
    fn new(count: usize) -> Self {
        Self {
            parent: (0..count).collect(),
            size: vec![1; count],
        }
    }

    fn find(&mut self, value: usize) -> usize {
        if self.parent[value] != value {
            self.parent[value] = self.find(self.parent[value]);
        }
        self.parent[value]
    }

    fn union(&mut self, left: usize, right: usize) {
        let mut left = self.find(left);
        let mut right = self.find(right);
        if left == right {
            return;
        }
        if self.size[left] < self.size[right] {
            std::mem::swap(&mut left, &mut right);
        }
        self.parent[right] = left;
        self.size[left] += self.size[right];
    }
}

fn subset_material(
    stage: &Stage,
    mesh: &ReadMesh,
    face: usize,
    default: MaterialId,
    materials: &mut MaterialLibrary,
) -> Result<MaterialId> {
    let face = i32::try_from(face)?;
    for subset in &mesh.subsets {
        if subset.indices.contains(&face) {
            return materials.id_for_prim(stage, &subset.path);
        }
    }
    Ok(default)
}

fn mesh_normal(mesh: &ReadMesh, face: usize, corner: usize, point: usize) -> Result<Option<Vec3>> {
    mesh.normals.as_ref().map_or(Ok(None), |normals| {
        Ok(Some(Vec3::from_array(primvar_value(
            normals,
            interpolation_index(normals.interpolation, face, corner, point),
            "normal",
        )?)))
    })
}

fn mesh_uv(mesh: &ReadMesh, face: usize, corner: usize, point: usize) -> Result<Option<Vec2>> {
    mesh.uvs.as_ref().map_or(Ok(None), |uvs| {
        Ok(Some(Vec2::from_array(primvar_value(
            uvs,
            interpolation_index(uvs.interpolation, face, corner, point),
            "uv",
        )?)))
    })
}

fn interpolation_index(
    interpolation: Interpolation,
    face: usize,
    corner: usize,
    point: usize,
) -> usize {
    match interpolation {
        Interpolation::Constant => 0,
        Interpolation::Uniform => face,
        Interpolation::Varying | Interpolation::Vertex => point,
        Interpolation::FaceVarying => corner,
    }
}

fn primvar_value<T: Copy>(primvar: &Primvar<T>, element: usize, label: &str) -> Result<T> {
    if primvar.element_size != 1 {
        return Err(Error::Invalid(format!(
            "{label} primvar has unsupported elementSize {}",
            primvar.element_size
        )));
    }
    let index = if primvar.indices.is_empty() {
        element
    } else {
        usize::try_from(*primvar.indices.get(element).ok_or_else(|| {
            Error::Invalid(format!("{label} primvar index {element} is missing"))
        })?)?
    };
    primvar
        .values
        .get(index)
        .copied()
        .ok_or_else(|| Error::Invalid(format!("{label} primvar value {index} is missing")))
}
