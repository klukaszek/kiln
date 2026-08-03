use glam::{Vec2, Vec4};
use kiln_rhi::{Device, gpu_struct};

use crate::render::{self, Error, GpuArray, GpuTextureBinding, GpuUploadBatch, TextureResources};
use crate::scene::{Scene, Surface};

pub(super) const NO_TEXTURE: u32 = u32::MAX;

gpu_struct! {
    pub(super) struct RasterVertex {
        pos: Vec4,
        normal: Vec4,
        uv: Vec2,
        material_id: u32,
        _pad: u32,
    }
}

gpu_struct! {
    pub(super) struct GpuPreviewMaterial {
        color: Vec4,
        texture_id: u32,
        _pad: [f32; 3],
    }
}

pub(super) struct PreviewScene {
    pub(super) vertices: GpuArray<RasterVertex>,
    pub(super) materials: GpuArray<GpuPreviewMaterial>,
    pub(super) texture_bindings: GpuArray<GpuTextureBinding>,
    pub(super) triangle_count: u32,
    textures: TextureResources,
}

impl PreviewScene {
    pub(super) fn build(device: &Device, scene: &Scene) -> render::Result<Self> {
        if scene.triangle_count() > (u32::MAX as usize) / 3 {
            return Err(Error::Capacity("preview geometry exceeds u32 indexing"));
        }

        let textures = TextureResources::upload(device, &scene.images, &scene.textures)?;
        let texture_bindings = textures.bindings();
        let materials = scene
            .materials
            .iter()
            .map(|material| {
                let texture_id = match material.surface {
                    Surface::Principled(surface) => surface
                        .base_color_texture
                        .map(|id| id.0 as u32)
                        .unwrap_or(NO_TEXTURE),
                    _ => NO_TEXTURE,
                };
                GpuPreviewMaterial {
                    color: material.preview_color().extend(1.0),
                    texture_id,
                    _pad: [0.0; 3],
                }
            })
            .collect::<Vec<_>>();

        let mut vertices = Vec::with_capacity(scene.triangle_count() * 3);
        for triangle in &scene.geometry.triangles {
            for vertex in &triangle.vertices {
                vertices.push(RasterVertex {
                    pos: vertex.position.extend(1.0),
                    normal: vertex.normal.unwrap_or(glam::Vec3::Y).extend(0.0),
                    uv: vertex.uv.unwrap_or_default(),
                    material_id: triangle.material.0 as u32,
                    _pad: 0,
                });
            }
        }

        let mut upload = GpuUploadBatch::new(device);
        upload.upload(&vertices)?;
        upload.upload(&materials)?;
        upload.upload(texture_bindings)?;
        let [vertices_gpu, materials_gpu, texture_bindings_gpu] = match upload.finish() {
            Ok(uploaded) => uploaded,
            Err(error) => {
                textures.destroy(device);
                return Err(error);
            }
        };
        Ok(Self {
            vertices: GpuArray::new(vertices_gpu, vertices.len()),
            materials: GpuArray::new(materials_gpu, materials.len()),
            texture_bindings: GpuArray::new(texture_bindings_gpu, texture_bindings.len()),
            triangle_count: scene.triangle_count() as u32,
            textures,
        })
    }

    pub(super) fn destroy(self, device: &Device) {
        self.vertices.destroy(device);
        self.materials.destroy(device);
        self.texture_bindings.destroy(device);
        self.textures.destroy(device);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raster_vertex_stays_compact() {
        assert_eq!(size_of::<RasterVertex>(), 48);
        assert_eq!(size_of::<GpuPreviewMaterial>(), 32);
    }
}
