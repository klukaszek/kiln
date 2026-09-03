use std::collections::HashMap;

use kiln_rhi::{
    AddressMode, Allocation, Device, FilterMode, Format, MemoryType, SampleCount, Sampler,
    SamplerDesc, SamplerHandle, Texture as RhiTexture, TextureDesc, TextureDimension,
    TextureHandle, TextureUsage, gpu_struct,
};

use crate::render::Result;
use crate::scene::{Channels, ColorSpace, Image, Texture, WrapMode};

use super::super::upload::GpuUploadBatch;

gpu_struct! {
    pub(crate) struct GpuTextureBinding {
        image: TextureHandle,
        sampler: SamplerHandle,
    }
}

/// Every scene image on the device, with one sampler per distinct wrap mode and the handle pairs
/// the shader indexes by texture id.
pub(super) struct Textures {
    images: Vec<GpuImage>,
    samplers: Vec<Sampler>,
    bindings: Vec<GpuTextureBinding>,
}

impl Textures {
    /// Allocate each image and stage its contents into `batch`, which the caller submits. Sharing
    /// the batch with the scene buffers keeps the whole upload to one queue stall.
    pub(super) fn upload(
        device: &Device,
        batch: &mut GpuUploadBatch<'_>,
        images: &[Image],
        textures: &[Texture],
    ) -> Result<Self> {
        let mut gpu_images = Vec::with_capacity(images.len());
        for image in images {
            let gpu_image = GpuImage::allocate(device, image)?;
            batch.upload_texture(&image.texels, &gpu_image.texture)?;
            gpu_images.push(gpu_image);
        }

        let mut samplers: Vec<Sampler> = Vec::new();
        let mut by_wrap: HashMap<(WrapMode, WrapMode), usize> = HashMap::new();
        let mut bindings = Vec::with_capacity(textures.len());
        for texture in textures {
            let key = (texture.wrap_u, texture.wrap_v);
            let slot = match by_wrap.get(&key) {
                Some(&slot) => slot,
                None => {
                    samplers.push(sampler(device, key.0, key.1)?);
                    by_wrap.insert(key, samplers.len() - 1);
                    samplers.len() - 1
                }
            };
            bindings.push(GpuTextureBinding {
                image: gpu_images[texture.image.0].handle,
                sampler: samplers[slot].gpu(),
            });
        }

        Ok(Self {
            images: gpu_images,
            samplers,
            bindings,
        })
    }

    pub(super) fn bindings(&self) -> &[GpuTextureBinding] {
        &self.bindings
    }

    pub(super) fn destroy(self, device: &Device) {
        for image in self.images {
            image.destroy(device);
        }
        for sampler in self.samplers {
            device.destroy(sampler);
        }
    }
}

fn sampler(device: &Device, wrap_u: WrapMode, wrap_v: WrapMode) -> Result<Sampler> {
    Ok(device.create_sampler(&SamplerDesc {
        min_filter: FilterMode::Linear,
        mag_filter: FilterMode::Linear,
        mip_filter: FilterMode::Linear,
        address_u: address_mode(wrap_u),
        address_v: address_mode(wrap_v),
        address_w: AddressMode::ClampToEdge,
        label: Some("spectra-texture-sampler".into()),
        ..Default::default()
    })?)
}

struct GpuImage {
    memory: Allocation,
    texture: RhiTexture,
    handle: TextureHandle,
}

impl GpuImage {
    fn allocate(device: &Device, image: &Image) -> Result<Self> {
        let desc = TextureDesc {
            width: image.width,
            height: image.height,
            depth: 1,
            mip_levels: 1,
            array_layers: 1,
            format: match (image.channels, image.color_space) {
                (Channels::One, _) => Format::R8Unorm,
                (Channels::Four, ColorSpace::Linear) => Format::R8G8B8A8Unorm,
                (Channels::Four, ColorSpace::Srgb) => Format::R8G8B8A8Srgb,
            },
            dimension: TextureDimension::D2,
            sample_count: SampleCount::S1,
            usage: TextureUsage::SAMPLED | TextureUsage::TRANSFER_DST,
            label: Some(image.name.clone()),
        };
        let size = device.texture_size_align(&desc)?;
        let memory = device.allocate_aligned(size.size, size.align, MemoryType::GpuOnly)?;
        let texture = match device.create_texture(&desc, memory.gpu()) {
            Ok(texture) => texture,
            Err(error) => {
                device.destroy(memory);
                return Err(error.into());
            }
        };
        Ok(Self {
            handle: texture.gpu(),
            memory,
            texture,
        })
    }

    fn destroy(self, device: &Device) {
        device.destroy(self.texture);
        device.destroy(self.memory);
    }
}

fn address_mode(mode: WrapMode) -> AddressMode {
    match mode {
        WrapMode::Black => AddressMode::ClampToBorder,
        WrapMode::Clamp => AddressMode::ClampToEdge,
        WrapMode::Mirror => AddressMode::MirroredRepeat,
        WrapMode::Repeat => AddressMode::Repeat,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_texture_binding_is_two_handles() {
        assert_eq!(size_of::<GpuTextureBinding>(), 16);
    }
}
