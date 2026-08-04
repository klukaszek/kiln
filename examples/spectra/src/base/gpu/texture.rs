use std::collections::HashMap;

use kiln_rhi::{
    AddressMode, Device, FilterMode, Format, GpuAllocation, MemoryType, SampleCount, Sampler,
    SamplerDesc, SamplerHandle, StageFlags, Texture as RhiTexture, TextureDesc, TextureDimension,
    TextureHandle, TextureId as RhiTextureId, TextureUsage, gpu_struct,
};

use crate::base::renderer::Result;
use crate::base::scene::{ColorSpace, Image, Texture, WrapMode};

gpu_struct! {
    pub(crate) struct GpuTextureBinding {
        image: TextureHandle,
        sampler: SamplerHandle,
    }
}

pub(crate) struct TextureResources {
    images: Vec<GpuImage>,
    samplers: Vec<Sampler>,
    bindings: Vec<GpuTextureBinding>,
}

impl TextureResources {
    pub(crate) fn upload(device: &Device, images: &[Image], textures: &[Texture]) -> Result<Self> {
        let mut upload = TextureUpload::new(device);
        for image in images {
            upload.add_image(image)?;
        }

        let mut sampler_ids = HashMap::new();
        let mut bindings = Vec::with_capacity(textures.len());
        for texture in textures {
            let key = (texture.wrap_u, texture.wrap_v);
            let sampler = match sampler_ids.get(&key) {
                Some(&sampler) => sampler,
                None => {
                    let sampler = upload.add_sampler(texture.wrap_u, texture.wrap_v)?;
                    sampler_ids.insert(key, sampler);
                    sampler
                }
            };
            bindings.push(GpuTextureBinding {
                image: upload.images[texture.image.0].handle,
                sampler: upload
                    .device
                    .bindless_sampler_handle(upload.samplers[sampler].id()),
            });
        }
        upload.finish(bindings)
    }

    pub(crate) fn bindings(&self) -> &[GpuTextureBinding] {
        &self.bindings
    }

    pub(crate) fn destroy(self, device: &Device) {
        for image in self.images {
            image.destroy(device);
        }
        for sampler in self.samplers {
            device.destroy_sampler(sampler);
        }
    }
}

struct GpuImage {
    memory: GpuAllocation,
    texture: RhiTexture,
    view: RhiTextureId,
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
            format: match image.color_space {
                ColorSpace::Linear => Format::R8G8B8A8Unorm,
                ColorSpace::Srgb => Format::R8G8B8A8Srgb,
            },
            dimension: TextureDimension::D2,
            sample_count: SampleCount::S1,
            usage: TextureUsage::SAMPLED | TextureUsage::TRANSFER_DST,
            label: Some(image.name.clone()),
        };
        let size = device.texture_size_align(&desc)?;
        let memory = device.malloc_aligned(size.size, size.align, MemoryType::GpuOnly)?;
        let texture = match device.create_texture(&desc, memory.gpu()) {
            Ok(texture) => texture,
            Err(error) => {
                device.free(memory);
                return Err(error.into());
            }
        };
        let view = match device.create_sampled_view(&texture, &Default::default()) {
            Ok(view) => view,
            Err(error) => {
                device.destroy_texture(texture);
                device.free(memory);
                return Err(error.into());
            }
        };
        Ok(Self {
            handle: device.bindless_texture_handle(view),
            memory,
            texture,
            view,
        })
    }

    fn destroy(self, device: &Device) {
        device.destroy_texture_view(self.view);
        device.destroy_texture(self.texture);
        device.free(self.memory);
    }
}

struct TextureUpload<'a> {
    device: &'a Device,
    images: Vec<GpuImage>,
    staging: Vec<GpuAllocation>,
    samplers: Vec<Sampler>,
}

impl<'a> TextureUpload<'a> {
    fn new(device: &'a Device) -> Self {
        Self {
            device,
            images: Vec::new(),
            staging: Vec::new(),
            samplers: Vec::new(),
        }
    }

    fn add_image(&mut self, image: &Image) -> Result<()> {
        let gpu_image = GpuImage::allocate(self.device, image)?;
        let staging = match self.device.upload_slice(&image.rgba8) {
            Ok(staging) => staging,
            Err(error) => {
                gpu_image.destroy(self.device);
                return Err(error.into());
            }
        };
        self.images.push(gpu_image);
        self.staging.push(staging);
        Ok(())
    }

    fn add_sampler(&mut self, wrap_u: WrapMode, wrap_v: WrapMode) -> Result<usize> {
        let sampler = self.device.create_sampler(&SamplerDesc {
            min_filter: FilterMode::Linear,
            mag_filter: FilterMode::Linear,
            mip_filter: FilterMode::Linear,
            address_u: address_mode(wrap_u),
            address_v: address_mode(wrap_v),
            address_w: AddressMode::ClampToEdge,
            label: Some("spectra-texture-sampler".into()),
            ..Default::default()
        })?;
        let id = self.samplers.len();
        self.samplers.push(sampler);
        Ok(id)
    }

    fn finish(mut self, bindings: Vec<GpuTextureBinding>) -> Result<TextureResources> {
        if !self.images.is_empty() {
            let mut commands = self.device.create_command_buffer()?;
            for (image, staging) in self.images.iter().zip(&self.staging) {
                commands.copy_to_texture(image.memory.gpu(), staging.gpu(), &image.texture);
            }
            commands.barrier(StageFlags::TRANSFER, StageFlags::ALL_COMMANDS);
            commands.end();
            self.device.queue().submit(commands)?;
            self.device.queue().wait_idle();
        }
        for staging in self.staging.drain(..) {
            self.device.free(staging);
        }
        Ok(TextureResources {
            images: std::mem::take(&mut self.images),
            samplers: std::mem::take(&mut self.samplers),
            bindings,
        })
    }
}

impl Drop for TextureUpload<'_> {
    fn drop(&mut self) {
        for image in self.images.drain(..) {
            image.destroy(self.device);
        }
        for staging in self.staging.drain(..) {
            self.device.free(staging);
        }
        for sampler in self.samplers.drain(..) {
            self.device.destroy_sampler(sampler);
        }
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
