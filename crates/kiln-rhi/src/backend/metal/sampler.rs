//! Metal sampler creation and bindless registration.

use objc2_metal::{MTLDevice, MTLSamplerDescriptor, MTLSamplerState};

use super::device::{MetalDevice, MetalRetiredResource};
use crate::error::{RhiError, RhiResult};
use crate::sampler::{Sampler, SamplerDesc};
use crate::types::{AddressMode, FilterMode, SamplerHandle, SamplerId};

fn filter_to_mtl(f: FilterMode) -> objc2_metal::MTLSamplerMinMagFilter {
    match f {
        FilterMode::Nearest => objc2_metal::MTLSamplerMinMagFilter::Nearest,
        FilterMode::Linear => objc2_metal::MTLSamplerMinMagFilter::Linear,
    }
}

fn mip_filter_to_mtl(f: FilterMode) -> objc2_metal::MTLSamplerMipFilter {
    match f {
        FilterMode::Nearest => objc2_metal::MTLSamplerMipFilter::Nearest,
        FilterMode::Linear => objc2_metal::MTLSamplerMipFilter::Linear,
    }
}

fn address_to_mtl(a: AddressMode) -> objc2_metal::MTLSamplerAddressMode {
    match a {
        AddressMode::Repeat => objc2_metal::MTLSamplerAddressMode::Repeat,
        AddressMode::MirroredRepeat => objc2_metal::MTLSamplerAddressMode::MirrorRepeat,
        AddressMode::ClampToEdge => objc2_metal::MTLSamplerAddressMode::ClampToEdge,
        AddressMode::ClampToBorder => objc2_metal::MTLSamplerAddressMode::ClampToBorderColor,
    }
}

impl MetalDevice {
    pub fn create_sampler(&self, desc: &SamplerDesc) -> RhiResult<Sampler> {
        let mtl_desc = MTLSamplerDescriptor::new();

        mtl_desc.setMinFilter(filter_to_mtl(desc.min_filter));
        mtl_desc.setMagFilter(filter_to_mtl(desc.mag_filter));
        mtl_desc.setMipFilter(mip_filter_to_mtl(desc.mip_filter));
        mtl_desc.setSAddressMode(address_to_mtl(desc.address_u));
        mtl_desc.setTAddressMode(address_to_mtl(desc.address_v));
        mtl_desc.setRAddressMode(address_to_mtl(desc.address_w));
        mtl_desc.setLodMinClamp(desc.min_lod);
        mtl_desc.setLodMaxClamp(desc.max_lod);

        if let Some(aniso) = desc.max_anisotropy {
            mtl_desc.setMaxAnisotropy(usize::from(aniso.get()));
        }

        if let Some(cmp) = desc.compare {
            mtl_desc.setCompareFunction(super::pipeline::compare_op_to_mtl(cmp));
        }
        mtl_desc.setSupportArgumentBuffers(true);

        let sampler = self
            .shared
            .device
            .newSamplerStateWithDescriptor(&mtl_desc)
            .ok_or_else(|| RhiError::Backend("Failed to create Metal sampler".into()))?;

        let resource_id = sampler.gpuResourceID().to_raw();
        let id = SamplerId(self.shared.samplers.allocate_id()?);
        self.shared.samplers.insert(id.0, sampler, resource_id);

        Ok(Sampler {
            id,
            handle: SamplerHandle::from_raw(resource_id),
            _owner: None,
        })
    }

    pub fn destroy_sampler(&self, sampler: Sampler) {
        let sampler_id = sampler.id();
        if let Some(sampler) = self.shared.samplers.take(sampler_id.0) {
            self.queue.release_resource(MetalRetiredResource::Sampler {
                id: sampler_id,
                sampler,
            });
        }
    }
}
