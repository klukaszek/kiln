//! Metal sampler creation and bindless registration.

use objc2_metal::{MTLDevice, MTLSamplerDescriptor, MTLSamplerState};

use super::device::{METAL_BINDLESS_SAMPLER_CAPACITY, MetalDevice, MetalRetiredResource};
use crate::error::{RhiError, RhiResult};
use crate::queue::QueueInner;
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
    fn allocate_sampler_id(&self) -> RhiResult<SamplerId> {
        if let Some(id) = self.shared.free_sampler_ids.borrow_mut().pop() {
            return Ok(id);
        }
        let next = self.shared.samplers.borrow().len();
        if next >= METAL_BINDLESS_SAMPLER_CAPACITY {
            return Err(RhiError::Backend(
                "Metal bindless sampler heap exhausted".into(),
            ));
        }
        Ok(SamplerId(next as u32))
    }

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
            mtl_desc.setMaxAnisotropy(aniso as usize);
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

        let id = self.allocate_sampler_id()?;
        let idx = id.0 as usize;
        let mut samplers = self.shared.samplers.borrow_mut();
        if samplers.len() <= idx {
            samplers.resize_with(idx + 1, || None);
        }
        samplers[idx] = Some(sampler.clone());
        drop(samplers);
        Self::write_heap_slot(
            &self.shared.sampler_heap,
            idx,
            sampler.gpuResourceID().to_raw(),
        );

        Ok(Sampler {
            id,
            handle: SamplerHandle::from_raw(sampler.gpuResourceID().to_raw()),
            _owner: None,
        })
    }

    pub fn destroy_sampler(&self, sampler: Sampler) {
        let sampler_id = sampler.id();
        let retired = {
            let mut samplers = self.shared.samplers.borrow_mut();
            let idx = sampler_id.0 as usize;
            if idx < samplers.len() {
                samplers[idx].take()
            } else {
                None
            }
        };
        if let Some(sampler) = retired {
            backend_expect!(&self.rhi_queue.inner, QueueInner::Metal).release_resource(
                MetalRetiredResource::Sampler {
                    id: sampler_id,
                    sampler,
                },
            );
        }
    }
}
