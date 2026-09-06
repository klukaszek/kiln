//! Vulkan sampler creation and bindless registration.

use ash::vk;

use super::device::{VulkanDevice, address_mode_to_vk, compare_op_to_vk};
use super::queue::VulkanRetiredResource;
use crate::error::{RhiError, RhiResult};
use crate::queue::QueueInner;
use crate::sampler::{Sampler, SamplerDesc};
use crate::types::{FilterMode, SamplerHandle, SamplerId};

impl VulkanDevice {
    fn allocate_sampler_id(&self) -> RhiResult<SamplerId> {
        if let Some(id) = self.free_sampler_ids.borrow_mut().pop() {
            return Ok(id);
        }
        let mut next = self.next_sampler_id.borrow_mut();
        let id = SamplerId(*next);
        *next = next
            .checked_add(1)
            .ok_or_else(|| RhiError::Backend("Vulkan sampler ID space exhausted".into()))?;
        Ok(id)
    }

    pub fn create_sampler(&self, desc: &SamplerDesc) -> RhiResult<Sampler> {
        let mag_filter = match desc.mag_filter {
            FilterMode::Nearest => vk::Filter::NEAREST,
            FilterMode::Linear => vk::Filter::LINEAR,
        };
        let min_filter = match desc.min_filter {
            FilterMode::Nearest => vk::Filter::NEAREST,
            FilterMode::Linear => vk::Filter::LINEAR,
        };
        let mip_mode = match desc.mip_filter {
            FilterMode::Nearest => vk::SamplerMipmapMode::NEAREST,
            FilterMode::Linear => vk::SamplerMipmapMode::LINEAR,
        };
        let address_u = address_mode_to_vk(desc.address_u);
        let address_v = address_mode_to_vk(desc.address_v);
        let address_w = address_mode_to_vk(desc.address_w);

        let mut sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(mag_filter)
            .min_filter(min_filter)
            .mipmap_mode(mip_mode)
            .address_mode_u(address_u)
            .address_mode_v(address_v)
            .address_mode_w(address_w)
            .mip_lod_bias(desc.mip_lod_bias)
            .min_lod(desc.min_lod)
            .max_lod(desc.max_lod);

        if let Some(max_aniso) = desc.max_anisotropy {
            sampler_info = sampler_info
                .anisotropy_enable(true)
                .max_anisotropy(max_aniso);
        }

        if let Some(compare) = desc.compare {
            sampler_info = sampler_info
                .compare_enable(true)
                .compare_op(compare_op_to_vk(compare));
        }

        let sampler = unsafe {
            self.device
                .create_sampler(&sampler_info, None)
                .map_err(|e| RhiError::Backend(format!("Sampler creation: {e}")))?
        };

        let id = match self.allocate_sampler_id() {
            Ok(id) => id,
            Err(err) => {
                unsafe { self.device.destroy_sampler(sampler, None) };
                return Err(err);
            }
        };

        if let Err(err) = self.write_sampler_descriptor(id, sampler) {
            unsafe { self.device.destroy_sampler(sampler, None) };
            self.recycle_sampler_id(id);
            return Err(err);
        }

        let idx = id.0 as usize;
        let mut samplers = self.samplers.borrow_mut();
        if samplers.len() <= idx {
            samplers.resize_with(idx + 1, || None);
        }
        samplers[idx] = Some(sampler);

        Ok(Sampler {
            id,
            handle: SamplerHandle::from_raw(id.0 as u64),
            _owner: None,
        })
    }

    pub fn destroy_sampler(&self, sampler: Sampler) {
        let sampler_id = sampler.id();
        let retired = self
            .samplers
            .borrow_mut()
            .get_mut(sampler_id.0 as usize)
            .and_then(Option::take);
        if let Some(sampler) = retired {
            backend_expect!(&self.queue.inner, QueueInner::Vulkan).release_resource(
                VulkanRetiredResource::Sampler {
                    id: sampler_id,
                    sampler,
                },
            );
        }
    }

    pub(crate) fn write_sampler_descriptor(
        &self,
        id: SamplerId,
        sampler: vk::Sampler,
    ) -> RhiResult<()> {
        let heap = &self.descriptor_buffer_heap;
        let loader = &self.descriptor_buffer_loader;

        let offset = heap.sampler_offset + (id.0 as u64) * heap.sampler_stride;
        if offset + heap.sampler_stride > heap.size {
            return Err(RhiError::Backend("Sampler descriptor heap overflow".into()));
        }

        let get_info = vk::DescriptorGetInfoEXT::default()
            .ty(vk::DescriptorType::SAMPLER)
            .data(vk::DescriptorDataEXT {
                p_sampler: &sampler,
            });

        unsafe {
            let dst = std::slice::from_raw_parts_mut(
                heap.mapped_ptr.add(offset as usize),
                heap.sampler_stride as usize,
            );
            loader.get_descriptor(&get_info, dst);
        }

        Ok(())
    }
}
