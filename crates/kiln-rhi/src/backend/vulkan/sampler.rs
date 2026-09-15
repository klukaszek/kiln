//! Vulkan sampler creation and bindless registration.

use ash::vk;

use super::device::{VulkanDevice, address_mode_to_vk, compare_op_to_vk};
use super::queue::VulkanRetiredResource;
use crate::error::{RhiError, RhiResult};
use crate::queue::QueueInner;
use crate::sampler::{Sampler, SamplerDesc};
use crate::types::{FilterMode, MAX_BINDLESS_SAMPLERS, SamplerHandle, SamplerId};

impl VulkanDevice {
    fn allocate_sampler_id(&self) -> RhiResult<SamplerId> {
        if let Some(id) = self.free_sampler_ids.borrow_mut().pop() {
            return Ok(id);
        }
        let mut next = self.next_sampler_id.borrow_mut();
        if *next >= MAX_BINDLESS_SAMPLERS {
            return Err(RhiError::Backend(
                "Vulkan bindless sampler heap exhausted".into(),
            ));
        }
        let id = SamplerId(*next);
        *next += 1;
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

        let id = self.allocate_sampler_id()?;
        if let Err(err) = self.write_sampler_descriptor(id, &sampler_info) {
            self.recycle_sampler_id(id);
            return Err(err);
        }

        Ok(Sampler {
            id,
            handle: SamplerHandle::from_raw(id.0 as u64),
            _owner: None,
        })
    }

    pub fn destroy_sampler(&self, sampler: Sampler) {
        backend_expect!(&self.queue.inner, QueueInner::Vulkan)
            .release_resource(VulkanRetiredResource::Sampler { id: sampler.id() });
    }

    /// Write one sampler descriptor into the sampler heap at slot `id`. The descriptor is built
    /// straight from the create-info; there is no `VkSampler` object under descriptor heaps.
    pub(crate) fn write_sampler_descriptor(
        &self,
        id: SamplerId,
        info: &vk::SamplerCreateInfo<'_>,
    ) -> RhiResult<()> {
        let heap = &self.descriptor_heaps.sampler;
        let slot = heap.slot(id.0);

        unsafe {
            let dst = std::slice::from_raw_parts_mut(heap.mapped_ptr.add(slot.start), slot.len());
            self.descriptor_heap_loader
                .write_sampler_descriptors(
                    std::slice::from_ref(info),
                    &[vk::HostAddressRangeEXT::default().address(dst)],
                )
                .map_err(|e| RhiError::Backend(format!("write sampler descriptor: {e}")))
        }
    }
}
