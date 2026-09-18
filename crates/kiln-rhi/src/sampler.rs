//! Sampler creation and descriptor registration.

use crate::types::{AddressMode, CompareOp, FilterMode, SamplerHandle, SamplerId};
use std::num::NonZeroU8;

/// Description for creating a sampler.
#[derive(Clone, Debug)]
pub struct SamplerDesc<'a> {
    pub min_filter: FilterMode,
    pub mag_filter: FilterMode,
    pub mip_filter: FilterMode,
    pub address_u: AddressMode,
    pub address_v: AddressMode,
    pub address_w: AddressMode,
    pub mip_lod_bias: f32,
    /// Maximum anisotropy, `None` to disable. Both backends take an integer here: Metal's
    /// `setMaxAnisotropy` rejects 0, and Vulkan caps this at `maxSamplerAnisotropy` (16 on every
    /// current implementation), so a float would only add a way to pass NaN.
    pub max_anisotropy: Option<NonZeroU8>,
    pub compare: Option<CompareOp>,
    pub min_lod: f32,
    pub max_lod: f32,
    pub label: Option<&'a str>,
}

impl Default for SamplerDesc<'_> {
    fn default() -> Self {
        Self {
            min_filter: FilterMode::Linear,
            mag_filter: FilterMode::Linear,
            mip_filter: FilterMode::Linear,
            address_u: AddressMode::Repeat,
            address_v: AddressMode::Repeat,
            address_w: AddressMode::Repeat,
            mip_lod_bias: 0.0,
            max_anisotropy: None,
            compare: None,
            min_lod: 0.0,
            max_lod: 1000.0,
            label: None,
        }
    }
}

/// Opaque sampler object.
pub struct Sampler {
    pub(crate) id: SamplerId,
    pub(crate) handle: SamplerHandle,
    pub(crate) _owner: Option<std::rc::Rc<crate::device::DeviceInner>>,
}

impl Sampler {
    /// Opaque shader handle for this sampler.
    pub fn gpu(&self) -> SamplerHandle {
        self.handle
    }

    pub(crate) fn id(&self) -> SamplerId {
        self.id
    }
}
