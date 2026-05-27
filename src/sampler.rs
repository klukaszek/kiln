use crate::types::{AddressMode, CompareOp, FilterMode, SamplerId};

/// Description for creating a sampler.
#[derive(Clone, Debug)]
pub struct SamplerDesc {
    pub min_filter: FilterMode,
    pub mag_filter: FilterMode,
    pub mip_filter: FilterMode,
    pub address_u: AddressMode,
    pub address_v: AddressMode,
    pub address_w: AddressMode,
    pub mip_lod_bias: f32,
    pub max_anisotropy: Option<f32>,
    pub compare: Option<CompareOp>,
    pub min_lod: f32,
    pub max_lod: f32,
    pub label: Option<String>,
}

impl Default for SamplerDesc {
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
}

impl Sampler {
    /// Get the SamplerId for use in shaders.
    pub fn id(&self) -> SamplerId {
        self.id
    }
}
