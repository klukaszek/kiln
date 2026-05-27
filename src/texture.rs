use crate::types::{Format, GpuAddress, SampleCount, TextureDimension, TextureId};

/// Sentinel for `GpuViewDesc::mip_count`: include all mip levels from `base_mip` to the last.
pub const ALL_MIPS: u8 = 0xFF;
/// Sentinel for `GpuViewDesc::layer_count`: include all array layers from `base_layer` to the last.
pub const ALL_LAYERS: u16 = 0xFFFF;

bitflags::bitflags! {
    /// Texture usage flags.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct TextureUsage: u32 {
        const SAMPLED           = 0x01;
        const STORAGE           = 0x02;
        const COLOR_ATTACHMENT  = 0x04;
        const DEPTH_STENCIL_ATTACHMENT = 0x08;
        const TRANSFER_SRC      = 0x10;
        const TRANSFER_DST      = 0x20;
    }
}

/// Description for creating a texture.
#[derive(Clone, Debug)]
pub struct TextureDesc {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub format: Format,
    pub dimension: TextureDimension,
    pub sample_count: SampleCount,
    pub usage: TextureUsage,
    pub label: Option<String>,
}

impl Default for TextureDesc {
    fn default() -> Self {
        Self {
            width: 1,
            height: 1,
            depth: 1,
            mip_levels: 1,
            array_layers: 1,
            format: Format::R8G8B8A8Unorm,
            dimension: TextureDimension::D2,
            sample_count: SampleCount::S1,
            usage: TextureUsage::SAMPLED | TextureUsage::TRANSFER_DST,
            label: None,
        }
    }
}

/// Size and alignment required for a placed texture allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureSizeAlign {
    pub size: u64,
    pub align: u64,
}

/// Opaque texture object.
/// The image is backed by caller-owned GPU memory at `gpu_address`.
pub struct Texture {
    pub(crate) id: TextureId,
    pub(crate) gpu_address: GpuAddress,
    pub(crate) desc: TextureDesc,
}

impl Texture {
    /// Get the TextureId for use in shaders (bindless index).
    pub fn id(&self) -> TextureId {
        self.id
    }

    /// Get the raw GPU allocation address used to create this texture.
    pub fn gpu_address(&self) -> GpuAddress {
        self.gpu_address
    }

    /// Get the texture description.
    pub fn desc(&self) -> &TextureDesc {
        &self.desc
    }
}

/// View descriptor for creating a non-default view of an existing texture.
///
/// Matches Aaltonen's `GpuViewDesc`. Used with `device.texture_view_descriptor()`
/// (sampled/SRV) and `device.rw_texture_view_descriptor()` (storage/UAV).
///
/// `mip_count = ALL_MIPS` → all mips from `base_mip` to the end.
/// `layer_count = ALL_LAYERS` → all layers from `base_layer` to the end.
/// `format = None` → same format as the source texture.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GpuViewDesc {
    /// Format override. `None` = same as source texture.
    pub format: Option<Format>,
    /// First mip level included in the view.
    pub base_mip: u8,
    /// Number of mip levels. Use `ALL_MIPS` for all remaining levels.
    pub mip_count: u8,
    /// First array layer included in the view.
    pub base_layer: u16,
    /// Number of array layers. Use `ALL_LAYERS` for all remaining layers.
    pub layer_count: u16,
}

impl Default for GpuViewDesc {
    fn default() -> Self {
        Self {
            format: None,
            base_mip: 0,
            mip_count: ALL_MIPS,
            base_layer: 0,
            layer_count: ALL_LAYERS,
        }
    }
}

/// Bytes per pixel for uncompressed formats.
pub fn bytes_per_pixel(format: Format) -> Option<usize> {
    match format {
        Format::R8Unorm => Some(1),
        Format::R8G8Unorm => Some(2),
        Format::R8G8B8A8Unorm | Format::R8G8B8A8Srgb => Some(4),
        Format::B8G8R8A8Unorm | Format::B8G8R8A8Srgb => Some(4),
        Format::R16Float => Some(2),
        Format::R16G16Float => Some(4),
        Format::R16G16B16A16Float => Some(8),
        Format::R32Float => Some(4),
        Format::R32G32Float => Some(8),
        Format::R32G32B32Float => Some(12),
        Format::R32G32B32A32Float => Some(16),
        Format::R10G10B10A2Unorm => Some(4),
        Format::R11G11B10Float => Some(4),
        _ => None,
    }
}
