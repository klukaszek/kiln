//! Texture creation, view descriptors, and bindless heap registration.

use crate::device::DeviceInner;
use crate::types::{Format, GpuPtr, SampleCount, TextureDimension, TextureHandle, TextureId};

/// Sentinel for `TextureViewDesc::mip_count`: include all remaining mip levels.
pub const ALL_MIPS: u8 = 0xFF;
/// Sentinel for `TextureViewDesc::layer_count`: include all remaining array layers.
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

/// A texture backed by caller-owned GPU memory.
pub struct Texture {
    pub(crate) id: TextureId,
    pub(crate) gpu_address: GpuPtr<u8>,
    pub(crate) handle: TextureHandle,
    pub(crate) views: Vec<TextureId>,
    pub(crate) desc: TextureDesc,
    pub(crate) _owner: Option<std::rc::Rc<crate::device::DeviceInner>>,
}

impl Texture {
    pub(crate) fn id(&self) -> TextureId {
        self.id
    }

    /// Opaque shader handle for the texture's full view.
    pub fn gpu(&self) -> TextureHandle {
        self.handle
    }

    /// Render-target reference for this texture.
    pub fn target(&self) -> crate::command::RenderTarget {
        crate::command::RenderTarget::texture(self.id)
    }

    /// Create a sampled subresource view. It is released with this texture.
    pub fn sampled_view(&mut self, view: &TextureViewDesc) -> crate::RhiResult<TextureHandle> {
        self.create_view(view, false).map(TextureHandle::from_raw)
    }

    /// Create a read-write subresource view. It is released with this texture.
    pub fn storage_view(&mut self, view: &TextureViewDesc) -> crate::RhiResult<TextureHandle> {
        self.create_view(view, true).map(TextureHandle::from_raw)
    }

    fn create_view(&mut self, view: &TextureViewDesc, storage: bool) -> crate::RhiResult<u64> {
        let usage = if storage {
            TextureUsage::STORAGE
        } else {
            TextureUsage::SAMPLED
        };
        crate::device::validate_texture_view(self, view, usage)?;
        let owner = self
            ._owner
            .clone()
            .ok_or_else(|| crate::RhiError::Backend("texture has no owning device".into()))?;
        let id = if storage {
            backend_dispatch!(owner.as_ref(), DeviceInner, d => d.create_storage_view(self, view))?
        } else {
            backend_dispatch!(owner.as_ref(), DeviceInner, d => d.create_sampled_view(self, view))?
        };
        let address = backend_dispatch!(owner.as_ref(), DeviceInner, d => d.texture_handle_raw(id));
        self.views.push(id);
        Ok(address)
    }

    /// Texture description.
    pub fn desc(&self) -> &TextureDesc {
        &self.desc
    }
}

/// A non-default sampled or storage view.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TextureViewDesc {
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

impl Default for TextureViewDesc {
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
        Format::R32G32B32A32Float => Some(16),
        Format::R10G10B10A2Unorm => Some(4),
        Format::R11G11B10Float => Some(4),
        _ => None,
    }
}
