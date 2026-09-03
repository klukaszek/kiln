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
        /// Allow format-reinterpreting views (see [`formats_are_view_compatible`]). Both
        /// backends need this at creation time, and it can rule out framebuffer compression.
        const FORMAT_VIEW       = 0x40;
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

    /// Subresource view; released with this texture. The source must carry `kind`'s usage.
    pub fn view(
        &mut self,
        kind: ViewKind,
        view: &TextureViewDesc,
    ) -> crate::RhiResult<TextureHandle> {
        self.create_view(view, kind).map(TextureHandle::from_raw)
    }

    fn create_view(&mut self, view: &TextureViewDesc, kind: ViewKind) -> crate::RhiResult<u64> {
        let storage = kind == ViewKind::Storage;
        crate::device::validate_texture_view(self, view, kind.required_usage())?;
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

    pub fn desc(&self) -> &TextureDesc {
        &self.desc
    }
}

/// How a [`Texture::view`] is read in shaders, fixing the usage the source must carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ViewKind {
    /// Read-only, sampled through a `SamplerState`.
    Sampled,
    /// Read-write storage image.
    Storage,
}

impl ViewKind {
    fn required_usage(self) -> TextureUsage {
        match self {
            ViewKind::Sampled => TextureUsage::SAMPLED,
            ViewKind::Storage => TextureUsage::STORAGE,
        }
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

/// Whether a view may reinterpret `a` as `b`: same channel layout and bit depth, differing only
/// in how the bits are read (`R8G8B8A8Unorm` / `R8G8B8A8Srgb`, `R32Float` / `R32Uint`).
///
/// The portable intersection, not the union: Vulkan would also allow swizzled pairs like
/// `R8G8B8A8Unorm` / `B8G8R8A8Unorm`, but Metal treats channel order as part of the format
/// family. Depth never reinterprets on either backend.
pub fn formats_are_view_compatible(a: Format, b: Format) -> bool {
    a == b || matches!((view_class(a), view_class(b)), (Some(x), Some(y)) if x == y)
}

/// Channel layout and bit depth, ignoring interpretation. `None` for depth/stencil.
fn view_class(format: Format) -> Option<u8> {
    Some(match format {
        Format::R8Unorm => 0,
        Format::R8G8Unorm => 1,
        Format::R8G8B8A8Unorm | Format::R8G8B8A8Srgb => 2,
        Format::B8G8R8A8Unorm | Format::B8G8R8A8Srgb => 3,
        Format::R16Float | Format::R16Uint => 4,
        Format::R16G16Float => 5,
        Format::R16G16B16A16Float => 6,
        Format::R32Float | Format::R32Uint => 7,
        Format::R32G32Float => 8,
        Format::R32G32B32A32Float => 9,
        Format::R10G10B10A2Unorm => 10,
        Format::R11G11B10Float => 11,
        Format::D16Unorm | Format::D32Float | Format::D24UnormS8Uint | Format::D32FloatS8Uint => {
            return None;
        }
    })
}

/// Bytes per pixel for colour formats. `None` for depth/stencil, which copy as separate aspects
/// and have no single texel size. Exhaustive on purpose: the copy paths panic on `None`, so a
/// catch-all arm here turns a new format into a copy failure.
pub fn bytes_per_pixel(format: Format) -> Option<usize> {
    Some(match format {
        Format::R8Unorm => 1,
        Format::R8G8Unorm => 2,
        Format::R8G8B8A8Unorm | Format::R8G8B8A8Srgb => 4,
        Format::B8G8R8A8Unorm | Format::B8G8R8A8Srgb => 4,
        Format::R16Float | Format::R16Uint => 2,
        Format::R16G16Float => 4,
        Format::R16G16B16A16Float => 8,
        Format::R32Float | Format::R32Uint => 4,
        Format::R32G32Float => 8,
        Format::R32G32B32A32Float => 16,
        Format::R10G10B10A2Unorm => 4,
        Format::R11G11B10Float => 4,
        Format::D16Unorm | Format::D32Float | Format::D24UnormS8Uint | Format::D32FloatS8Uint => {
            return None;
        }
    })
}
