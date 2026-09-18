//! Texture creation, view descriptors, and bindless heap registration.

use crate::types::{Format, SampleCount, TextureDimension, TextureHandle, TextureId};

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
pub struct TextureDesc<'a> {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mip_levels: u32,
    /// Number of whole images, never faces: slices for `D2Array`, cubes for `Cube` (always 1)
    /// and `CubeArray`. Must be 1 for every other dimension.
    pub array_layers: u32,
    pub format: Format,
    pub dimension: TextureDimension,
    pub sample_count: SampleCount,
    pub usage: TextureUsage,
    pub label: Option<&'a str>,
}

impl TextureDesc<'_> {
    /// Reject descriptors whose dimension and counts disagree, before either backend sees them;
    /// the two APIs express these differently and would fail in backend-specific ways.
    /// Only the rule the native APIs cannot state: `TextureDimension` is kiln's type, and a
    /// single-image dimension carrying several layers builds an image and a default view that
    /// disagree about the texture's type -- differently on each backend. Extents, mip counts and
    /// layer counts are VUIDs; the validation layer owns those.
    pub(crate) fn validate(&self) -> crate::RhiResult<()> {
        if self.array_layers > 1 && !self.dimension.is_array() {
            return Err(crate::RhiError::TextureCreation(
                format!(
                    "{:?} holds a single image but array_layers is {}; use D2Array or CubeArray",
                    self.dimension, self.array_layers
                )
                .into(),
            ));
        }
        Ok(())
    }
}

impl Default for TextureDesc<'_> {
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

/// One mip level and array layer of a texture, optionally a sub-box within it. Copy entry points
/// take `impl Into<Option<Self>>`; `None` is the whole of mip 0, layer 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TextureRegion {
    pub mip: u32,
    /// Array layer, or cube face for a cube texture.
    pub layer: u32,
    /// Texel offset within the mip. `z` indexes depth slices of a 3D texture.
    pub origin: [u32; 3],
    /// Texel extent measured from `origin`. `None` runs to the edge of the mip.
    pub extent: Option<[u32; 3]>,
}

impl TextureRegion {
    pub(crate) fn resolve(self, desc: &TextureDesc<'_>) -> ResolvedRegion {
        let at_mip = |extent: u32| (extent >> self.mip).max(1);
        let full = [
            at_mip(desc.width),
            at_mip(desc.height),
            match desc.dimension {
                TextureDimension::D3 => at_mip(desc.depth),
                _ => 1,
            },
        ];
        let extent = self.extent.unwrap_or_else(|| {
            [
                full[0].saturating_sub(self.origin[0]),
                full[1].saturating_sub(self.origin[1]),
                full[2].saturating_sub(self.origin[2]),
            ]
        });
        ResolvedRegion {
            mip: self.mip,
            layer: self.layer,
            origin: self.origin,
            extent,
        }
    }
}

/// A [`TextureRegion`] with its extent filled in against a concrete texture.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ResolvedRegion {
    pub(crate) mip: u32,
    pub(crate) layer: u32,
    pub(crate) origin: [u32; 3],
    pub(crate) extent: [u32; 3],
}

impl ResolvedRegion {
    /// Tightly packed linear layout of this region: `(bytes per row, bytes per image)`.
    pub(crate) fn linear_strides(&self, bytes_per_texel: usize) -> (usize, usize) {
        let bytes_per_row = self.extent[0] as usize * bytes_per_texel;
        (bytes_per_row, bytes_per_row * self.extent[1] as usize)
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
    pub(crate) handle: TextureHandle,
    pub(crate) views: Vec<TextureId>,
    /// The descriptor it was created with. The label is dropped: nothing reads it back, and
    /// keeping it would tie `Texture` to the caller's string.
    pub(crate) desc: TextureDesc<'static>,
    pub(crate) _owner: Option<std::rc::Rc<crate::device::DeviceInner>>,
}

impl Texture {
    pub(crate) fn id(&self) -> TextureId {
        self.id
    }

    /// Opaque shader handle for the texture's full view, as a **sampled** texture.
    ///
    /// One slot holds one descriptor, so a `SAMPLED | STORAGE` texture resolves here as sampled;
    /// get its storage form from [`Device::create_texture_view`](crate::Device::create_texture_view)
    /// with [`ViewKind::Storage`]. A storage-only texture returns its storage handle.
    pub fn gpu(&self) -> TextureHandle {
        self.handle
    }

    /// Render-target reference for this texture.
    pub fn target(&self) -> crate::command::RenderTarget {
        crate::command::RenderTarget::texture(self.id)
    }

    /// The descriptor this texture was created with, its label cleared.
    pub fn desc(&self) -> &TextureDesc<'static> {
        &self.desc
    }
}

/// How a [`Device::create_texture_view`](crate::Device::create_texture_view) result is read in
/// shaders, fixing the usage the source must carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ViewKind {
    /// Read-only, sampled through a `SamplerState`.
    Sampled,
    /// Read-write storage image.
    Storage,
}

/// A non-default sampled or storage view. `Default` covers the whole texture; the counts are
/// `None` rather than a sentinel, since `0xFF` is indistinguishable from a real count of 255.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TextureViewDesc {
    /// Format override. `None` = same as source texture.
    pub format: Option<Format>,
    /// First mip level included in the view.
    pub base_mip: u8,
    /// Number of mip levels. `None` = all levels from `base_mip` on.
    pub mip_count: Option<u8>,
    /// First array layer included in the view.
    pub base_layer: u16,
    /// Number of array layers. `None` = all layers from `base_layer` on.
    pub layer_count: Option<u16>,
}

impl TextureViewDesc {
    /// Mip levels this view covers, given the source's total.
    pub(crate) fn resolved_mip_count(&self, source_mips: u32) -> u32 {
        match self.mip_count {
            Some(count) => u32::from(count),
            None => source_mips.saturating_sub(u32::from(self.base_mip)),
        }
    }

    /// Array layers (or cube faces) this view covers, given the source's total.
    pub(crate) fn resolved_layer_count(&self, source_layers: u32) -> u32 {
        match self.layer_count {
            Some(count) => u32::from(count),
            None => source_layers.saturating_sub(u32::from(self.base_layer)),
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(dimension: TextureDimension, array_layers: u32) -> TextureDesc<'static> {
        TextureDesc {
            dimension,
            array_layers,
            ..Default::default()
        }
    }

    #[test]
    fn validate_rejects_descs_the_backends_would_disagree_about() {
        // A single-image dimension with several layers built an image and a default view that
        // disagreed about the texture's type, differently on each backend.
        assert!(desc(TextureDimension::D2, 4).validate().is_err());
        assert!(desc(TextureDimension::D3, 2).validate().is_err());
        assert!(desc(TextureDimension::Cube, 2).validate().is_err());
        assert!(desc(TextureDimension::D2Array, 4).validate().is_ok());
        assert!(desc(TextureDimension::CubeArray, 4).validate().is_ok());
        assert!(desc(TextureDimension::D2, 1).validate().is_ok());
    }
}
