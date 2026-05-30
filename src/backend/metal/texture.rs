use objc2_metal::MTLPixelFormat;

use crate::types::Format;

/// Convert RHI Format to MTLPixelFormat.
pub fn format_to_mtl(format: Format) -> MTLPixelFormat {
    match format {
        // Color
        Format::R8Unorm => MTLPixelFormat::R8Unorm,
        Format::R8G8Unorm => MTLPixelFormat::RG8Unorm,
        Format::R8G8B8A8Unorm => MTLPixelFormat::RGBA8Unorm,
        Format::R8G8B8A8Srgb => MTLPixelFormat::RGBA8Unorm_sRGB,
        Format::B8G8R8A8Unorm => MTLPixelFormat::BGRA8Unorm,
        Format::B8G8R8A8Srgb => MTLPixelFormat::BGRA8Unorm_sRGB,
        Format::R16Float => MTLPixelFormat::R16Float,
        Format::R16G16Float => MTLPixelFormat::RG16Float,
        Format::R16G16B16A16Float => MTLPixelFormat::RGBA16Float,
        Format::R32Float => MTLPixelFormat::R32Float,
        Format::R32G32Float => MTLPixelFormat::RG32Float,
        Format::R32G32B32A32Float => MTLPixelFormat::RGBA32Float,
        Format::R10G10B10A2Unorm => MTLPixelFormat::RGB10A2Unorm,
        Format::R11G11B10Float => MTLPixelFormat::RG11B10Float,
        // Depth
        Format::D16Unorm => MTLPixelFormat::Depth16Unorm,
        Format::D32Float => MTLPixelFormat::Depth32Float,
        Format::D24UnormS8Uint => MTLPixelFormat::Depth24Unorm_Stencil8,
        Format::D32FloatS8Uint => MTLPixelFormat::Depth32Float_Stencil8,
        // Index (not used as pixel format, but map for completeness)
        Format::R16Uint => MTLPixelFormat::R16Uint,
        Format::R32Uint => MTLPixelFormat::R32Uint,
    }
}

/// Convert MTLPixelFormat back to RHI Format — the faithful inverse of [`format_to_mtl`]
/// (used for swapchain format detection). Panics on a format the RHI does not model rather
/// than silently substituting a wrong one, which would cause hard-to-debug render-target
/// format mismatches.
pub fn mtl_to_format(mtl: MTLPixelFormat) -> Format {
    match mtl {
        MTLPixelFormat::R8Unorm => Format::R8Unorm,
        MTLPixelFormat::RG8Unorm => Format::R8G8Unorm,
        MTLPixelFormat::RGBA8Unorm => Format::R8G8B8A8Unorm,
        MTLPixelFormat::RGBA8Unorm_sRGB => Format::R8G8B8A8Srgb,
        MTLPixelFormat::BGRA8Unorm => Format::B8G8R8A8Unorm,
        MTLPixelFormat::BGRA8Unorm_sRGB => Format::B8G8R8A8Srgb,
        MTLPixelFormat::R16Float => Format::R16Float,
        MTLPixelFormat::RG16Float => Format::R16G16Float,
        MTLPixelFormat::RGBA16Float => Format::R16G16B16A16Float,
        MTLPixelFormat::R32Float => Format::R32Float,
        MTLPixelFormat::RG32Float => Format::R32G32Float,
        MTLPixelFormat::RGBA32Float => Format::R32G32B32A32Float,
        MTLPixelFormat::RGB10A2Unorm => Format::R10G10B10A2Unorm,
        MTLPixelFormat::RG11B10Float => Format::R11G11B10Float,
        MTLPixelFormat::Depth16Unorm => Format::D16Unorm,
        MTLPixelFormat::Depth32Float => Format::D32Float,
        MTLPixelFormat::Depth24Unorm_Stencil8 => Format::D24UnormS8Uint,
        MTLPixelFormat::Depth32Float_Stencil8 => Format::D32FloatS8Uint,
        MTLPixelFormat::R16Uint => Format::R16Uint,
        MTLPixelFormat::R32Uint => Format::R32Uint,
        other => panic!("MTLPixelFormat {other:?} has no kiln-rhi Format mapping"),
    }
}
