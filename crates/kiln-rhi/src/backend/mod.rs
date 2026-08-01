//! Backend implementations (Vulkan, Metal) and the `BackendKind` discriminant.
//! Not part of the public RHI surface; use [`crate::Device`] instead.

#[cfg(feature = "vulkan")]
pub mod vulkan;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(any(feature = "vulkan", feature = "metal"))]
pub(crate) mod suballoc;

/// Active backend kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Vulkan,
    Metal,
}
