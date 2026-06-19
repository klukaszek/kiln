//! Backend implementations (Vulkan, Metal) and the `BackendKind` discriminant.
//! Not part of the public RHI surface; use [`crate::Device`] instead.

#[cfg(feature = "vulkan")]
pub mod vulkan;

#[cfg(feature = "metal")]
pub mod metal;

/// Active backend kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Vulkan,
    Metal,
}
