//! Backend implementations (Vulkan and Metal).
//! Not part of the public RHI surface; use [`crate::Device`] instead.

#[cfg(feature = "vulkan")]
pub mod vulkan;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(any(feature = "vulkan", feature = "metal"))]
pub(crate) mod suballoc;
