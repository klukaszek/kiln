//! Internal helper macros.

/// Dispatch to whichever backend an enum-dispatch wrapper currently holds.
///
/// Collapses the repetitive backend-passthrough match:
/// ```ignore
/// match &self.inner {
///     #[cfg(feature = "vulkan")] DeviceInner::Vulkan(d) => d.foo(x),
///     #[cfg(feature = "metal")]  DeviceInner::Metal(d)  => d.foo(x),
/// }
/// ```
/// into `backend_dispatch!(&self.inner, DeviceInner, d => d.foo(x))`. Works with `&` or
/// `&mut` receivers; the body is duplicated into each backend arm under its `cfg`.
macro_rules! backend_dispatch {
    ($value:expr, $variant:ident, $bind:ident => $body:expr $(,)?) => {
        match $value {
            #[cfg(feature = "vulkan")]
            $variant::Vulkan($bind) => $body,
            #[cfg(feature = "metal")]
            $variant::Metal($bind) => $body,
        }
    };
}
