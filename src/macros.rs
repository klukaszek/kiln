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

/// Define a GPU-facing struct once, generating both the `#[repr(C)]` Rust type (a
/// [`GpuPod`](crate::GpuPod) via zerocopy derives) and a matching Slang declaration string
/// `Name::SLANG` to prepend to a shader. Keeps the host/device data contract in lockstep so
/// roots are built type-safely instead of poking bytes at hand-computed offsets.
///
/// Each field gives its Rust type and the equivalent Slang type. The struct must be
/// padding-free (zerocopy `IntoBytes` requires it) — add explicit tail-padding fields where
/// alignment would otherwise insert padding.
///
/// ```ignore
/// gpu_struct! {
///     pub struct Material {
///         albedo: u32 as "uint",   // bindless texture id
///         tint:   [f32; 4] as "float4",
///         data:   u64 as "Surface*", // 64-bit GPU pointer
///     }
/// }
/// ```
#[macro_export]
macro_rules! gpu_struct {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident {
            $( $fname:ident : $fty:ty as $slang:literal ),* $(,)?
        }
    ) => {
        $(#[$meta])*
        #[repr(C)]
        #[derive(
            Clone,
            Copy,
            $crate::zerocopy::IntoBytes,
            $crate::zerocopy::FromBytes,
            $crate::zerocopy::Immutable,
        )]
        $vis struct $name {
            $( pub $fname : $fty ),*
        }
        impl $name {
            /// Slang declaration matching this struct's layout; prepend to shader source.
            pub const SLANG: &'static str = concat!(
                "struct ", stringify!($name), " {\n",
                $( "    ", $slang, " ", stringify!($fname), ";\n", )*
                "};\n"
            );
        }
    };
}
