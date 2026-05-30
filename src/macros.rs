//! Internal helper macros.

/// Collapse a backend-passthrough match into `backend_dispatch!(&self.inner, DeviceInner, d
/// => d.foo(x))`. The body is duplicated into each backend arm under its `cfg`.
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

/// Define a GPU-facing struct once, generating the `#[repr(C)]` [`GpuPod`](crate::GpuPod) Rust
/// type and a matching Slang declaration string `Name::SLANG` to prepend to a shader — keeping
/// the host/device layout in lockstep. Each field gives its Rust and Slang type. Must be
/// padding-free (add explicit tail padding where alignment would insert it).
///
/// ```ignore
/// gpu_struct! {
///     pub struct Material {
///         albedo: u32 as "uint",            // bindless texture id
///         tint:   [f32; 4] as "float4",
///         data:   GpuAddress as "Surface*", // 64-bit GPU pointer
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
