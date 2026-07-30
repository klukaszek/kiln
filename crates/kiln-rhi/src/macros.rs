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
/// the host/device layout in lockstep. Must be padding-free (add explicit tail padding where
/// alignment would insert it).
///
/// The Slang type of a field is inferred from its Rust type (see [`gpu_slang_ty!`] for the
/// table: `Vec4` → `float4`, `u32` → `uint`, …). Pointers are the exception — `GpuAddress`
/// erases its pointee, so device pointers spell out the Slang type with `as "T*"`. The same
/// `as "..."` override works for any field whose mapping isn't built in.
///
/// ```ignore
/// gpu_struct! {
///     pub struct Material {
///         albedo: u32,                      // -> uint
///         tint:   Vec4,                     // -> float4
///         data:   GpuAddress as "Surface*", // 64-bit device pointer (pointee is explicit)
///     }
/// }
/// ```
#[macro_export]
macro_rules! gpu_struct {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident {
            $( $fname:ident : $fty:tt $(as $slang:literal)? ),* $(,)?
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
                $( "    ", $crate::gpu_slang_ty!($fty $(, $slang)?), " ", stringify!($fname), ";\n", )*
                "};\n"
            );
        }
    };
}

/// Map a Rust field type to its Slang spelling for [`gpu_struct!`]. A trailing
/// `, "literal"` overrides the mapping — used for `GpuAddress` device pointers,
/// whose pointee the Rust type can't carry. Field types must be a single token
/// (import the type rather than writing a path).
#[doc(hidden)]
#[macro_export]
macro_rules! gpu_slang_ty {
    // Explicit override (device pointers, or any type without a built-in mapping).
    ($fty:tt, $slang:literal) => {
        $slang
    };

    (AccelHandle) => {
        "DescriptorHandle<RaytracingAccelerationStructure>"
    };
    (TextureHandle) => {
        "DescriptorHandle<Texture2D>"
    };
    (StorageTextureHandle) => {
        "DescriptorHandle<RWTexture2D<float>>"
    };
    (SamplerHandle) => {
        "DescriptorHandle<SamplerState>"
    };

    (f32) => {
        "float"
    };
    (u32) => {
        "uint"
    };
    (i32) => {
        "int"
    };
    (Vec2) => {
        "float2"
    };
    (Vec3) => {
        "float3"
    };
    (Vec4) => {
        "float4"
    };
    (UVec2) => {
        "uint2"
    };
    (UVec3) => {
        "uint3"
    };
    (UVec4) => {
        "uint4"
    };
    (IVec2) => {
        "int2"
    };
    (IVec3) => {
        "int3"
    };
    (IVec4) => {
        "int4"
    };
    (Mat3) => {
        "float3x3"
    };
    (Mat4) => {
        "float4x4"
    };
    ([f32; 2]) => {
        "float2"
    };
    ([f32; 3]) => {
        "float3"
    };
    ([f32; 4]) => {
        "float4"
    };
    ([f32; 16]) => {
        "float4x4"
    };
}
