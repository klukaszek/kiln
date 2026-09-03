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

/// Unwrap a backend-tagged handle to the variant the calling backend owns:
/// `backend_expect!(&pso.inner, ComputePsoInner::Metal)`. The fallback arm exists only in
/// multi-backend builds, where reaching it means a handle from the other backend crossed over.
macro_rules! backend_expect {
    ($value:expr, $variant:path) => {
        match $value {
            $variant(inner) => inner,
            #[allow(unreachable_patterns)]
            _ => unreachable!("handle belongs to a different backend"),
        }
    };
}

/// Define a GPU struct once, emitting the `#[repr(C)]` Rust type and a matching Slang
/// declaration in `Name::SLANG`. Field types map through [`gpu_slang_ty!`](crate::gpu_slang_ty);
/// `as "..."` overrides a mapping. Padding is rejected at compile time by `IntoBytes`.
///
/// ```ignore
/// gpu_struct! {
///     pub struct Material {
///         albedo: u32,                      // -> uint
///         tint:   Vec4,                     // -> float4
///         data:   GpuPtr<Surface>,          // -> Surface*
///         table:  GpuPtr<f32, Read>,        // -> Ptr<float, Access.Read>
///     }
/// }
/// ```
///
/// A pointer's second parameter is a shader-side annotation, not a Rust type parameter: the field
/// is a plain [`GpuPtr<T>`](crate::GpuPtr) either way. `Read` emits Slang's read-only pointer, which
/// lets the compiler assume nothing writes through it; the default is a read-write `T*`. Access is a
/// property of how *this* shader uses the buffer, so the same allocation can be `Read` in one root
/// struct and read-write in another.
#[macro_export]
macro_rules! gpu_struct {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident {
            $($fields:tt)*
        }
    ) => {
        $crate::__gpu_struct_parse! {
            [$(#[$meta])*] [$vis] [$name] [] [] ; $($fields)*,
        }
    };
}

/// Parser for [`gpu_struct!`] fields. Kept public only because exported macros expand in the
/// downstream crate; it is not API.
#[doc(hidden)]
#[macro_export]
macro_rules! __gpu_struct_parse {
    ([$($meta:tt)*] [$vis:vis] [$name:ident] [$($rust:tt)*] [$($slang:tt)*] ; $(,)*) => {
        $($meta)*
        #[repr(C)]
        #[derive(
            Clone,
            Copy,
            $crate::zerocopy::IntoBytes,
            $crate::zerocopy::FromBytes,
            $crate::zerocopy::Immutable,
        )]
        $vis struct $name {
            $($rust)*
        }
        impl $name {
            /// Slang declaration matching this struct's layout; prepend to shader source.
            pub const SLANG: &'static str = concat!(
                "struct ", stringify!($name), " {\n",
                $($slang)*
                "};\n"
            );
        }
    };

    // Device pointer the shader only reads. Slang can then assume no aliasing writes.
    ([$($meta:tt)*] [$vis:vis] [$name:ident] [$($rust:tt)*] [$($out:tt)*] ;
        $field:ident : GpuPtr<$pointee:tt, Read>, $($rest:tt)*) => {
        $crate::__gpu_struct_parse! {
            [$($meta)*] [$vis] [$name]
            [$($rust)* pub $field: $crate::GpuPtr<$pointee>,]
            [$($out)* "    Ptr<", $crate::gpu_slang_ty!($pointee), ", Access.Read> ",
                stringify!($field), ";\n",]
            ; $($rest)*
        }
    };

    // Device pointer the shader writes through, spelled out.
    ([$($meta:tt)*] [$vis:vis] [$name:ident] [$($rust:tt)*] [$($out:tt)*] ;
        $field:ident : GpuPtr<$pointee:tt, ReadWrite>, $($rest:tt)*) => {
        $crate::__gpu_struct_parse! {
            [$($meta)*] [$vis] [$name]
            [$($rust)* pub $field: $crate::GpuPtr<$pointee>,]
            [$($out)* "    ", $crate::gpu_slang_ty!($pointee), "* ", stringify!($field), ";\n",]
            ; $($rest)*
        }
    };

    // Device pointer, read-write by default.
    ([$($meta:tt)*] [$vis:vis] [$name:ident] [$($rust:tt)*] [$($out:tt)*] ;
        $field:ident : GpuPtr<$pointee:tt>, $($rest:tt)*) => {
        $crate::__gpu_struct_parse! {
            [$($meta)*] [$vis] [$name]
            [$($rust)* pub $field: $crate::GpuPtr<$pointee>,]
            [$($out)* "    ", $crate::gpu_slang_ty!($pointee), "* ", stringify!($field), ";\n",]
            ; $($rest)*
        }
    };

    // Ordinary field with an explicit Slang spelling.
    ([$($meta:tt)*] [$vis:vis] [$name:ident] [$($rust:tt)*] [$($out:tt)*] ;
        $field:ident : $ty:tt as $slang:literal, $($rest:tt)*) => {
        $crate::__gpu_struct_parse! {
            [$($meta)*] [$vis] [$name]
            [$($rust)* pub $field: $ty,]
            [$($out)* "    ", $slang, " ", stringify!($field), ";\n",]
            ; $($rest)*
        }
    };

    // Ordinary field using the built-in Rust-to-Slang mapping.
    ([$($meta:tt)*] [$vis:vis] [$name:ident] [$($rust:tt)*] [$($out:tt)*] ;
        $field:ident : $ty:tt, $($rest:tt)*) => {
        $crate::__gpu_struct_parse! {
            [$($meta)*] [$vis] [$name]
            [$($rust)* pub $field: $ty,]
            [$($out)* "    ", $crate::gpu_slang_ty!($ty), " ", stringify!($field), ";\n",]
            ; $($rest)*
        }
    };
}

/// Map a Rust field type to its Slang spelling for [`gpu_struct!`]. A trailing
/// `, "literal"` overrides the mapping when the Rust and Slang names differ.
/// Field types must be a single token (import the type rather than writing a path).
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

    // A `gpu_struct!` type, whose Slang declaration carries its Rust name.
    ($other:tt) => {
        stringify!($other)
    };
}
