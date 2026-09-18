//! Internal helper macros.

/// Name a backend's type for a handle. Exactly one backend is compiled in (the features are
/// mutually exclusive), so this is a plain alias and a handle's `inner` *is* the backend object.
macro_rules! backend_enum {
    (
        $(#[$meta:meta])*
        $name:ident { vulkan: $vulkan:ty, metal: $metal:ty $(,)? }
    ) => {
        #[cfg(feature = "vulkan")]
        $(#[$meta])*
        pub(crate) type $name = $vulkan;
        #[cfg(feature = "metal")]
        $(#[$meta])*
        pub(crate) type $name = $metal;
    };
}

/// Define a GPU struct once, emitting the `#[repr(C)]` Rust type and a matching Slang
/// declaration in `Name::SLANG`. Field types map through [`gpu_slang_ty!`](crate::gpu_slang_ty);
/// `as "..."` overrides a mapping.
///
/// ```ignore
/// gpu_struct! {
///     pub struct DrawRoot {
///         tint:     Vec4,                // -> float4
///         vertices: GpuPtr<Vertex>,      // -> Vertex*
///         table:    GpuPtr<f32, Read>,   // -> Ptr<float, Access.Read>
///         count:    u32,                 // -> uint
///         pad:      u32,                 // declared, so the struct has no implicit padding
///     }
/// }
/// ```
///
/// The struct is memcpy'd to the GPU, so padding must be declared as a field: `IntoBytes` rejects
/// implicit padding, and a declared field keeps the compiler catching omissions at every
/// construction site.
///
/// A pointer's second parameter is a shader-side annotation, not a Rust type parameter — the field
/// is a plain [`GpuPtr<T>`](crate::GpuPtr) either way. The same allocation can be `Read` in one
/// root struct and read-write in another.
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

    // Read-only device pointer, so Slang can assume no aliasing writes.
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

    // Read-write device pointer, spelled out.
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

    // Stored as a handle; a Slang property converts it on access, since the backends build the
    // structure from those bytes differently.
    ([$($meta:tt)*] [$vis:vis] [$name:ident] [$($rust:tt)*] [$($out:tt)*] ;
        $field:ident : AccelHandle, $($rest:tt)*) => {
        $crate::__gpu_struct_parse! {
            [$($meta)*] [$vis] [$name]
            [$($rust)* pub $field: $crate::AccelHandle,]
            [$($out)*
                "    DescriptorHandle<RaytracingAccelerationStructure> ",
                stringify!($field), "_handle;\n",
                "    property RaytracingAccelerationStructure ", stringify!($field),
                " { get { return kiln::accel(", stringify!($field), "_handle); } }\n",]
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

/// Map a Rust field type to its Slang spelling for [`gpu_struct!`]. A trailing `, "literal"`
/// overrides the mapping. Field types must be a single token, so import the type rather than
/// writing a path.
#[doc(hidden)]
#[macro_export]
macro_rules! gpu_slang_ty {
    ($fty:tt, $slang:literal) => {
        $slang
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
