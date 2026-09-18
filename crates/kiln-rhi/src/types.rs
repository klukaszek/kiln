//! Shared primitive types: addresses, handles, formats, and ray-tracing descriptors.

use std::marker::PhantomData;
use zerocopy::{FromBytes, Immutable, IntoBytes};

macro_rules! shader_handle {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[repr(transparent)]
        #[derive(
            Clone, Copy, Debug, Default, PartialEq, Eq, Hash, IntoBytes, FromBytes, Immutable,
        )]
        pub struct $name(pub(crate) u64);

        impl $name {
            pub const NULL: Self = Self(0);

            pub(crate) const fn from_raw(value: u64) -> Self {
                Self(value)
            }

            #[inline]
            pub const fn is_null(self) -> bool {
                self.0 == 0
            }
        }
    };
}

shader_handle!(
    AccelHandle,
    "Opaque acceleration-structure shader handle produced by `AccelerationStructure::gpu`."
);
shader_handle!(
    TextureHandle,
    "Opaque shader handle produced by `Texture::gpu`."
);
shader_handle!(
    SamplerHandle,
    "Opaque sampler shader handle produced by `Sampler::gpu`."
);

/// A GPU virtual address. The type parameter gives element arithmetic and documents the ABI; it
/// carries no ownership, lifetime, or bounds. `GpuPtr<u8>` for untyped memory.
#[repr(transparent)]
#[derive(IntoBytes, FromBytes, Immutable)]
pub struct GpuPtr<T: ?Sized> {
    pub(crate) address: u64,
    marker: PhantomData<fn() -> T>,
}

impl<T: ?Sized> GpuPtr<T> {
    pub const NULL: Self = Self::from_addr(0);

    #[inline]
    pub const fn from_addr(address: u64) -> Self {
        Self {
            address,
            marker: PhantomData,
        }
    }

    /// Integer representation of this pointer, for diagnostics and native interop.
    #[inline]
    pub const fn addr(self) -> u64 {
        self.address
    }

    #[inline]
    pub const fn is_null(self) -> bool {
        self.address == 0
    }

    /// Whether this address is a multiple of `align`.
    ///
    /// # Panics
    /// In debug builds, if `align` is not a non-zero power of two. Release builds report `false`
    /// rather than aborting, but a non-power-of-two alignment is always a caller bug: returning
    /// `false` silently is how an unaligned address gets blamed on the allocator.
    #[inline]
    pub fn is_aligned_to(self, align: u64) -> bool {
        debug_assert!(
            align.is_power_of_two(),
            "alignment {align} is not a non-zero power of two"
        );
        align.is_power_of_two() && self.address & (align - 1) == 0
    }

    /// Offset by bytes, wrapping with raw-pointer semantics.
    #[inline]
    pub fn byte_add(self, bytes: u64) -> Self {
        Self::from_addr(self.address.wrapping_add(bytes))
    }

    /// Reinterpret the pointee type without changing the address.
    #[inline]
    pub const fn cast<U: ?Sized>(self) -> GpuPtr<U> {
        GpuPtr::from_addr(self.address)
    }
}

impl<T> GpuPtr<T> {
    /// Offset by `count` elements, wrapping with raw-pointer semantics.
    #[inline]
    pub fn offset(self, count: u64) -> Self {
        self.byte_add((std::mem::size_of::<T>() as u64).wrapping_mul(count))
    }
}

impl<T: ?Sized> Copy for GpuPtr<T> {}

impl<T: ?Sized> Clone for GpuPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> Default for GpuPtr<T> {
    fn default() -> Self {
        Self::NULL
    }
}

impl<T: ?Sized> PartialEq for GpuPtr<T> {
    fn eq(&self, other: &Self) -> bool {
        self.address == other.address
    }
}

impl<T: ?Sized> Eq for GpuPtr<T> {}

impl<T: ?Sized> std::hash::Hash for GpuPtr<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.address.hash(state);
    }
}

impl<T: ?Sized> std::fmt::Debug for GpuPtr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GpuPtr(0x{:x})", self.address)
    }
}

impl<T: ?Sized> std::fmt::LowerHex for GpuPtr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::LowerHex::fmt(&self.address, f)
    }
}

/// Default bindless texture-heap capacity. See [`DeviceDesc::bindless`](crate::DeviceDesc).
pub const DEFAULT_BINDLESS_TEXTURES: u32 = 1_000_000;
/// Default bindless sampler-heap capacity.
pub const DEFAULT_BINDLESS_SAMPLERS: u32 = 256;

/// How many bindless slots a device reserves. Both heaps are allocated in full at creation and
/// cannot grow, and the default million textures costs tens of MiB on Vulkan, 8 MiB on Metal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BindlessCapacity {
    pub textures: u32,
    pub samplers: u32,
}

impl Default for BindlessCapacity {
    fn default() -> Self {
        Self {
            textures: DEFAULT_BINDLESS_TEXTURES,
            samplers: DEFAULT_BINDLESS_SAMPLERS,
        }
    }
}

/// Texture handle -- index into the global bindless heap.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, IntoBytes, FromBytes, Immutable)]
pub(crate) struct TextureId(pub u32);

/// Sampler handle -- index into the sampler table.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, IntoBytes, FromBytes, Immutable)]
pub(crate) struct SamplerId(pub u32);

/// Pixel / vertex / depth format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Format {
    // Color formats
    R8Unorm,
    R8G8Unorm,
    R8G8B8A8Unorm,
    R8G8B8A8Srgb,
    B8G8R8A8Unorm,
    B8G8R8A8Srgb,
    R16Float,
    R16G16Float,
    R16G16B16A16Float,
    R32Float,
    R32G32Float,
    R32G32B32A32Float,
    R10G10B10A2Unorm,
    R11G11B10Float,

    // Depth/stencil
    D16Unorm,
    D32Float,
    D24UnormS8Uint,
    D32FloatS8Uint,

    // Unsigned integer
    R16Uint,
    R32Uint,
}

/// Every [`Format`] the RHI defines, so conversion tables can be checked exhaustively.
/// The `match` in [`bytes_per_pixel`] is the compiler-checked list; this one is kept beside it.
#[cfg(test)]
pub(crate) const ALL_FORMATS: &[Format] = &[
    Format::R8Unorm,
    Format::R8G8Unorm,
    Format::R8G8B8A8Unorm,
    Format::R8G8B8A8Srgb,
    Format::B8G8R8A8Unorm,
    Format::B8G8R8A8Srgb,
    Format::R16Float,
    Format::R16G16Float,
    Format::R16G16B16A16Float,
    Format::R32Float,
    Format::R32G32Float,
    Format::R32G32B32A32Float,
    Format::R10G10B10A2Unorm,
    Format::R11G11B10Float,
    Format::D16Unorm,
    Format::D32Float,
    Format::D24UnormS8Uint,
    Format::D32FloatS8Uint,
    Format::R16Uint,
    Format::R32Uint,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Topology {
    TriangleList,
    TriangleStrip,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SampleCount {
    S1,
    S2,
    S4,
    S8,
    S16,
}

impl SampleCount {
    /// Samples per pixel, as both backends' APIs want it.
    pub const fn count(self) -> u32 {
        match self {
            Self::S1 => 1,
            Self::S2 => 2,
            Self::S4 => 4,
            Self::S8 => 8,
            Self::S16 => 16,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompareOp {
    Never,
    Less,
    Equal,
    LessOrEqual,
    Greater,
    NotEqual,
    GreaterOrEqual,
    Always,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlendFactor {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    DstColor,
    OneMinusDstColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

/// Texture dimension.
///
/// [`TextureDesc::array_layers`](crate::TextureDesc::array_layers) counts whole *images*, never
/// faces: slices for `D2Array`, cubes for `Cube` (always 1) and `CubeArray`. Backends multiply by
/// six where their own API wants faces. Every other dimension requires `array_layers == 1`; use
/// `D2Array` for an array of 2D images.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureDimension {
    D1,
    D2,
    D2Array,
    D3,
    Cube,
    CubeArray,
}

impl TextureDimension {
    /// Whether this dimension stores more than one image, i.e. whether `array_layers` may exceed 1.
    pub(crate) const fn is_array(self) -> bool {
        matches!(self, Self::D2Array | Self::CubeArray)
    }

    /// Number of 2D faces backing `array_layers` images: six per cube, one otherwise. This is what
    /// Vulkan calls `arrayLayers` and what a Metal texture view counts as slices.
    pub(crate) const fn face_count(self, array_layers: u32) -> u32 {
        match self {
            Self::Cube | Self::CubeArray => array_layers.saturating_mul(6),
            _ => array_layers,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FilterMode {
    Nearest,
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AddressMode {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
    ClampToBorder,
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    /// Color write mask.
    pub struct ColorWriteMask: u8 {
        const R = 0x01;
        const G = 0x02;
        const B = 0x04;
        const A = 0x08;
        const ALL = 0x0F;
    }
}

bitflags::bitflags! {
    /// Depth mode: empty = disabled, `READ` = test only, `READ | WRITE` = test and write.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
    pub struct DepthFlags: u8 {
        const READ  = 0x1;
        const WRITE = 0x2;
    }
}

/// Cull mode. Front face is always CCW; the variant names the winding culled. `Cw` is the common
/// case (back-face culling).
///
/// No "cull everything": Metal has no `FRONT_AND_BACK`, and flipping the winding to fake it culls
/// only half the triangles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cull {
    None,
    Cw,
    Ccw,
}

/// Maximum frames in flight.
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

bitflags::bitflags! {
    /// Per-geometry flags for a BLAS build.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct GeometryFlags: u8 {
        /// Geometry is opaque — skip any-hit shaders (better performance).
        const OPAQUE        = 0x01;
        /// Do not invoke any-hit shaders for duplicate intersections.
        const NO_DUPLICATE_ANYHIT = 0x02;
    }
}

bitflags::bitflags! {
    /// Per-instance flags for a TLAS instance.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct InstanceFlags: u8 {
        const TRIANGLE_FACING_CULL_DISABLE  = 0x01;
        const TRIANGLE_FLIP_FACING          = 0x02;
        const FORCE_OPAQUE                  = 0x04;
        const FORCE_NO_OPAQUE               = 0x08;
    }
}

bitflags::bitflags! {
    /// Acceleration structure build flags. No `ALLOW_UPDATE`: every build is a full rebuild, so
    /// the flag would describe an operation the RHI cannot request.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct BuildAccelFlags: u8 {
        const PREFER_FAST_TRACE = 0x02;
        const PREFER_FAST_BUILD = 0x04;
        const MINIMIZE_MEMORY   = 0x08;
    }
}

/// One geometry entry in a BLAS descriptor.
#[derive(Clone, Copy, Debug)]
pub struct BlasMeshDesc {
    pub flags: GeometryFlags,
    pub geometry: BlasGeometry,
}

/// What a BLAS geometry holds. The variant *is* the geometry type, so a mesh cannot carry
/// bounding boxes, a procedural one cannot carry indices, and neither can lack its buffer.
#[derive(Clone, Copy, Debug)]
pub enum BlasGeometry {
    Triangles {
        /// Vertex positions.
        vertices: GpuPtr<[f32; 3]>,
        /// Bytes between successive positions.
        stride: u64,
        count: u32,
        /// `None` reads `vertices` three at a time.
        indices: Option<BlasIndices>,
    },
    /// Axis-aligned bounding boxes, for procedural geometry.
    Aabbs { buffer: GpuPtr<Aabb>, count: u32 },
}

/// The index buffer of an indexed triangle geometry.
#[derive(Clone, Copy, Debug)]
pub struct BlasIndices {
    pub buffer: GpuPtr<u32>,
    pub count: u32,
}

impl BlasGeometry {
    /// Primitives this geometry builds. Both backends size their builds from it, so the rule
    /// lives here rather than in each of them.
    pub(crate) fn primitive_count(self) -> u32 {
        match self {
            Self::Triangles { count, indices, .. } => indices.map_or(count, |i| i.count) / 3,
            Self::Aabbs { count, .. } => count,
        }
    }
}

/// Axis-aligned bounding box consumed by procedural BLAS geometry.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, IntoBytes, FromBytes, Immutable)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

/// Descriptor for building a Bottom-Level Acceleration Structure.
#[derive(Clone, Debug, Default)]
pub struct BlasDesc<'a> {
    pub meshes: &'a [BlasMeshDesc],
    pub flags: BuildAccelFlags,
}

/// One logical instance entry in a TLAS, stored in GPU-visible memory.
/// Each backend encodes it into its native instance-descriptor layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, IntoBytes, FromBytes, Immutable)]
pub struct TlasInstance {
    /// Row-major 3×4 transform matrix.
    pub transform: [[f32; 4]; 3],
    /// Low 24 bits: instance custom index (gl_InstanceCustomIndex).
    /// High 8 bits: mask (the ray's ray mask is ANDed with this).
    pub instance_custom_index_and_mask: u32,
    /// Low 24 bits: shader binding table hit group offset.
    /// High 8 bits: `InstanceFlags`.
    pub instance_sbt_offset_and_flags: u32,
    /// BLAS referenced by this instance — assign `blas.gpu()`, the same handle a shader takes.
    pub acceleration_structure_reference: AccelHandle,
}

/// Descriptor for building a Top-Level Acceleration Structure.
#[derive(Clone, Debug, Default)]
pub struct TlasDesc {
    /// GPU address of an array of `TlasInstance`.
    pub instance_buffer: GpuPtr<TlasInstance>,
    pub instance_count: u32,
    pub flags: BuildAccelFlags,
}
