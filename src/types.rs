use zerocopy::{FromBytes, Immutable, IntoBytes};

/// GPU virtual address for buffer device address / Metal gpuAddress.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, IntoBytes, FromBytes, Immutable)]
pub struct GpuAddress(pub u64);

impl GpuAddress {
    pub const NULL: Self = Self(0);

    /// Offset this address by `byte_offset`.
    #[inline]
    pub fn offset(self, byte_offset: u64) -> Self {
        Self(self.0 + byte_offset)
    }

    #[inline]
    pub fn is_null(self) -> bool {
        self.0 == 0
    }
}

/// Maximum number of bindless textures supported by the RHI.
pub const MAX_BINDLESS_TEXTURES: u32 = 1_000_000;
/// Maximum number of bindless samplers supported by the RHI.
pub const MAX_BINDLESS_SAMPLERS: u32 = 256;

/// Texture handle -- index into the global bindless heap.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, IntoBytes, FromBytes, Immutable)]
pub struct TextureId(pub u32);

impl TextureId {
    pub const INVALID: Self = Self(u32::MAX);
}

/// Sampler handle -- index into the sampler table.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, IntoBytes, FromBytes, Immutable)]
pub struct SamplerId(pub u32);

impl SamplerId {
    pub const INVALID: Self = Self(u32::MAX);
}

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
    R32G32B32Float,
    R32G32B32A32Float,
    R10G10B10A2Unorm,
    R11G11B10Float,

    // Depth/stencil
    D16Unorm,
    D32Float,
    D24UnormS8Uint,
    D32FloatS8Uint,

    // Index
    R16Uint,
    R32Uint,
}

/// Primitive topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Topology {
    TriangleList,
    TriangleStrip,
    /// Triangle fan. **Not natively supported on Metal** — requires CPU-side index rewriting
    /// to `TriangleList` before submission. Use only on Vulkan or with pre-converted data.
    TriangleFan,
}

/// MSAA sample count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SampleCount {
    S1,
    S2,
    S4,
    S8,
    S16,
}

/// Compare operation for depth/stencil.
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

/// Blend factor.
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

/// Blend operation.
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
/// Matches Aaltonen's `TEXTURE { TEXTURE_1D, TEXTURE_2D, TEXTURE_3D, TEXTURE_CUBE,
/// TEXTURE_2D_ARRAY, TEXTURE_CUBE_ARRAY }`.
///
/// For `D2Array`, `TextureDesc::array_layers` is the number of slices.
/// For `CubeArray`, `TextureDesc::array_layers` is the total number of faces (n × 6).
/// For `Cube`, `array_layers` is the number of cubes (1 for a single cubemap);
/// both backends automatically multiply by 6 when allocating storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureDimension {
    D1,
    D2,
    D2Array,
    D3,
    Cube,
    CubeArray,
}

/// Filter mode for samplers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FilterMode {
    Nearest,
    Linear,
}

/// Address mode for samplers.
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
    /// Depth mode flags. Matches Aaltonen's `DEPTH_FLAGS { DEPTH_READ = 0x1, DEPTH_WRITE = 0x2 }`.
    ///
    /// - `0` = depth fully disabled
    /// - `READ` = depth test only (no writes; useful for transparent surfaces)
    /// - `READ | WRITE` = full depth test and write
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
    pub struct DepthFlags: u8 {
        const READ  = 0x1;
        const WRITE = 0x2;
    }
}

/// Stencil operation applied when a stencil test passes or fails.
///
/// Corresponds to Aaltonen's `OP_KEEP`, `OP_ZERO`, etc. in the stencil context.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StencilOp {
    Keep,
    Zero,
    Replace,
    IncrementClamp,
    DecrementClamp,
    Invert,
    IncrementWrap,
    DecrementWrap,
}

/// Cull mode for rasterization.
///
/// Encodes both culling direction and implied front-face winding in a single value,
/// matching Aaltonen's `CULL { CULL_CCW, CULL_CW, CULL_ALL, CULL_NONE }`.
///
/// - `Cw`  — cull clockwise-winding triangles (standard back-face culling; front = CCW)
/// - `Ccw` — cull counter-clockwise-winding triangles (front-face culling; front = CCW)
/// - `All` — cull both faces
/// - `None` — no culling
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cull {
    None,
    Cw,
    Ccw,
    All,
}

/// Clip-space Y direction. Vulkan is Y-down; Metal/OpenGL is Y-up.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClipSpaceY {
    /// NDC Y points down (Vulkan). Projection must negate Y.
    Down,
    /// NDC Y points up (Metal, OpenGL). Standard projection.
    Up,
}

/// Maximum frames in flight.
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

// ---------------------------------------------------------------------------
// Ray tracing types
// ---------------------------------------------------------------------------

/// Opaque acceleration structure handle (BLAS or TLAS).
///
/// In Aaltonen's model everything is GPU memory. An acceleration structure is
/// a GPU allocation whose GPU address is placed directly in a root struct —
/// exactly the same as any other buffer pointer. The CPU-side handle exists
/// only so the backend can issue `build` commands against the right object.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AccelerationStructureId(pub u32);

impl AccelerationStructureId {
    pub const INVALID: Self = Self(u32::MAX);
}

/// Geometry type inside a BLAS.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GeometryType {
    /// Triangle geometry. Backend reads positions from `vertex_buffer` using
    /// `vertex_format` and `vertex_stride`.
    Triangles,
    /// Axis-aligned bounding boxes for procedural geometry.
    Aabbs,
}

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
    /// Acceleration structure build flags.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct BuildAccelFlags: u8 {
        const ALLOW_UPDATE      = 0x01;
        const ALLOW_COMPACTION  = 0x02;
        const PREFER_FAST_TRACE = 0x04;
        const PREFER_FAST_BUILD = 0x08;
        const MINIMIZE_MEMORY   = 0x10;
    }
}

/// One geometry entry in a BLAS descriptor.
#[derive(Clone, Debug)]
pub struct BlasMeshDesc {
    pub geometry_type: GeometryType,
    pub flags: GeometryFlags,
    /// GPU address of the vertex position buffer (for Triangles).
    pub vertex_buffer: GpuAddress,
    /// Bytes between successive vertex positions (for Triangles).
    pub vertex_stride: u64,
    /// Number of vertices.
    pub vertex_count: u32,
    /// GPU address of index buffer (0 = non-indexed).
    pub index_buffer: GpuAddress,
    /// Index count (0 = non-indexed).
    pub index_count: u32,
    /// GPU address of the AABB buffer (for Aabbs geometry).
    pub aabb_buffer: GpuAddress,
    /// Number of AABBs.
    pub aabb_count: u32,
}

/// Descriptor for building a Bottom-Level Acceleration Structure.
#[derive(Clone, Debug)]
pub struct BlasDesc {
    pub meshes: Vec<BlasMeshDesc>,
    pub flags: BuildAccelFlags,
}

/// One instance entry in a TLAS, stored in GPU-visible memory.
/// Layout matches `VkAccelerationStructureInstanceKHR` / Metal `MTLAccelerationStructureInstanceDescriptor`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TlasInstance {
    /// Row-major 3×4 transform matrix.
    pub transform: [[f32; 4]; 3],
    /// Low 24 bits: instance custom index (gl_InstanceCustomIndex).
    /// High 8 bits: mask (the ray's ray mask is ANDed with this).
    pub instance_custom_index_and_mask: u32,
    /// Low 24 bits: shader binding table hit group offset.
    /// High 8 bits: `InstanceFlags`.
    pub instance_sbt_offset_and_flags: u32,
    /// GPU address of the BLAS for this instance.
    pub acceleration_structure_reference: u64,
}

/// Descriptor for building a Top-Level Acceleration Structure.
#[derive(Clone, Debug)]
pub struct TlasDesc {
    /// GPU address of an array of `TlasInstance`.
    pub instance_buffer: GpuAddress,
    /// Number of instances.
    pub instance_count: u32,
    pub flags: BuildAccelFlags,
}
