//! The GPU record layout the trace kernel reads.
//!
//! These declarations are the ABI: `gpu_struct!` emits both the Rust type and the Slang struct, so
//! a field added on one side cannot go missing on the other.

use glam::{Vec2, Vec4};
use kiln_rhi::gpu_struct;

gpu_struct! {
    pub(crate) struct GpuMaterial {
        // Constant tint, multiplied by the base-colour map when one is bound.
        base_color: Vec4,
        // Fallbacks used wherever the matching map is absent.
        roughness: f32,
        metallic: f32,
        ior: f32,
        _pad: f32,
        // Texture-binding index, or NO_MAP. Scalar maps pack their channel into the high byte.
        base_color_map: u32,
        roughness_map: u32,
        metallic_map: u32,
        _pad2: u32,
    }
}

/// Sentinel for a material input that is authored as a constant rather than sampled.
pub(crate) const NO_MAP: u32 = u32::MAX;

gpu_struct! {
    pub(crate) struct GpuEmissiveHit {
        light_index: u32,
    }
}

gpu_struct! {
    pub(crate) struct GpuLight {
        p0_emission: Vec4,
        edge1_area: Vec4,
        edge2: Vec4,
        normal: Vec4,
    }
}

gpu_struct! {
    pub(crate) struct GpuMeshLightTriangle {
        p0: Vec4,
        edge1: Vec4,
        edge2: Vec4,
        normal_area: Vec4,
    }
}

gpu_struct! {
    pub(crate) struct GpuTriangle {
        // Mesh-local; instances apply the normal transform in the shader.
        normal_area: Vec4,
        uv01: Vec4,
        uv2: Vec2,
        material_id: u32,
        emission: f32,
    }
}

gpu_struct! {
    pub(crate) struct GpuInstance {
        triangle_base: u32,
        _pad: u32,
        _pad2: u32,
        _pad3: u32,
        // Columns of the inverse-transpose normal matrix.
        normal_x: Vec4,
        normal_y: Vec4,
        normal_z_determinant: Vec4,
    }
}
