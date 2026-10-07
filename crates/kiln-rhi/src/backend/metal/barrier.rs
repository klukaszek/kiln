//! RHI barrier flags to Metal's. Encoding stays with the command buffer.

use objc2_metal::{MTL4VisibilityOptions, MTLRenderStages, MTLStages};

use crate::barrier::{HazardFlags, StageFlags};

pub(crate) const ALL_RENDER_STAGES: MTLRenderStages = MTLRenderStages(
    MTLRenderStages::Vertex.0
        | MTLRenderStages::Fragment.0
        | MTLRenderStages::Object.0
        | MTLRenderStages::Mesh.0,
);

/// Empty in means empty out; see `to_vk_stage_flags` for why this must not widen to `All`.
pub(crate) fn to_mtl_stages(flags: StageFlags) -> MTLStages {
    if flags.contains(StageFlags::ALL_COMMANDS) {
        return MTLStages::All;
    }

    let mut stages = MTLStages::empty();
    if flags.contains(StageFlags::VERTEX_SHADER) {
        stages |= MTLStages::Vertex;
    }
    if flags.contains(StageFlags::PIXEL_SHADER) || flags.contains(StageFlags::RASTER_COLOR_OUT) {
        stages |= MTLStages::Fragment;
    }
    if flags.contains(StageFlags::COMPUTE) {
        stages |= MTLStages::Dispatch;
    }
    if flags.contains(StageFlags::TRANSFER) {
        stages |= MTLStages::Blit;
    }
    if flags.contains(StageFlags::ACCELERATION_STRUCTURE) {
        stages |= MTLStages::AccelerationStructure;
    }
    if flags.contains(StageFlags::RASTER_DEPTH_OUT) {
        stages |= MTLStages::Fragment;
    }
    if flags.contains(StageFlags::MESH_SHADER) {
        stages |= MTLStages::Mesh | MTLStages::Object;
    }

    if flags.contains(StageFlags::ALL_GRAPHICS) {
        stages |= MTLStages::Vertex;
        stages |= MTLStages::Fragment;
        stages |= MTLStages::Tile;
        stages |= MTLStages::Mesh;
        stages |= MTLStages::Object;
    }

    stages
}

pub(crate) fn visibility_from_hazard(hazard: HazardFlags) -> MTL4VisibilityOptions {
    if hazard.is_empty() {
        return MTL4VisibilityOptions::None;
    }

    // Device visibility is required for GPU-written arguments, depth, and descriptor aliases.
    let needs_device = hazard.intersects(
        HazardFlags::DRAW_ARGUMENTS | HazardFlags::DEPTH_STENCIL | HazardFlags::DESCRIPTORS,
    );
    let needs_alias = hazard.contains(HazardFlags::DESCRIPTORS);

    let mut visibility = if needs_device {
        MTL4VisibilityOptions::Device
    } else {
        MTL4VisibilityOptions::None
    };
    if needs_alias {
        visibility |= MTL4VisibilityOptions::ResourceAlias;
    }
    visibility
}
