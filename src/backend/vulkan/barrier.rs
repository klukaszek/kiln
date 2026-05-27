use crate::barrier::{HazardFlags, StageFlags};
use ash::vk;

/// Convert RHI stage flags to Vulkan pipeline stage flags (synchronization2).
pub fn to_vk_stage_flags(flags: StageFlags) -> vk::PipelineStageFlags2 {
    if flags.contains(StageFlags::ALL_COMMANDS) {
        return vk::PipelineStageFlags2::ALL_COMMANDS;
    }

    let mut result = vk::PipelineStageFlags2::NONE;

    if flags.contains(StageFlags::VERTEX_SHADER) {
        result |= vk::PipelineStageFlags2::VERTEX_SHADER;
    }
    if flags.contains(StageFlags::PIXEL_SHADER) {
        result |= vk::PipelineStageFlags2::FRAGMENT_SHADER;
    }
    if flags.contains(StageFlags::COMPUTE) {
        result |= vk::PipelineStageFlags2::COMPUTE_SHADER;
    }
    if flags.contains(StageFlags::RASTER_COLOR_OUT) {
        result |= vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT;
    }
    if flags.contains(StageFlags::RASTER_DEPTH_OUT) {
        result |= vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
    }
    if flags.contains(StageFlags::TRANSFER) {
        result |= vk::PipelineStageFlags2::TRANSFER;
    }
    if flags.contains(StageFlags::ALL_GRAPHICS) {
        result |= vk::PipelineStageFlags2::ALL_GRAPHICS;
    }

    if result.is_empty() {
        vk::PipelineStageFlags2::ALL_COMMANDS
    } else {
        result
    }
}

/// Convert RHI hazard flags to Vulkan access flags (for global memory barriers).
///
/// Aaltonen's three hazard cases map cleanly onto Vulkan's access flag model.
pub fn to_vk_access_flags(hazard: HazardFlags, is_src: bool) -> vk::AccessFlags2 {
    let mut result = vk::AccessFlags2::NONE;

    if hazard.contains(HazardFlags::DRAW_ARGUMENTS) {
        if is_src {
            result |= vk::AccessFlags2::SHADER_WRITE;
        } else {
            result |= vk::AccessFlags2::INDIRECT_COMMAND_READ;
        }
    }
    if hazard.contains(HazardFlags::DESCRIPTORS) {
        // Descriptor heap written by CPU or compute; samplers need cache invalidation.
        result |= vk::AccessFlags2::SHADER_READ;
        if is_src {
            result |= vk::AccessFlags2::SHADER_WRITE;
        }
    }
    if hazard.contains(HazardFlags::DEPTH_STENCIL) {
        result |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
    }

    result
}
