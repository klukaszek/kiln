//! Stage and hazard flags for pipeline barriers.

bitflags::bitflags! {
    /// Producer/consumer stages for a barrier. Stage-only — no per-resource state tracking.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct StageFlags: u32 {
        const VERTEX_SHADER     = 0x01;
        const PIXEL_SHADER      = 0x02;
        const COMPUTE           = 0x04;
        const RASTER_COLOR_OUT  = 0x08;
        const RASTER_DEPTH_OUT  = 0x10;
        const TRANSFER          = 0x20;
        /// Builds and ray-query traversal. Not covered by `COMPUTE` on either backend.
        const ACCELERATION_STRUCTURE = 0x40;
        /// Mesh-shader pipelines ([`CommandBuffer::draw_meshlets`](crate::CommandBuffer::draw_meshlets)).
        /// Not covered by `VERTEX_SHADER`: the mesh stage replaces it rather than following it.
        const MESH_SHADER       = 0x80;

        /// Every stage the rasterizer runs. Composed rather than spelled as a literal, so adding
        /// a stage above cannot leave this behind.
        const ALL_GRAPHICS = Self::VERTEX_SHADER.bits()
            | Self::PIXEL_SHADER.bits()
            | Self::MESH_SHADER.bits()
            | Self::RASTER_COLOR_OUT.bits()
            | Self::RASTER_DEPTH_OUT.bits();
        /// Every stage above.
        const ALL_COMMANDS = Self::ALL_GRAPHICS.bits()
            | Self::COMPUTE.bits()
            | Self::TRANSFER.bits()
            | Self::ACCELERATION_STRUCTURE.bits();
    }
}

bitflags::bitflags! {
    /// Extra cache invalidation; most barriers need none.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct HazardFlags: u32 {
        /// GPU-written indirect args; stalls the command-processor prefetcher.
        const DRAW_ARGUMENTS    = 0x0001;
        /// Descriptor heap written; invalidates the sampler descriptor cache.
        const DESCRIPTORS       = 0x0002;
        /// Depth written by compute; invalidates HiZ/depth caches.
        const DEPTH_STENCIL     = 0x0004;
    }
}

#[cfg(test)]
mod tests {
    use super::StageFlags;

    /// A stage added to the enum but left out of `ALL_COMMANDS` makes every `ALL_COMMANDS`
    /// barrier silently skip it.
    #[test]
    fn all_commands_names_every_stage() {
        assert_eq!(StageFlags::all(), StageFlags::ALL_COMMANDS);
        assert!(StageFlags::ALL_COMMANDS.contains(StageFlags::ALL_GRAPHICS));
    }
}
