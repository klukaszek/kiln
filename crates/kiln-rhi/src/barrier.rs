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
        const ALL_GRAPHICS      = 0x1B; // VERTEX | PIXEL | RASTER_COLOR_OUT | RASTER_DEPTH_OUT
        const ALL_COMMANDS      = 0x3F; // all six stages
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
