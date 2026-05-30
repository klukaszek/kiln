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
    /// Special-case cache-invalidation hints added to a barrier; most barriers need none.
    ///
    /// - `DRAW_ARGUMENTS`: GPU-written indirect args — stall the command-processor prefetcher.
    /// - `DESCRIPTORS`: descriptor heap written — invalidate the sampler descriptor cache.
    /// - `DEPTH_STENCIL`: depth written by compute — invalidate HiZ/depth caches.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct HazardFlags: u32 {
        const DRAW_ARGUMENTS    = 0x0001;
        const DESCRIPTORS       = 0x0002;
        const DEPTH_STENCIL     = 0x0004;
    }
}
