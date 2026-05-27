bitflags::bitflags! {
    /// Pipeline stage flags for barrier synchronization.
    ///
    /// Matches the Aaltonen "No Graphics API" stage model.
    /// Stage-only barriers — no per-resource state tracking.
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
    /// Hazard flags for barrier special-case cache invalidation.
    ///
    /// Modern GPUs flush the majority of non-coherent caches automatically on
    /// every barrier. Only these three cases require explicit hints:
    ///
    /// - `DRAW_ARGUMENTS`: GPU-written indirect args — stalls the command
    ///   processor prefetcher so it sees the updated draw/dispatch parameters.
    /// - `DESCRIPTORS`: Texture descriptor heap was written — invalidates the
    ///   sampler's internal descriptor cache.
    /// - `DEPTH_STENCIL`: Depth buffer written by compute — invalidates HiZ
    ///   and depth caches that are not automatically flushed.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct HazardFlags: u32 {
        const DRAW_ARGUMENTS    = 0x0001;
        const DESCRIPTORS       = 0x0002;
        const DEPTH_STENCIL     = 0x0004;
    }
}
