bitflags::bitflags! {
    /// Pipeline stage flags for barrier synchronization.
    ///
    /// Matches the Aaltonen "No Graphics API" stage model.
    /// Stage-only barriers — no per-resource state tracking.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct StageFlags: u32 {
        const VERTEX_SHADER     = 0x0001;
        const PIXEL_SHADER      = 0x0002;
        const COMPUTE           = 0x0004;
        const RASTER_COLOR_OUT  = 0x0008;
        const RASTER_DEPTH_OUT  = 0x0010;
        const TRANSFER          = 0x0020;
        const ALL_GRAPHICS      = 0x00FF;
        const ALL_COMMANDS      = 0xFFFF;
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
