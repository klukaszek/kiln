//! Metal shader module: a compiled library plus the entry point to link against.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLLibrary;

/// A compiled `.metallib` plus the entry point a PSO selects from it. Lives here rather than in a
/// module of its own: pipeline creation is the only thing that ever consumes one.
pub struct MetalShaderModule {
    pub(crate) library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pub(crate) entry_point: String,
}
