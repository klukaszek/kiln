use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLLibrary;

pub struct MetalShaderModule {
    pub(crate) library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pub(crate) entry_point: String,
}
