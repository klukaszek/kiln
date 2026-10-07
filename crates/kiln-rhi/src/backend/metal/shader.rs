//! Metal shader module: a compiled library plus the entry point to link against.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTL4LibraryFunctionDescriptor, MTLLibrary};

pub struct MetalShaderModule {
    pub(crate) library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pub(crate) entry_point: String,
}

impl MetalShaderModule {
    pub(crate) fn function_descriptor(&self) -> Retained<MTL4LibraryFunctionDescriptor> {
        let desc = MTL4LibraryFunctionDescriptor::new();
        desc.setName(Some(&NSString::from_str(&self.entry_point)));
        desc.setLibrary(Some(&self.library));
        desc
    }
}
