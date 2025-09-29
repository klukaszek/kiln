use objc2::rc::Retained;
use objc2::ClassType;
use objc2::runtime::ProtocolObject;

use objc2_foundation::NSString;

use objc2_metal::{
    MTL4Compiler, MTL4CompilerDescriptor, MTL4FunctionDescriptor, MTL4LibraryDescriptor,
    MTL4LibraryFunctionDescriptor, MTL4RenderPipelineDescriptor, MTLLibrary, MTLRenderPipelineState,
    MTLPixelFormat, MTLDevice,
};

pub type Library = Retained<ProtocolObject<dyn MTLLibrary>>;
pub type PipelineState = Retained<ProtocolObject<dyn MTLRenderPipelineState>>;

pub fn from_source(
    device: &ProtocolObject<dyn MTLDevice>,
    name: &str,
    source: &str,
) -> Result<Library, String> {
    unsafe {
        let compiler_desc = MTL4CompilerDescriptor::new();
        let compiler: Retained<ProtocolObject<dyn MTL4Compiler>> = device
            .newCompilerWithDescriptor_error(&compiler_desc)
            .map_err(|_| "failed to create MTL4 compiler".to_string())?;

        let lib_desc = MTL4LibraryDescriptor::new();
        let src_ns = NSString::from_str(source);
        let name_ns = NSString::from_str(name);
        lib_desc.setSource(Some(&src_ns));
        lib_desc.setName(Some(&name_ns));

        compiler
            .newLibraryWithDescriptor_error(&lib_desc)
            .map_err(|_| "failed to compile Metal library".to_string())
    }
}

pub fn new(device: &ProtocolObject<dyn MTLDevice>, name: &str, source: &str) -> Result<Library, String> {
    from_source(device, name, source)
}

pub fn pipeline_state(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &Library,
    vertex_fn: &str,
    fragment_fn: &str,
    color_format: MTLPixelFormat,
) -> Result<PipelineState, String> {
    unsafe {
        let compiler_desc = MTL4CompilerDescriptor::new();
        let compiler: Retained<ProtocolObject<dyn MTL4Compiler>> = device
            .newCompilerWithDescriptor_error(&compiler_desc)
            .map_err(|_| "failed to create MTL4 compiler".to_string())?;

        let vfd = MTL4LibraryFunctionDescriptor::new();
        let vname = NSString::from_str(vertex_fn);
        vfd.setName(Some(&vname));
        vfd.setLibrary(Some(library));

        let ffd = MTL4LibraryFunctionDescriptor::new();
        let fname = NSString::from_str(fragment_fn);
        ffd.setName(Some(&fname));
        ffd.setLibrary(Some(library));

        let rp_desc = MTL4RenderPipelineDescriptor::new();
        let vfd_base: &MTL4FunctionDescriptor = (&*vfd).as_super();
        let ffd_base: &MTL4FunctionDescriptor = (&*ffd).as_super();
        rp_desc.setVertexFunctionDescriptor(Some(vfd_base));
        rp_desc.setFragmentFunctionDescriptor(Some(ffd_base));
        let ca0 = rp_desc.colorAttachments().objectAtIndexedSubscript(0);
        ca0.setPixelFormat(color_format);

        compiler
            .newRenderPipelineStateWithDescriptor_compilerTaskOptions_error(&rp_desc, None)
            .map_err(|_| "failed to create pipeline state".to_string())
    }
}

pub struct RenderPipelineBuilder<'a> {
    device: &'a ProtocolObject<dyn MTLDevice>,
    library: &'a Library,
    vertex_name: Option<String>,
    fragment_name: Option<String>,
    color_format: Option<MTLPixelFormat>,
}

impl<'a> RenderPipelineBuilder<'a> {
    pub fn new(device: &'a ProtocolObject<dyn MTLDevice>, library: &'a Library) -> Self {
        Self { device, library, vertex_name: None, fragment_name: None, color_format: None }
    }
    pub fn vertex(mut self, name: &str) -> Self { self.vertex_name = Some(name.to_string()); self }
    pub fn fragment(mut self, name: &str) -> Self { self.fragment_name = Some(name.to_string()); self }
    pub fn color_format(mut self, pf: MTLPixelFormat) -> Self { self.color_format = Some(pf); self }
    pub fn build(self) -> Result<PipelineState, String> {
        let v = self.vertex_name.ok_or_else(|| "vertex function name not set".to_string())?;
        let f = self.fragment_name.ok_or_else(|| "fragment function name not set".to_string())?;
        let cf = self.color_format.ok_or_else(|| "color format not set".to_string())?;
        pipeline_state(self.device, self.library, &v, &f, cf)
    }
}
