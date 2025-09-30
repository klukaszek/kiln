use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use zerocopy::{FromBytes, Immutable, IntoBytes};

use objc2_metal::MTLDevice;

use super::MTLDrawableSource as RawRenderSurface;
use super::pipeline;

pub use objc2_metal::MTLResourceOptions;

#[derive(Debug, Clone)]
pub struct Device {
    pub(crate) raw: Retained<ProtocolObject<dyn MTLDevice>>,
}
impl Device {
    pub fn from_surface<T: RawRenderSurface + ?Sized>(surface: &T) -> Self {
        Self { raw: surface.device() }
    }

    pub fn compile_library_from_source(&self, name: &str, source: &str) -> Result<super::Library, String> {
        pipeline::compile_library_from_source(&self.raw, name, source)
    }

    pub fn pipeline_builder<'a>(&'a self, library: &'a super::Library) -> super::RenderPipelineBuilder<'a> {
        super::RenderPipelineBuilder::new(self, library)
    }

    pub fn vertex_buffer_from_slice<T>(&self, data: &[T], options: MTLResourceOptions) -> super::Vertex<T>
    where
        T: IntoBytes + FromBytes + Copy + Immutable,
    {
        super::Vertex::from_slice(&self.raw, data, options)
    }

    pub fn uniform_buffer_with_len<T>(&self, len_elems: usize, options: MTLResourceOptions) -> super::Uniform<T>
    where
        T: IntoBytes + FromBytes + Copy + Immutable,
    {
        super::Uniform::with_len(&self.raw, len_elems, options)
    }

    pub fn index_buffer_from_u16(&self, data: &[u16], options: MTLResourceOptions) -> super::IndexBuffer {
        super::IndexBuffer::from_u16(&self.raw, data, options)
    }
    pub fn index_buffer_from_u32(&self, data: &[u32], options: MTLResourceOptions) -> super::IndexBuffer {
        super::IndexBuffer::from_u32(&self.raw, data, options)
    }

    pub fn new_queue(&self) -> super::Queue {
        let raw = unsafe { self.raw.newMTL4CommandQueue().expect("create queue") };
        super::Queue { raw }
    }
    pub fn new_command_allocator(&self) -> super::CommandAllocator {
        let raw = unsafe { self.raw.newCommandAllocator().expect("create allocator") };
        super::CommandAllocator { raw }
    }
    pub fn new_argument_buffer(&self, max_buffers: u32, max_textures: u32) -> super::ArgumentBuffer {
        let raw = unsafe {
            let desc = objc2_metal::MTL4ArgumentTableDescriptor::new();
            desc.setMaxBufferBindCount(max_buffers as _);
            desc.setMaxTextureBindCount(max_textures as _);
            self.raw
                .newArgumentTableWithDescriptor_error(&desc)
                .expect("create argument table")
        };
        super::ArgumentBuffer { raw }
    }
}
