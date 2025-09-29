//! Safe, curated Metal facade for Kiln.
//!
//! Consolidates safe enums/structs and high-level helpers, avoiding exposure
//! of unsafe constructors and protocol types. Where data crosses the FFI
//! boundary, we require zerocopy bounds (IntoBytes/FromBytes/Immutable).

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;

use zerocopy::{FromBytes, Immutable, IntoBytes};

use objc2_metal::{
    MTLDevice,
    MTL4CommandQueue,
    MTL4CommandAllocator,
    MTL4CommandBuffer,
    MTL4RenderCommandEncoder,
    MTL4CommandEncoder,
    MTL4ArgumentTable,
    MTLDrawable,
};
use crate::kiln::gfx::{self, shader};
use crate::kiln::swapchain::RenderSurface as RawRenderSurface;

pub use crate::kiln::gfx::{PackedFloat3, SceneProperties, VertexInput};

// Safe data-only Metal types (no constructors that require `unsafe`).
pub use objc2_metal::{
    MTLClearColor,
    MTLLoadAction,
    MTLPixelFormat,
    MTLPrimitiveType,
    MTLRenderStages,
    MTLResourceOptions,
    MTLStoreAction,
};

#[derive(Debug)]
pub struct Device {
    raw: Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>,
}
impl Device {
    pub fn from_surface<T: RawRenderSurface + ?Sized>(surface: &T) -> Self {
        Self { raw: surface.device() }
    }

    // Create a library from MSL source using the device's MTL4 compiler.
    pub fn compile_library_from_source(&self, name: &str, source: &str) -> Result<Library, String> {
        shader::from_source(&self.raw, name, source).map(Library)
    }

    // Pipeline builder using only safe arguments and enums.
    pub fn pipeline_builder<'a>(&'a self, library: &'a Library) -> RenderPipelineBuilder<'a> {
        RenderPipelineBuilder::new(self, library)
    }

    // Zero-copy safe vertex buffer creation from a slice of PODs.
    pub fn vertex_buffer_from_slice<T>(&self, data: &[T], options: MTLResourceOptions) -> buffer::Vertex<T>
    where
        T: IntoBytes + FromBytes + Copy + Immutable,
    {
        buffer::Vertex::from_slice(&self.raw, data, options)
    }

    // Zero-copy safe uniform buffer with fixed length (elements of T).
    pub fn uniform_buffer_with_len<T>(&self, len_elems: usize, options: MTLResourceOptions) -> buffer::Uniform<T>
    where
        T: IntoBytes + FromBytes + Copy + Immutable,
    {
        buffer::Uniform::with_len(&self.raw, len_elems, options)
    }

    // Zero-copy safe index buffer creation helpers.
    pub fn index_buffer_from_u16(&self, data: &[u16], options: MTLResourceOptions) -> buffer::IndexBuffer {
        buffer::IndexBuffer::from_u16(&self.raw, data, options)
    }
    pub fn index_buffer_from_u32(&self, data: &[u32], options: MTLResourceOptions) -> buffer::IndexBuffer {
        buffer::IndexBuffer::from_u32(&self.raw, data, options)
    }

    pub fn new_queue(&self) -> Queue {
        let raw = unsafe { self.raw.newMTL4CommandQueue().expect("create queue") };
        Queue { raw }
    }

    pub fn new_command_allocator(&self) -> CommandAllocator {
        let raw = unsafe { self.raw.newCommandAllocator().expect("create allocator") };
        CommandAllocator { raw }
    }

    pub fn new_argument_table(&self, max_buffers: u32, max_textures: u32) -> ArgumentTable {
        let raw = unsafe {
            let desc = objc2_metal::MTL4ArgumentTableDescriptor::new();
            desc.setMaxBufferBindCount(max_buffers as _);
            desc.setMaxTextureBindCount(max_textures as _);
            self.raw
                .newArgumentTableWithDescriptor_error(&desc)
                .expect("create argument table")
        };
        ArgumentTable { raw }
    }
}

pub struct Library(gfx::shader::Library);
pub struct PipelineState(pub gfx::shader::PipelineState);
impl PipelineState {
    pub fn into_raw(self) -> Retained<ProtocolObject<dyn objc2_metal::MTLRenderPipelineState>> { self.0 }
    pub fn raw(&self) -> &ProtocolObject<dyn objc2_metal::MTLRenderPipelineState> { &*self.0 }
}

pub struct RenderPipelineBuilder<'a> {
    device: &'a Device,
    library: &'a Library,
    vertex_name: Option<String>,
    fragment_name: Option<String>,
    color_format: Option<MTLPixelFormat>,
}

impl<'a> RenderPipelineBuilder<'a> {
    fn new(device: &'a Device, library: &'a Library) -> Self {
        Self { device, library, vertex_name: None, fragment_name: None, color_format: None }
    }
    pub fn vertex(mut self, name: &str) -> Self { self.vertex_name = Some(name.to_string()); self }
    pub fn fragment(mut self, name: &str) -> Self { self.fragment_name = Some(name.to_string()); self }
    pub fn color_format(mut self, pf: MTLPixelFormat) -> Self { self.color_format = Some(pf); self }
    pub fn build(self) -> Result<PipelineState, String> {
        let v = self.vertex_name.ok_or_else(|| "vertex function name not set".to_string())?;
        let f = self.fragment_name.ok_or_else(|| "fragment function name not set".to_string())?;
        let cf = self.color_format.ok_or_else(|| "color format not set".to_string())?;
        shader::pipeline_state(&self.device.raw, &self.library.0, &v, &f, cf).map(PipelineState)
    }
}

// Re-export the buffer module under metal for convenience (safe constructors
// are provided via Device where practical, but the types themselves are safe).
pub mod buffer {
    pub use crate::kiln::gfx::buffer::{IndexBuffer, Uniform, Vertex};
}

#[derive(Debug)]
pub struct Queue {
    raw: Retained<ProtocolObject<dyn objc2_metal::MTL4CommandQueue>>,
}
impl Queue {
    pub fn wait_for_drawable(&self, drawable: &Drawable) {
        unsafe { self.raw.waitForDrawable(ProtocolObject::from_ref(&*drawable.raw)) }
    }
    pub fn commit_one(&self, cmd: &CommandBuffer) {
        // Safe wrapper for commit_count(&[&cmd], 1)
        let mut arr = [core::ptr::NonNull::from(&*cmd.raw)];
        let ptr = unsafe { core::ptr::NonNull::new_unchecked(arr.as_mut_ptr()) };
        unsafe { self.raw.commit_count(ptr, 1) }
    }
    pub fn signal_drawable(&self, drawable: &Drawable) {
        unsafe { self.raw.signalDrawable(ProtocolObject::from_ref(&*drawable.raw)) }
    }
}

#[derive(Debug)]
pub struct CommandAllocator { raw: Retained<ProtocolObject<dyn objc2_metal::MTL4CommandAllocator>> }
impl CommandAllocator { pub fn reset(&self) { unsafe { self.raw.reset() } } }

pub struct CommandBuffer { raw: Retained<ProtocolObject<dyn objc2_metal::MTL4CommandBuffer>> }
impl CommandBuffer {
    pub fn begin_with_allocator(device: &Device, allocator: &CommandAllocator) -> Option<Self> {
        let Some(raw) = (unsafe { device.raw.newCommandBuffer() }) else { return None; };
        unsafe { raw.beginCommandBufferWithAllocator(&allocator.raw) };
        Some(Self { raw })
    }
    pub fn end(&self) { unsafe { self.raw.endCommandBuffer() } }
    pub fn render_encoder(&self, rp: &RenderPass) -> Option<RenderEncoder> {
        let Some(enc) = (unsafe { self.raw.renderCommandEncoderWithDescriptor(&rp.raw) }) else { return None; };
        Some(RenderEncoder { raw: enc })
    }
}

pub struct RenderPass { raw: Retained<objc2_metal::MTL4RenderPassDescriptor> }
impl RenderPass {
    pub fn from_surface_current<T: RawRenderSurface + ?Sized>(surface: &T) -> Option<Self> {
        surface.current_mtl4_render_pass_descriptor().map(|raw| Self { raw })
    }
    pub fn set_clear(&self, color: MTLClearColor, load: MTLLoadAction) {
        unsafe {
            let ca0 = self.raw.colorAttachments().objectAtIndexedSubscript(0);
            ca0.setLoadAction(load);
            ca0.setClearColor(color);
        }
    }
}

pub struct RenderEncoder { raw: Retained<ProtocolObject<dyn objc2_metal::MTL4RenderCommandEncoder>> }
impl RenderEncoder {
    pub fn set_pipeline(&self, pso: &PipelineState) { unsafe { self.raw.setRenderPipelineState(pso.raw()) } }
    pub fn set_argument_table_at_stages(&self, table: &ArgumentTable, stages: MTLRenderStages) {
        unsafe { self.raw.setArgumentTable_atStages(&table.raw, stages) }
    }
    pub fn draw_primitives(&self, prim: MTLPrimitiveType, start: usize, count: usize) {
        unsafe { self.raw.drawPrimitives_vertexStart_vertexCount(prim, start, count) }
    }
    pub fn draw_primitives_instanced(&self, prim: MTLPrimitiveType, start: usize, count: usize, instances: usize) {
        unsafe { self.raw.drawPrimitives_vertexStart_vertexCount_instanceCount(prim, start, count, instances) }
    }
    pub fn end(self) { unsafe { self.raw.endEncoding() } }
}

pub struct Drawable { raw: Retained<ProtocolObject<dyn objc2_quartz_core::CAMetalDrawable>> }
impl Drawable {
    pub fn from_surface_current<T: RawRenderSurface + ?Sized>(surface: &T) -> Option<Self> {
        surface.current_drawable().map(|raw| Self { raw })
    }
    pub fn present(&self) { self.raw.present() }
}

#[derive(Debug)]
pub struct ArgumentTable { raw: Retained<ProtocolObject<dyn objc2_metal::MTL4ArgumentTable>> }
impl ArgumentTable {
    pub fn bind2<A: Bindable, B: Bindable>(&self, a_index: u32, a: &A, b_index: u32, b: &B) {
        unsafe {
            self.raw.setAddress_atIndex(a.gpu_address_u64(), a_index as _);
            self.raw.setAddress_atIndex(b.gpu_address_u64(), b_index as _);
        }
    }
}

pub trait Bindable { fn gpu_address_u64(&self) -> u64; }
impl<T: IntoBytes + FromBytes + Copy + Immutable> Bindable for crate::kiln::gfx::buffer::Uniform<T> {
    fn gpu_address_u64(&self) -> u64 { self.gpu_address() }
}
impl<T: IntoBytes + FromBytes + Copy + Immutable> Bindable for crate::kiln::gfx::buffer::Vertex<T> {
    fn gpu_address_u64(&self) -> u64 { self.gpu_address() }
}
impl Bindable for crate::kiln::gfx::buffer::IndexBuffer {
    fn gpu_address_u64(&self) -> u64 { self.gpu_address() }
}
