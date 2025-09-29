use core::ffi::c_void;
use core::mem;
use core::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::ClassType;

use std::sync::OnceLock;
use std::time::Instant;

use objc2_metal::MTL4CommandEncoder;
use objc2_metal::MTL4RenderCommandEncoder as _;
use objc2_metal::MTLDrawable;
use objc2_metal::MTLPixelFormat;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4BlendState, MTL4CommandAllocator,
    MTL4CommandBuffer, MTL4CommandQueue, MTL4Compiler, MTL4CompilerDescriptor,
    MTL4FunctionDescriptor, MTL4LibraryDescriptor, MTL4LibraryFunctionDescriptor,
    MTL4RenderPipelineDescriptor, MTLBuffer, MTLDevice, MTLLibrary, MTLPrimitiveType,
    MTLRenderPipelineState, MTLRenderStages, MTLResourceOptions,
};

// Use kiln::swapchain types when the example provides the `kiln` module.
pub use crate::kiln::swapchain::{ColorSpace, PresentMode, RenderSurface, SwapchainConfig};

#[derive(Copy, Clone)]
#[repr(C)]
pub struct PackedFloat3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
impl PackedFloat3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct SceneProperties {
    pub time: f32,
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct VertexInput {
    pub position: PackedFloat3,
    pub color: PackedFloat3,
}

#[derive(Debug)]
pub struct Renderer {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    command_allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    argument_table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    scene_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    vertex_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl Renderer {
    pub fn new<S: RenderSurface + ?Sized>(surface: &S, _swapchain: SwapchainConfig) -> Self {
        let device = surface.device();
        let command_queue = unsafe { device.newMTL4CommandQueue().expect("create queue") };
        let command_allocator = unsafe { device.newCommandAllocator().expect("create allocator") };

        let compiler_desc = unsafe { MTL4CompilerDescriptor::new() };
        let compiler = unsafe {
            device
                .newCompilerWithDescriptor_error(&compiler_desc)
                .expect("create compiler")
        };

        let lib_desc = unsafe { MTL4LibraryDescriptor::new() };
        unsafe {
            // Keep kiln self-contained: embed a local copy of the MSL shader
            lib_desc.setSource(Some(objc2_foundation::ns_string!(include_str!(
                ".././shaders/metal4_triangle.metal"
            ))));
            lib_desc.setName(Some(objc2_foundation::ns_string!("shared_renderer_lib")));
        }
        let library: Retained<ProtocolObject<dyn MTLLibrary>> = unsafe {
            compiler
                .newLibraryWithDescriptor_error(&lib_desc)
                .expect("create lib")
        };

        let vfd = unsafe { MTL4LibraryFunctionDescriptor::new() };
        unsafe {
            vfd.setName(Some(objc2_foundation::ns_string!("vertex_main")));
            vfd.setLibrary(Some(&library));
        }
        let ffd = unsafe { MTL4LibraryFunctionDescriptor::new() };
        unsafe {
            ffd.setName(Some(objc2_foundation::ns_string!("fragment_main")));
            ffd.setLibrary(Some(&library));
        }

        let rp_desc = unsafe { MTL4RenderPipelineDescriptor::new() };
        let vfd_base: &MTL4FunctionDescriptor = (&*vfd).as_super();
        let ffd_base: &MTL4FunctionDescriptor = (&*ffd).as_super();
        unsafe {
            rp_desc.setVertexFunctionDescriptor(Some(vfd_base));
            rp_desc.setFragmentFunctionDescriptor(Some(ffd_base));
            let ca0 = rp_desc.colorAttachments().objectAtIndexedSubscript(0);
            ca0.setPixelFormat(surface.color_pixel_format());
            ca0.setBlendingState(MTL4BlendState::Enabled);
        }
        let pipeline_state = unsafe {
            compiler
                .newRenderPipelineStateWithDescriptor_compilerTaskOptions_error(&rp_desc, None)
                .expect("create pipeline")
        };

        let at_desc = unsafe { MTL4ArgumentTableDescriptor::new() };
        unsafe {
            at_desc.setMaxBufferBindCount(2);
            at_desc.setMaxTextureBindCount(0);
        }
        let argument_table = unsafe {
            device
                .newArgumentTableWithDescriptor_error(&at_desc)
                .expect("create arg table")
        };

        let scene_buf_len = mem::size_of::<SceneProperties>();
        let scene_buffer = device
            .newBufferWithLength_options(
                scene_buf_len,
                MTLResourceOptions::CPUCacheModeDefaultCache,
            )
            .expect("create scene buf");

        let verts: [VertexInput; 3] = [
            VertexInput {
                position: PackedFloat3::new(-f32::sqrt(3.0) / 4.0, -0.25, 0.0),
                color: PackedFloat3::new(1.0, 0.0, 0.0),
            },
            VertexInput {
                position: PackedFloat3::new(f32::sqrt(3.0) / 4.0, -0.25, 0.0),
                color: PackedFloat3::new(0.0, 1.0, 0.0),
            },
            VertexInput {
                position: PackedFloat3::new(0.0, 0.5, 0.0),
                color: PackedFloat3::new(0.0, 0.0, 1.0),
            },
        ];
        let verts_len = mem::size_of_val(&verts);
        let verts_ptr = NonNull::new(verts.as_ptr() as *mut c_void).unwrap();
        let vertex_buffer = unsafe {
            device
                .newBufferWithBytes_length_options(
                    verts_ptr,
                    verts_len,
                    MTLResourceOptions::CPUCacheModeDefaultCache,
                )
                .expect("create vbuf")
        };

        Self {
            device,
            command_queue,
            command_allocator,
            pipeline_state,
            argument_table,
            scene_buffer,
            vertex_buffer,
        }
    }

    pub fn draw_frame<S: RenderSurface + ?Sized>(&self, surface: &S, time: f32) {
        // Update scene data (zero-copy via contents())
        let scene = SceneProperties { time };
        let dst = self.scene_buffer.contents();
        let src_ptr = &scene as *const SceneProperties as *const u8;
        unsafe {
            core::ptr::copy_nonoverlapping(
                src_ptr,
                dst.as_ptr().cast::<u8>(),
                mem::size_of::<SceneProperties>(),
            );
        }

        unsafe {
            self.argument_table
                .setAddress_atIndex(self.scene_buffer.gpuAddress(), 0);
        }
        unsafe {
            self.argument_table
                .setAddress_atIndex(self.vertex_buffer.gpuAddress(), 1);
        }

        let Some(rp) = surface.current_mtl4_render_pass_descriptor() else {
            return;
        };
        let Some(drawable) = surface.current_drawable() else {
            return;
        };

        unsafe {
            self.command_allocator.reset();
        }
        let Some(cmd) = (unsafe { self.device.newCommandBuffer() }) else {
            return;
        };
        unsafe {
            cmd.beginCommandBufferWithAllocator(&self.command_allocator);
        }

        // Ensure unified clear behavior across backends
        unsafe {
            let ca0 = rp.colorAttachments().objectAtIndexedSubscript(0);
            ca0.setLoadAction(objc2_metal::MTLLoadAction::Clear);
            ca0.setClearColor(objc2_metal::MTLClearColor {
                red: 0.1,
                green: 0.1,
                blue: 0.12,
                alpha: 1.0,
            });
        }

        let Some(enc) = (unsafe { cmd.renderCommandEncoderWithDescriptor(&rp) }) else {
            return;
        };
        unsafe {
            enc.setRenderPipelineState(&self.pipeline_state);
        }
        unsafe {
            enc.setArgumentTable_atStages(&self.argument_table, MTLRenderStages::Vertex);
        }
        unsafe {
            enc.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
        }
        unsafe {
            enc.endEncoding();
        }
        unsafe {
            cmd.endCommandBuffer();
        }

        unsafe {
            self.command_queue
                .waitForDrawable(ProtocolObject::from_ref(&*drawable));
        }
        let mut arr = [NonNull::from(&*cmd)];
        let ptr = unsafe { NonNull::new_unchecked(arr.as_mut_ptr()) };
        unsafe {
            self.command_queue.commit_count(ptr, 1);
        }
        unsafe {
            self.command_queue
                .signalDrawable(ProtocolObject::from_ref(&*drawable));
        }
        drawable.present();
    }
}

// Monotonic time since first call (seconds). Cross-platform and non-negative.
static START_INSTANT: OnceLock<Instant> = OnceLock::new();
pub fn now_time() -> f32 {
    let start = START_INSTANT.get_or_init(Instant::now);
    start.elapsed().as_secs_f32()
}
