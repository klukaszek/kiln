use core::mem;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::ClassType; // retained for parity with other modules
use std::ptr::NonNull;

use std::sync::OnceLock;
use std::time::Instant;

use objc2_metal::MTL4CommandEncoder;
use objc2_metal::MTL4RenderCommandEncoder as _;
use objc2_metal::MTLDrawable;
use objc2_metal::{
    MTL4ArgumentTable, MTL4CommandAllocator, MTL4CommandBuffer, MTL4CommandQueue, MTLBuffer,
    MTLDevice, MTLPrimitiveType, MTLRenderPipelineState, MTLRenderStages, MTLResourceOptions,
};

// Expose swapchain under renderer namespace for a cohesive API
pub mod swapchain {
    pub use crate::kiln::swapchain::*;
}

// Internal use of swapchain traits/types
use crate::kiln::swapchain::{RenderSurface, SwapchainConfig};

pub use crate::kiln::gfx::{PackedFloat3, SceneProperties, VertexInput};

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
    pub fn new<S: RenderSurface + ?Sized>(
        surface: &S,
        _swapchain: SwapchainConfig,
        pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
        vertex_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    ) -> Self {
        let device = surface.device();
        let command_queue = unsafe { device.newMTL4CommandQueue().expect("create queue") };
        let command_allocator = unsafe { device.newCommandAllocator().expect("create allocator") };

        let argument_table = unsafe {
            let at_desc = objc2_metal::MTL4ArgumentTableDescriptor::new();
            at_desc.setMaxBufferBindCount(2);
            at_desc.setMaxTextureBindCount(0);
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
