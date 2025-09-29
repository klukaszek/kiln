use core::mem;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::ClassType; // retained for parity with other modules
use std::ptr::NonNull;

use std::sync::OnceLock;
use std::time::Instant;

use crate::kiln::metal;
use crate::kiln::metal::{MTLPrimitiveType, MTLRenderStages, MTLResourceOptions};
use objc2_metal::MTLRenderPipelineState;

// Expose swapchain under renderer namespace for a cohesive API
pub mod swapchain {
    pub use crate::kiln::swapchain::*;
}

// Internal use of swapchain traits/types
use crate::kiln::swapchain::{RenderSurface, SwapchainConfig};

pub use crate::kiln::gfx::buffer;
pub use crate::kiln::gfx::{self, PackedFloat3, SceneProperties, VertexInput};

#[derive(Debug)]
pub struct Renderer {
    device: metal::Device,
    queue: metal::Queue,
    allocator: metal::CommandAllocator,
    pipeline_state: Retained<ProtocolObject<dyn objc2_metal::MTLRenderPipelineState>>,
    argument_table: metal::ArgumentTable,
    scene_buffer: buffer::Uniform<SceneProperties>,
    vertex_buffer: buffer::Vertex<VertexInput>,
    index_buffer: Option<buffer::IndexBuffer>,
    instance_count: usize,
}

impl Renderer {
    pub fn new<S: RenderSurface + ?Sized>(
        surface: &S,
        _swapchain: SwapchainConfig,
        pipeline_state: metal::PipelineState,
        vertex_buffer: buffer::Vertex<VertexInput>,
    ) -> Self {
        let device = metal::Device::from_surface(surface);
        let command_queue = device.new_queue();
        let command_allocator = device.new_command_allocator();
        let argument_table = device.new_argument_table(2, 0);

        // Safe creation via Device: one-element uniform buffer for scene data.
        let scene_buffer = device.uniform_buffer_with_len::<SceneProperties>(
            1,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );

        Self {
            device,
            queue: command_queue,
            allocator: command_allocator,
            pipeline_state: pipeline_state.into_raw(),
            argument_table,
            scene_buffer,
            vertex_buffer,
            index_buffer: None,
            instance_count: 1,
        }
    }

    pub fn draw_frame<S: RenderSurface + ?Sized>(&self, surface: &S, time: f32) {
        // Update scene data (zero-copy via IntoBytes)
        let scene = SceneProperties { time };
        let _ = self.scene_buffer.write_one(0, &scene);

        self.argument_table
            .bind2(0, &self.scene_buffer, 1, &self.vertex_buffer);

        let Some(rp) = metal::RenderPass::from_surface_current(surface) else {
            return;
        };
        let Some(drawable) = metal::Drawable::from_surface_current(surface) else {
            return;
        };

        self.allocator.reset();
        let Some(cmd) = metal::CommandBuffer::begin_with_allocator(&self.device, &self.allocator)
        else {
            return;
        };

        rp.set_clear(
            metal::MTLClearColor {
                red: 0.1,
                green: 0.1,
                blue: 0.12,
                alpha: 1.0,
            },
            metal::MTLLoadAction::Clear,
        );

        let Some(enc) = cmd.render_encoder(&rp) else {
            return;
        };
        enc.set_pipeline(&metal::PipelineState(self.pipeline_state.clone()));
        enc.set_argument_table_at_stages(&self.argument_table, MTLRenderStages::Vertex);
        match &self.index_buffer {
            Some(_ib) => {
                // TODO: Enable once objc2_metal exposes indexed draw on MTL4 encoder.
                if self.instance_count > 1 {
                    enc.draw_primitives_instanced(
                        MTLPrimitiveType::Triangle,
                        0,
                        3,
                        self.instance_count,
                    );
                } else {
                    enc.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
                }
            }
            None => {
                if self.instance_count > 1 {
                    enc.draw_primitives_instanced(
                        MTLPrimitiveType::Triangle,
                        0,
                        3,
                        self.instance_count,
                    );
                } else {
                    enc.draw_primitives(MTLPrimitiveType::Triangle, 0, 3)
                }
            }
        }
        enc.end();
        cmd.end();

        self.queue.wait_for_drawable(&drawable);
        self.queue.commit_one(&cmd);
        self.queue.signal_drawable(&drawable);
        drawable.present();
    }
}

impl Renderer {
    pub fn set_instance_count(&mut self, count: usize) {
        self.instance_count = count.max(1);
    }
    pub fn set_index_buffer(&mut self, ib: buffer::IndexBuffer) {
        self.index_buffer = Some(ib);
    }
}

// Monotonic time since first call (seconds). Cross-platform and non-negative.
static START_INSTANT: OnceLock<Instant> = OnceLock::new();
pub fn now_time() -> f32 {
    let start = START_INSTANT.get_or_init(Instant::now);
    start.elapsed().as_secs_f32()
}
