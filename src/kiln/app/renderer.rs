//
use std::sync::OnceLock;
use std::time::Instant;

use crate::kiln::metal;

// Expose swapchain under renderer namespace for a cohesive API
pub mod swapchain { pub use crate::kiln::app::swapchain::*; }

// Internal use of swapchain traits/types
use crate::kiln::metal::MTLDrawableSource as RenderSurface;

// Re-exports for convenience to examples
pub use crate::kiln::gfx::{self, PackedFloat3, SceneProperties, VertexInput};

#[derive(Debug)]
pub struct Renderer {
    device: metal::Device,
    queue: metal::Queue,
    allocator: metal::CommandAllocator,
}

impl Renderer {
    // Construct renderer context from a surface; stays zero-copy by relying on kiln::metal wrappers
    pub fn new<S: RenderSurface + ?Sized>(surface: &S) -> Self {
        let device = metal::Device::from_surface(surface);
        let queue = device.new_queue();
        let allocator = device.new_command_allocator();
        Self {
            device,
            queue,
            allocator,
        }
    }

    pub fn device(&self) -> &metal::Device {
        &self.device
    }
    pub fn queue(&self) -> &metal::Queue {
        &self.queue
    }

    // Begin a frame: acquires render pass + drawable and creates a command buffer.
    pub fn begin_frame<S: RenderSurface + ?Sized>(&self, surface: &S) -> Option<RenderFrame<'_>> {
        let rp = metal::RenderPass::from_surface_current(surface)?;
        let drawable = metal::Drawable::from_surface_current(surface)?;

        self.allocator.reset();
        let cmd = metal::CommandBuffer::begin_with_allocator(&self.device, &self.allocator)?;

        // Default clear to a dark gray; can be overridden via set_clear on the frame
        rp.set_clear(
            metal::MTLClearColor {
                red: 0.1,
                green: 0.1,
                blue: 0.12,
                alpha: 1.0,
            },
            metal::MTLLoadAction::Clear,
        );

        Some(RenderFrame {
            renderer: self,
            cmd,
            rp,
            drawable,
            ended: false,
        })
    }
}

pub struct RenderFrame<'a> {
    renderer: &'a Renderer,
    cmd: metal::CommandBuffer,
    rp: metal::RenderPass,
    drawable: metal::Drawable,
    ended: bool,
}
impl<'a> RenderFrame<'a> {
    pub fn set_clear(&self, color: metal::MTLClearColor, load: metal::MTLLoadAction) {
        self.rp.set_clear(color, load)
    }
    pub fn encoder(&self) -> Option<metal::RenderEncoder> {
        self.cmd.render_encoder(&self.rp)
    }
    pub fn end(&mut self) {
        if !self.ended {
            self.cmd.end();
            self.ended = true;
        }
    }
}
impl<'a> Drop for RenderFrame<'a> {
    fn drop(&mut self) {
        if !self.ended {
            self.cmd.end();
            self.ended = true;
        }
        self.renderer.queue.wait_for_drawable(&self.drawable);
        self.renderer.queue.commit_one(&self.cmd);
        self.renderer.queue.signal_drawable(&self.drawable);
        self.drawable.present();
    }
}

// Monotonic time since first call (seconds). Cross-platform and non-negative.
static START_INSTANT: OnceLock<Instant> = OnceLock::new();
pub fn now_time() -> f32 {
    let start = START_INSTANT.get_or_init(Instant::now);
    start.elapsed().as_secs_f32()
}
