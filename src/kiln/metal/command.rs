use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;

use objc2_metal::{MTL4CommandAllocator, MTL4CommandBuffer, MTL4CommandQueue, MTLDevice};

use super::pass::RenderPass;

#[derive(Debug)]
pub struct Queue { pub(crate) raw: Retained<ProtocolObject<dyn MTL4CommandQueue>> }
impl Queue {
    pub fn wait_for_drawable(&self, drawable: &super::Drawable) {
        unsafe { self.raw.waitForDrawable(ProtocolObject::from_ref(&*drawable.raw)) }
    }
    pub fn commit_one(&self, cmd: &CommandBuffer) {
        let mut arr = [core::ptr::NonNull::from(&*cmd.raw)];
        let ptr = unsafe { core::ptr::NonNull::new_unchecked(arr.as_mut_ptr()) };
        unsafe { self.raw.commit_count(ptr, 1) }
    }
    pub fn signal_drawable(&self, drawable: &super::Drawable) {
        unsafe { self.raw.signalDrawable(ProtocolObject::from_ref(&*drawable.raw)) }
    }
}

#[derive(Debug)]
pub struct CommandAllocator { pub(crate) raw: Retained<ProtocolObject<dyn MTL4CommandAllocator>> }
impl CommandAllocator { pub fn reset(&self) { unsafe { self.raw.reset() } } }

pub struct CommandBuffer { pub(crate) raw: Retained<ProtocolObject<dyn MTL4CommandBuffer>> }
impl CommandBuffer {
    pub fn begin_with_allocator(device: &super::Device, allocator: &CommandAllocator) -> Option<Self> {
        let Some(raw) = (unsafe { device.raw.newCommandBuffer() }) else { return None; };
        unsafe { raw.beginCommandBufferWithAllocator(&allocator.raw) };
        Some(Self { raw })
    }
    pub fn end(&self) { unsafe { self.raw.endCommandBuffer() } }
    pub fn render_encoder(&self, rp: &RenderPass) -> Option<super::RenderEncoder> {
        let Some(enc) = (unsafe { self.raw.renderCommandEncoderWithDescriptor(&rp.raw) }) else { return None; };
        Some(super::RenderEncoder { raw: enc })
    }
}
