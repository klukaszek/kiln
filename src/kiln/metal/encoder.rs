use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;

use objc2_metal::{MTL4CommandEncoder, MTL4RenderCommandEncoder};

use super::{ArgumentBuffer, MTLPrimitiveType, MTLRenderStages, PipelineState};

pub struct RenderEncoder { pub(crate) raw: Retained<ProtocolObject<dyn MTL4RenderCommandEncoder>> }
impl RenderEncoder {
    pub fn set_pipeline(&self, pso: &PipelineState) { unsafe { self.raw.setRenderPipelineState(pso.raw()) } }
    pub fn set_argument_table_at_stages(&self, table: &ArgumentBuffer, stages: MTLRenderStages) {
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
