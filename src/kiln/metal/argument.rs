use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;

use objc2_metal::MTL4ArgumentTable;
use zerocopy::{FromBytes, Immutable, IntoBytes};

#[derive(Debug)]
pub struct ArgumentBuffer { pub(crate) raw: Retained<ProtocolObject<dyn MTL4ArgumentTable>> }
impl ArgumentBuffer {
    pub fn bind2<A: Bindable, B: Bindable>(&self, a_index: u32, a: &A, b_index: u32, b: &B) {
        unsafe {
            self.raw.setAddress_atIndex(a.gpu_address_u64(), a_index as _);
            self.raw.setAddress_atIndex(b.gpu_address_u64(), b_index as _);
        }
    }
}

pub trait Bindable { fn gpu_address_u64(&self) -> u64; }
impl<T: IntoBytes + FromBytes + Copy + Immutable> Bindable for crate::kiln::metal::Uniform<T> {
    fn gpu_address_u64(&self) -> u64 { self.gpu_address() }
}
impl<T: IntoBytes + FromBytes + Copy + Immutable> Bindable for crate::kiln::metal::Vertex<T> {
    fn gpu_address_u64(&self) -> u64 { self.gpu_address() }
}
impl Bindable for crate::kiln::metal::IndexBuffer {
    fn gpu_address_u64(&self) -> u64 { self.gpu_address() }
}
