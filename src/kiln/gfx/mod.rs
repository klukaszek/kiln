use core::ffi::c_void;
use core::mem;
use core::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;

use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTLBuffer, MTLDevice, MTLResourceOptions,
};

pub mod shader;

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

pub type ArgumentTable = Retained<ProtocolObject<dyn MTL4ArgumentTable>>;

pub fn new_argument_table(
    device: &ProtocolObject<dyn MTLDevice>,
    max_buffers: u32,
    max_textures: u32,
) -> Result<ArgumentTable, String> {
    unsafe {
        let desc = MTL4ArgumentTableDescriptor::new();
        desc.setMaxBufferBindCount(max_buffers as _);
        desc.setMaxTextureBindCount(max_textures as _);
        device
            .newArgumentTableWithDescriptor_error(&desc)
            .map_err(|_| "failed to create argument table".to_string())
    }
}

pub fn new_buffer_with_length(
    device: &ProtocolObject<dyn MTLDevice>,
    len: usize,
    options: MTLResourceOptions,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    device
        .newBufferWithLength_options(len, options)
        .expect("create buffer with length")
}

pub fn new_buffer_with_bytes<T: Copy>(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[T],
    options: MTLResourceOptions,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let byte_len = mem::size_of_val(data);
    let ptr = NonNull::new(data.as_ptr() as *mut c_void).unwrap();
    unsafe {
        device
            .newBufferWithBytes_length_options(ptr, byte_len, options)
            .expect("create buffer with bytes")
    }
}

pub fn write_struct<T: Copy>(dst_buffer: &ProtocolObject<dyn MTLBuffer>, value: &T) {
    let dst = dst_buffer.contents();
    let src_ptr = value as *const T as *const u8;
    unsafe {
        core::ptr::copy_nonoverlapping(src_ptr, dst.as_ptr().cast::<u8>(), mem::size_of::<T>());
    }
}
