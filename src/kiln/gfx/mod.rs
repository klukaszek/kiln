use core::ffi::c_void;
use core::mem;
use core::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use core::marker::PhantomData;

use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTLBuffer, MTLDevice, MTLResourceOptions,
};
use zerocopy::{IntoBytes, FromBytes, Immutable};

pub mod shader;
pub mod buffer;

#[derive(Copy, Clone, IntoBytes, FromBytes, Immutable, Debug)]
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

#[derive(Copy, Clone, IntoBytes, FromBytes, Immutable, Debug)]
#[repr(C)]
pub struct SceneProperties {
    pub time: f32,
}

#[derive(Copy, Clone, IntoBytes, FromBytes, Immutable, Debug)]
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

#[derive(Debug, Clone)]
pub struct GpuBuffer<T: IntoBytes + FromBytes + Copy + Immutable> {
    buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    len_elems: usize,
    _pd: PhantomData<T>,
}

impl<T: IntoBytes + FromBytes + Copy + Immutable> GpuBuffer<T> {
    pub fn from_slice(
        device: &ProtocolObject<dyn MTLDevice>,
        data: &[T],
        options: MTLResourceOptions,
    ) -> Self {
        let buf = new_buffer_from_slice(device, data, options);
        Self { buf, len_elems: data.len(), _pd: PhantomData }
    }
    pub fn with_len(
        device: &ProtocolObject<dyn MTLDevice>,
        len_elems: usize,
        options: MTLResourceOptions,
    ) -> Self {
        let byte_len = len_elems * core::mem::size_of::<T>();
        let buf = new_buffer_with_length(device, byte_len, options);
        Self { buf, len_elems, _pd: PhantomData }
    }
    pub fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> { &*self.buf }
    pub fn into_raw(self) -> Retained<ProtocolObject<dyn MTLBuffer>> { self.buf }
    pub fn len(&self) -> usize { self.len_elems }
    pub fn gpu_address(&self) -> u64 { self.buf.gpuAddress() }
    pub fn write_all(&self, data: &[T]) -> Result<(), String> {
        if data.len() > self.len_elems { return Err("slice larger than buffer".into()); }
        write_slice_as_bytes(self.raw(), data);
        Ok(())
    }
    pub fn write_one(&self, index: usize, value: &T) -> Result<(), String> {
        if index >= self.len_elems { return Err("index out of range".into()); }
        let dst = self.buf.contents();
        let offset = index * core::mem::size_of::<T>();
        let bytes = value.as_bytes();
        unsafe {
            core::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                dst.as_ptr().cast::<u8>().add(offset),
                bytes.len(),
            );
        }
        Ok(())
    }
    pub fn from_raw(buf: Retained<ProtocolObject<dyn MTLBuffer>>, len_elems: usize) -> Self {
        Self { buf, len_elems, _pd: PhantomData }
    }
}

pub fn bind_argument_buffer<T>(
    table: &ProtocolObject<dyn MTL4ArgumentTable>,
    index: u32,
    buffer: &GpuBuffer<T>,
) where T: IntoBytes + FromBytes + Copy + Immutable {
    unsafe { table.setAddress_atIndex(buffer.gpu_address(), index as _); }
}

pub trait HasGpuAddress {
    fn gpu_address_u64(&self) -> u64;
}
impl<T: IntoBytes + FromBytes + Copy + Immutable> HasGpuAddress for GpuBuffer<T> {
    fn gpu_address_u64(&self) -> u64 { self.gpu_address() }
}

pub fn bind2<A: HasGpuAddress, B: HasGpuAddress>(
    table: &ProtocolObject<dyn MTL4ArgumentTable>,
    a_index: u32,
    a: &A,
    b_index: u32,
    b: &B,
) {
    unsafe {
        table.setAddress_atIndex(a.gpu_address_u64(), a_index as _);
        table.setAddress_atIndex(b.gpu_address_u64(), b_index as _);
    }
}

#[derive(Debug, Clone)]
pub enum IndexBuffer {
    U16(GpuBuffer<u16>),
    U32(GpuBuffer<u32>),
}
impl IndexBuffer {
    pub fn len(&self) -> usize { match self { Self::U16(b) => b.len(), Self::U32(b) => b.len(), } }
    pub fn gpu_address(&self) -> u64 { match self { Self::U16(b) => b.gpu_address(), Self::U32(b) => b.gpu_address(), } }
}

pub fn new_buffer_from_slice<T: IntoBytes + Immutable>(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[T],
    options: MTLResourceOptions,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let bytes = data.as_bytes();
    let byte_len = bytes.len();
    let ptr = NonNull::new(bytes.as_ptr() as *mut c_void).unwrap();
    unsafe {
        device
            .newBufferWithBytes_length_options(ptr, byte_len, options)
            .expect("create buffer with bytes (AsBytes)")
    }
}

pub fn write_as_bytes<T: IntoBytes + Immutable>(dst_buffer: &ProtocolObject<dyn MTLBuffer>, value: &T) {
    let bytes = value.as_bytes();
    let dst = dst_buffer.contents();
    unsafe {
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), dst.as_ptr().cast::<u8>(), bytes.len());
    }
}

pub fn write_slice_as_bytes<T: IntoBytes + Immutable>(
    dst_buffer: &ProtocolObject<dyn MTLBuffer>,
    data: &[T],
) {
    let bytes = data.as_bytes();
    let dst = dst_buffer.contents();
    unsafe {
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), dst.as_ptr().cast::<u8>(), bytes.len());
    }
}
