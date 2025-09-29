use core::marker::PhantomData;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTL4ArgumentTable};
use zerocopy::{IntoBytes, FromBytes, Immutable};

#[derive(Debug, Clone)]
struct Generic<T: IntoBytes + FromBytes + Copy + Immutable> {
    buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    len_elems: usize,
    _pd: PhantomData<T>,
}

impl<T: IntoBytes + FromBytes + Copy + Immutable> Generic<T> {
    fn from_slice(
        device: &ProtocolObject<dyn MTLDevice>,
        data: &[T],
        options: MTLResourceOptions,
    ) -> Self {
        let bytes = data.as_bytes();
        let len = bytes.len();
        let ptr = core::ptr::NonNull::new(bytes.as_ptr() as *mut core::ffi::c_void).unwrap();
        let buf = unsafe { device.newBufferWithBytes_length_options(ptr, len, options) }
            .expect("create buffer from slice");
        Self { buf, len_elems: data.len(), _pd: PhantomData }
    }
    fn with_len(
        device: &ProtocolObject<dyn MTLDevice>,
        len_elems: usize,
        options: MTLResourceOptions,
    ) -> Self {
        let len = len_elems * core::mem::size_of::<T>();
        let buf = device.newBufferWithLength_options(len, options)
            .expect("create buffer with length");
        Self { buf, len_elems, _pd: PhantomData }
    }
    fn from_raw(buf: Retained<ProtocolObject<dyn MTLBuffer>>, len_elems: usize) -> Self {
        Self { buf, len_elems, _pd: PhantomData }
    }
    fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> { &*self.buf }
    fn into_raw(self) -> Retained<ProtocolObject<dyn MTLBuffer>> { self.buf }
    fn len(&self) -> usize { self.len_elems }
    fn gpu_address(&self) -> u64 { self.buf.gpuAddress() }
    fn write_all(&self, data: &[T]) -> Result<(), String> {
        if data.len() > self.len_elems { return Err("slice larger than buffer".into()); }
        let bytes = data.as_bytes();
        let dst = self.buf.contents();
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), dst.as_ptr().cast::<u8>(), bytes.len()); }
        Ok(())
    }
    fn write_one(&self, index: usize, value: &T) -> Result<(), String> {
        if index >= self.len_elems { return Err("index out of range".into()); }
        let dst = self.buf.contents();
        let offset = index * core::mem::size_of::<T>();
        let bytes = value.as_bytes();
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), dst.as_ptr().cast::<u8>().add(offset), bytes.len()); }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Uniform<T: IntoBytes + FromBytes + Copy + Immutable>(Generic<T>);
impl<T: IntoBytes + FromBytes + Copy + Immutable> Uniform<T> {
    pub fn from_slice(device: &ProtocolObject<dyn MTLDevice>, data: &[T], options: MTLResourceOptions) -> Self { Self(Generic::from_slice(device, data, options)) }
    pub fn with_len(device: &ProtocolObject<dyn MTLDevice>, len_elems: usize, options: MTLResourceOptions) -> Self { Self(Generic::with_len(device, len_elems, options)) }
    pub fn from_raw(buf: Retained<ProtocolObject<dyn MTLBuffer>>, len_elems: usize) -> Self { Self(Generic::from_raw(buf, len_elems)) }
    pub fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> { self.0.raw() }
    pub fn into_raw(self) -> Retained<ProtocolObject<dyn MTLBuffer>> { self.0.into_raw() }
    pub fn len(&self) -> usize { self.0.len() }
    pub fn gpu_address(&self) -> u64 { self.0.gpu_address() }
    pub fn write_one(&self, index: usize, value: &T) -> Result<(), String> { self.0.write_one(index, value) }
    pub fn write_all(&self, data: &[T]) -> Result<(), String> { self.0.write_all(data) }
}

#[derive(Debug, Clone)]
pub struct Vertex<T: IntoBytes + FromBytes + Copy + Immutable>(Generic<T>);
impl<T: IntoBytes + FromBytes + Copy + Immutable> Vertex<T> {
    pub fn from_slice(device: &ProtocolObject<dyn MTLDevice>, data: &[T], options: MTLResourceOptions) -> Self { Self(Generic::from_slice(device, data, options)) }
    pub fn with_len(device: &ProtocolObject<dyn MTLDevice>, len_elems: usize, options: MTLResourceOptions) -> Self { Self(Generic::with_len(device, len_elems, options)) }
    pub fn from_raw(buf: Retained<ProtocolObject<dyn MTLBuffer>>, len_elems: usize) -> Self { Self(Generic::from_raw(buf, len_elems)) }
    pub fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> { self.0.raw() }
    pub fn into_raw(self) -> Retained<ProtocolObject<dyn MTLBuffer>> { self.0.into_raw() }
    pub fn len(&self) -> usize { self.0.len() }
    pub fn gpu_address(&self) -> u64 { self.0.gpu_address() }
    pub fn write_all(&self, data: &[T]) -> Result<(), String> { self.0.write_all(data) }
}

#[derive(Debug, Clone)]
pub enum IndexBuffer {
    U16(Generic<u16>),
    U32(Generic<u32>),
}
impl IndexBuffer {
    pub fn from_u16(device: &ProtocolObject<dyn MTLDevice>, data: &[u16], options: MTLResourceOptions) -> Self {
        Self::U16(Generic::from_slice(device, data, options))
    }
    pub fn from_u32(device: &ProtocolObject<dyn MTLDevice>, data: &[u32], options: MTLResourceOptions) -> Self {
        Self::U32(Generic::from_slice(device, data, options))
    }
    pub fn len(&self) -> usize { match self { Self::U16(b) => b.len(), Self::U32(b) => b.len() } }
    pub fn gpu_address(&self) -> u64 { match self { Self::U16(b) => b.gpu_address(), Self::U32(b) => b.gpu_address() } }
}

pub trait HasGpuAddress { fn gpu_address_u64(&self) -> u64; }
impl<T: IntoBytes + FromBytes + Copy + Immutable> HasGpuAddress for Uniform<T> { fn gpu_address_u64(&self) -> u64 { self.gpu_address() } }
impl<T: IntoBytes + FromBytes + Copy + Immutable> HasGpuAddress for Vertex<T> { fn gpu_address_u64(&self) -> u64 { self.gpu_address() } }
impl HasGpuAddress for IndexBuffer { fn gpu_address_u64(&self) -> u64 { self.gpu_address() } }

pub fn bind_argument(table: &ProtocolObject<dyn MTL4ArgumentTable>, index: u32, buf: &impl HasGpuAddress) {
    unsafe { table.setAddress_atIndex(buf.gpu_address_u64(), index as _); }
}
pub fn bind2(table: &ProtocolObject<dyn MTL4ArgumentTable>, a_index: u32, a: &impl HasGpuAddress, b_index: u32, b: &impl HasGpuAddress) {
    unsafe {
        table.setAddress_atIndex(a.gpu_address_u64(), a_index as _);
        table.setAddress_atIndex(b.gpu_address_u64(), b_index as _);
    }
}

