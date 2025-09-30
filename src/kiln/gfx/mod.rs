use zerocopy::{FromBytes, Immutable, IntoBytes};


#[derive(Copy, Clone, IntoBytes, FromBytes, Immutable, Debug)]
#[repr(C)]
pub struct PackedFloat3 { pub x: f32, pub y: f32, pub z: f32 }
impl PackedFloat3 { pub const fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } } }

#[derive(Copy, Clone, IntoBytes, FromBytes, Immutable, Debug)]
#[repr(C)]
pub struct SceneProperties { pub time: f32 }

#[derive(Copy, Clone, IntoBytes, FromBytes, Immutable, Debug)]
#[repr(C)]
pub struct VertexInput { pub position: PackedFloat3, pub color: PackedFloat3 }
