//! Shader-visible heap of `gpuResourceID`s. Slots come from [`SlotTable`].

use std::ops::Deref;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;

use crate::backend::slots::SlotTable;

/// One `gpuResourceID` per slot.
const HEAP_ENTRY_BYTES: usize = size_of::<u64>();

/// A [`SlotTable`] plus the id heap that mirrors it.
pub(crate) struct BindlessTable<T> {
    slots: SlotTable<T>,
    /// `gpuResourceID`s indexed by id, bound once per command buffer at a fixed argument-table
    /// slot and never rebound.
    heap: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl<T> Deref for BindlessTable<T> {
    type Target = SlotTable<T>;
    fn deref(&self) -> &SlotTable<T> {
        &self.slots
    }
}

impl<T> BindlessTable<T> {
    pub(crate) fn new(
        heap: Retained<ProtocolObject<dyn MTLBuffer>>,
        capacity: u32,
        what: &'static str,
    ) -> Self {
        Self {
            slots: SlotTable::new(capacity, what),
            heap,
        }
    }

    pub(crate) fn heap(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.heap
    }

    /// Store `value` at `id` and publish `resource_id` to the shader-visible heap.
    pub(crate) fn insert(&self, id: u32, value: T, resource_id: u64) {
        self.slots.insert(id, value);
        self.write_heap_slot(id, resource_id);
    }

    /// A destroyed resource leaves its slot stale rather than zeroed: an in-flight frame may
    /// still read it, and the id is not recycled until that frame retires.
    fn write_heap_slot(&self, id: u32, resource_id: u64) {
        assert!(
            id < self.slots.capacity(),
            "bindless id {id} is past the heap's {} slots",
            self.slots.capacity()
        );
        // SAFETY: the heap was allocated with `capacity` entries of `HEAP_ENTRY_BYTES`, `id` is
        // inside it by the assert above, and it is `StorageModeShared` so the pointer is valid
        // for CPU writes.
        unsafe {
            let base = self.heap.contents().as_ptr().cast::<u8>();
            base.add(id as usize * HEAP_ENTRY_BYTES)
                .cast::<u64>()
                .write(resource_id);
        }
    }
}
