//! Which shader-visible slots exist and what lives in them. The heap write itself is the
//! backend's: Vulkan copies a driver-sized descriptor, Metal writes an eight-byte resource id.

use std::cell::{Cell, RefCell};

use crate::error::{RhiError, RhiResult};

pub(crate) struct SlotTable<T> {
    /// Indexed by id. `None` once the resource is destroyed and before the id is handed out again.
    slots: RefCell<Vec<Option<T>>>,
    /// Retired ids, reused before the table grows.
    free: RefCell<Vec<u32>>,
    /// Ids handed out so far. Tracked apart from `slots.len()` so an id that has been reserved
    /// but not yet filled is never handed out a second time.
    next: Cell<u32>,
    capacity: u32,
    /// Names this table in the error raised when it fills up.
    what: &'static str,
}

impl<T> SlotTable<T> {
    pub(crate) fn new(capacity: u32, what: &'static str) -> Self {
        Self {
            slots: RefCell::new(Vec::new()),
            free: RefCell::new(Vec::new()),
            next: Cell::new(0),
            capacity,
            what,
        }
    }

    /// Metal bounds-checks its heap write against this.
    #[cfg(feature = "metal")]
    pub(crate) fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Reuse a retired id, or take the next one. Errors when the table is full.
    pub(crate) fn allocate_id(&self) -> RhiResult<u32> {
        if let Some(id) = self.free.borrow_mut().pop() {
            return Ok(id);
        }
        let next = self.next.get();
        if next >= self.capacity {
            return Err(RhiError::Backend(
                format!("bindless {} heap exhausted", self.what).into(),
            ));
        }
        self.next.set(next + 1);
        Ok(next)
    }

    /// Hand `id` back for reuse, once nothing in flight refers to it.
    pub(crate) fn recycle(&self, id: u32) {
        self.free.borrow_mut().push(id);
    }

    pub(crate) fn insert(&self, id: u32, value: T) {
        let index = id as usize;
        let mut slots = self.slots.borrow_mut();
        if slots.len() <= index {
            slots.resize_with(index + 1, || None);
        }
        slots[index] = Some(value);
    }

    /// Take the resource out of `id`, leaving the slot empty. The id is not recycled here: it
    /// stays reserved until the GPU work referencing it has retired.
    pub(crate) fn take(&self, id: u32) -> Option<T> {
        self.slots.borrow_mut().get_mut(id as usize)?.take()
    }

    /// Take `id` only if the stored value satisfies `predicate`. Used to claim a texture view
    /// without disturbing a base texture that happens to share the slot space.
    pub(crate) fn take_if(&self, id: u32, predicate: impl FnOnce(&T) -> bool) -> Option<T> {
        let mut slots = self.slots.borrow_mut();
        let slot = slots.get_mut(id as usize)?;
        if slot.as_ref().is_some_and(predicate) {
            slot.take()
        } else {
            None
        }
    }

    /// Read something out of the slot at `id` without cloning the whole entry.
    pub(crate) fn with<R>(&self, id: u32, read: impl FnOnce(&T) -> R) -> Option<R> {
        Some(read(self.slots.borrow().get(id as usize)?.as_ref()?))
    }

    /// Empty the table, for the device's teardown. Metal's entries are released by ARC.
    #[cfg(feature = "vulkan")]
    pub(crate) fn drain(&self) -> Vec<T> {
        self.slots.borrow_mut().drain(..).flatten().collect()
    }
}
