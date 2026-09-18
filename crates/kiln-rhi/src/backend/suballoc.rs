//! Backend-agnostic suballocation over fixed blocks of device memory.
//!
//! A block is an `MTLHeap` (Placement) on Metal and a `VkDeviceMemory` on Vulkan; the bookkeeping
//! is the same integer arithmetic either way. Vulkan has a hard reason to suballocate at all:
//! `maxMemoryAllocationCount` caps live allocations, commonly at 4096.

use std::collections::BTreeMap;

/// Size of one backing block, shared by both backends so they fragment identically. A block is an
/// `MTLHeap` on Metal and a `VkDeviceMemory` plus its spanning buffer on Vulkan.
pub(crate) const BLOCK_SIZE: u64 = 4 * 1024 * 1024;

/// Address-ordered free ranges within one block. Allocation is first-fit; freeing merges adjacent
/// ranges so long-lived resource churn does not permanently fragment a block.
#[derive(Default, Debug)]
pub(crate) struct FreeRanges {
    /// Disjoint, non-adjacent free ranges keyed by start offset.
    ranges: BTreeMap<u64, u64>,
    /// Total block size, so a block can recognise when it has become entirely free.
    size: u64,
}

impl FreeRanges {
    pub(crate) fn full(size: u64) -> Self {
        Self {
            ranges: if size == 0 {
                BTreeMap::new()
            } else {
                BTreeMap::from([(0, size)])
            },
            size,
        }
    }

    /// Carve `size` bytes at `align`, returning the chosen offset.
    pub(crate) fn allocate(&mut self, size: u64, align: u64) -> Option<u64> {
        if size == 0 {
            return None;
        }
        let (free_offset, free_size, aligned) =
            self.ranges.iter().find_map(|(&offset, &free)| {
                let aligned = align_up(offset, align)?;
                let end = aligned.checked_add(size)?;
                (end <= offset.checked_add(free)?).then_some((offset, free, aligned))
            })?;

        self.ranges.remove(&free_offset);
        if aligned > free_offset {
            self.ranges.insert(free_offset, aligned - free_offset);
        }
        let end = aligned + size;
        let free_end = free_offset + free_size;
        if end < free_end {
            self.ranges.insert(end, free_end - end);
        }
        Some(aligned)
    }

    /// Return `[offset, offset + size)` to the block, coalescing with either neighbour.
    pub(crate) fn release(&mut self, mut offset: u64, mut size: u64) {
        if size == 0 {
            return;
        }
        debug_assert!(
            offset.saturating_add(size) <= self.size,
            "released range escapes the block"
        );
        if let Some((&previous_offset, &previous_size)) = self.ranges.range(..offset).next_back()
            && previous_offset + previous_size == offset
        {
            self.ranges.remove(&previous_offset);
            offset = previous_offset;
            size += previous_size;
        }
        if let Some((&next_offset, &next_size)) = self.ranges.range(offset..).next()
            && offset + size == next_offset
        {
            self.ranges.remove(&next_offset);
            size += next_size;
        }
        self.ranges.insert(offset, size);
    }

    /// True when nothing in this block is allocated. Note this is the opposite of
    /// the inner `BTreeMap` being empty, which would mean the block is entirely handed out.
    pub(crate) fn is_fully_free(&self) -> bool {
        match self.ranges.iter().next() {
            Some((&offset, &size)) => self.ranges.len() == 1 && offset == 0 && size == self.size,
            // A zero-sized block holds no ranges and can never have handed anything out.
            None => self.size == 0,
        }
    }
}

/// A pool of fixed-size blocks, each subdivided by a [`FreeRanges`].
///
/// `K` is what makes two blocks interchangeable (memory type, plus buffer usage on Vulkan) and
/// `P` is the native block. Creating and destroying one is the caller's closure, since Vulkan
/// passes its `&ash::Device` in per call while Metal's pool owns its device.
pub(crate) struct BlockPool<K, P> {
    /// Emptied, never removed, so a suballocation's block index stays valid for its whole life.
    blocks: Vec<Option<PoolBlock<K, P>>>,
}

pub(crate) struct PoolBlock<K, P> {
    pub(crate) key: K,
    pub(crate) payload: P,
    free_ranges: FreeRanges,
}

impl<K, P> Default for BlockPool<K, P> {
    fn default() -> Self {
        Self { blocks: Vec::new() }
    }
}

impl<K: Copy + PartialEq, P> BlockPool<K, P> {
    /// Carve `size` bytes at `align` from a block matching `key`, creating one if none has room.
    ///
    /// `create` is handed the size the new block must be at least — the request itself, which may
    /// exceed [`BLOCK_SIZE`] — and returns the native block. Returns the block's index and the
    /// offset within it.
    pub(crate) fn allocate<E>(
        &mut self,
        key: K,
        size: u64,
        align: u64,
        create: impl FnOnce(u64) -> Result<P, E>,
    ) -> Result<(usize, u64), E> {
        if let Some(found) = self.allocate_from_existing(key, size, align) {
            return Ok(found);
        }
        let payload = create(size.max(BLOCK_SIZE))?;
        let index = self.insert(PoolBlock {
            key,
            payload,
            free_ranges: FreeRanges::full(size.max(BLOCK_SIZE)),
        });
        let offset = self.blocks[index]
            .as_mut()
            .expect("the block was just inserted")
            .free_ranges
            .allocate(size, align)
            .expect("a fresh block is at least as large as the request that created it");
        Ok((index, offset))
    }

    fn allocate_from_existing(&mut self, key: K, size: u64, align: u64) -> Option<(usize, u64)> {
        self.blocks
            .iter_mut()
            .enumerate()
            .find_map(|(index, slot)| {
                let block = slot.as_mut()?;
                (block.key == key)
                    .then(|| block.free_ranges.allocate(size, align))
                    .flatten()
                    .map(|offset| (index, offset))
            })
    }

    fn insert(&mut self, block: PoolBlock<K, P>) -> usize {
        match self.blocks.iter().position(Option::is_none) {
            Some(index) => {
                self.blocks[index] = Some(block);
                index
            }
            None => {
                self.blocks.push(Some(block));
                self.blocks.len() - 1
            }
        }
    }

    pub(crate) fn block(&self, index: usize) -> &PoolBlock<K, P> {
        self.blocks[index]
            .as_ref()
            .expect("a block outlives its suballocations")
    }

    /// Return a range to its block. Never frees the block — see [`trim`](Self::trim).
    pub(crate) fn release(&mut self, index: usize, offset: u64, size: u64) {
        self.blocks[index]
            .as_mut()
            .expect("a block outlives its suballocations")
            .free_ranges
            .release(offset, size);
    }

    /// Destroy empty blocks, keeping one per key. Call only where already synchronising.
    pub(crate) fn trim(&mut self, mut destroy: impl FnMut(P)) {
        let mut kept: Vec<K> = Vec::new();
        for slot in &mut self.blocks {
            let Some(block) = slot.as_ref() else { continue };
            if !block.free_ranges.is_fully_free() {
                continue;
            }
            if kept.contains(&block.key) {
                let block = slot.take().expect("checked just above");
                destroy(block.payload);
            } else {
                kept.push(block.key);
            }
        }
    }

    /// Destroy every block. Call only once all GPU work has retired.
    ///
    /// Vulkan-only in practice: a `VkDeviceMemory` has to be freed explicitly at device teardown,
    /// while an `MTLHeap` is released by ARC when the pool drops.
    #[cfg(feature = "vulkan")]
    pub(crate) fn destroy_all(&mut self, mut destroy: impl FnMut(P)) {
        for block in self.blocks.drain(..).flatten() {
            destroy(block.payload);
        }
    }
}

/// Round `offset` up to a multiple of `align`, `None` on overflow.
pub(crate) fn align_up(offset: u64, align: u64) -> Option<u64> {
    let align = align.max(1);
    let remainder = offset % align;
    if remainder == 0 {
        return Some(offset);
    }
    offset.checked_add(align - remainder)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn splits_aligns_and_coalesces() {
        let mut ranges = FreeRanges::full(128);
        assert_eq!(ranges.allocate(17, 16), Some(0));
        assert_eq!(ranges.allocate(9, 32), Some(32));
        // No room once the request is aligned up.
        assert_eq!(ranges.allocate(80, 64), None);
        assert!(!ranges.is_fully_free());
        ranges.release(0, 17);
        ranges.release(32, 9);
        assert!(ranges.is_fully_free(), "freeing everything must coalesce");
    }

    #[test]
    fn coalesces_a_hole_between_two_live_ranges() {
        let mut ranges = FreeRanges::full(96);
        let a = ranges.allocate(32, 1).expect("a");
        let b = ranges.allocate(32, 1).expect("b");
        let c = ranges.allocate(32, 1).expect("c");
        // Outer two first, so the middle release has a neighbour on each side.
        ranges.release(a, 32);
        ranges.release(c, 32);
        assert!(!ranges.is_fully_free());
        ranges.release(b, 32);
        assert!(ranges.is_fully_free());
        // The whole block is available again as a single run.
        assert_eq!(ranges.allocate(96, 1), Some(0));
    }

    #[test]
    fn a_pool_reuses_a_block_with_room_and_creates_one_otherwise() {
        let mut pool: BlockPool<u8, &'static str> = BlockPool::default();
        let created = Cell::new(0);
        let create = |_size: u64| -> Result<&'static str, ()> {
            created.set(created.get() + 1);
            Ok("block")
        };
        let (first, _) = pool.allocate(1, 64, 16, create).expect("first");
        let (second, _) = pool.allocate(1, 64, 16, create).expect("second");
        assert_eq!((first, second), (0, 0), "same key reuses the same block");
        assert_eq!(created.get(), 1);

        let (other, _) = pool.allocate(2, 64, 16, create).expect("other key");
        assert_eq!(other, 1, "a different key needs its own block");
        assert_eq!(created.get(), 2);
    }

    #[test]
    fn a_request_larger_than_a_block_gets_a_block_of_its_own() {
        let mut pool: BlockPool<u8, u64> = BlockPool::default();
        let huge = BLOCK_SIZE * 3;
        let (index, offset) = pool
            .allocate(0, huge, 1, |size| -> Result<u64, ()> {
                assert_eq!(size, huge, "the block must cover the request");
                Ok(size)
            })
            .expect("oversized allocation");
        assert_eq!((index, offset), (0, 0));
    }

    #[test]
    fn trim_keeps_one_empty_block_per_key() {
        let mut pool: BlockPool<u8, u32> = BlockPool::default();
        let next = Cell::new(0);
        let create = |_: u64| -> Result<u32, ()> {
            next.set(next.get() + 1);
            Ok(next.get())
        };
        // Two blocks under one key: the second only exists because the first was full.
        let (a, a_off) = pool.allocate(7, BLOCK_SIZE, 1, create).expect("a");
        let (b, b_off) = pool.allocate(7, BLOCK_SIZE, 1, create).expect("b");
        assert_ne!(a, b);

        pool.release(a, a_off, BLOCK_SIZE);
        pool.release(b, b_off, BLOCK_SIZE);

        let mut destroyed = Vec::new();
        pool.trim(|payload| destroyed.push(payload));
        assert_eq!(destroyed.len(), 1, "one of the two empty blocks is kept");
    }

    #[test]
    fn align_up_saturates_instead_of_wrapping() {
        assert_eq!(align_up(17, 16), Some(32));
        assert_eq!(align_up(32, 16), Some(32));
        assert_eq!(align_up(0, 0), Some(0));
        assert_eq!(align_up(u64::MAX, 16), None);
    }
}
