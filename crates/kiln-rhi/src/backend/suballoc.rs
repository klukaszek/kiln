//! Backend-agnostic suballocation over fixed blocks of device memory.
//!
//! A block is an `MTLHeap` (Placement) on Metal and a `VkDeviceMemory` on Vulkan; the bookkeeping
//! is the same integer arithmetic either way. Vulkan has a hard reason to suballocate at all:
//! `maxMemoryAllocationCount` caps live allocations, commonly at 4096.

use std::collections::BTreeMap;

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

    /// True when nothing in this block is allocated.
    pub(crate) fn is_empty(&self) -> bool {
        match self.ranges.iter().next() {
            Some((&offset, &size)) => self.ranges.len() == 1 && offset == 0 && size == self.size,
            // A zero-sized block holds no ranges and can never have handed anything out.
            None => self.size == 0,
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

    #[test]
    fn splits_aligns_and_coalesces() {
        let mut ranges = FreeRanges::full(128);
        assert_eq!(ranges.allocate(17, 16), Some(0));
        assert_eq!(ranges.allocate(9, 32), Some(32));
        assert!(!ranges.is_empty());
        ranges.release(0, 17);
        ranges.release(32, 9);
        assert!(ranges.is_empty(), "freeing everything must coalesce");
    }

    #[test]
    fn rejects_an_allocation_that_does_not_fit_after_alignment() {
        let mut ranges = FreeRanges::full(32);
        assert_eq!(ranges.allocate(8, 1), Some(0));
        assert_eq!(ranges.allocate(17, 32), None);
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
        assert!(!ranges.is_empty());
        ranges.release(b, 32);
        assert!(ranges.is_empty());
        // The whole block is available again as a single run.
        assert_eq!(ranges.allocate(96, 1), Some(0));
    }

    #[test]
    fn a_fully_freed_block_is_reported_empty_only_when_truly_free() {
        let mut ranges = FreeRanges::full(64);
        assert!(ranges.is_empty());
        let offset = ranges.allocate(1, 1).expect("one byte");
        assert!(!ranges.is_empty());
        ranges.release(offset, 1);
        assert!(ranges.is_empty());
    }

    #[test]
    fn align_up_saturates_instead_of_wrapping() {
        assert_eq!(align_up(17, 16), Some(32));
        assert_eq!(align_up(32, 16), Some(32));
        assert_eq!(align_up(0, 0), Some(0));
        assert_eq!(align_up(u64::MAX, 16), None);
    }
}
