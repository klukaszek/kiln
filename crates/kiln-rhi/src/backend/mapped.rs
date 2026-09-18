//! Reverse index from CPU-mapped addresses back to GPU addresses, for
//! `Device::host_to_device_pointer`: a predecessor search plus an end check, on both backends.

use std::collections::BTreeMap;

use crate::types::GpuPtr;

/// One CPU-mapped allocation's extent, keyed in [`MappedAllocations`] by its base CPU address.
pub(crate) struct MappedAllocation {
    pub(crate) gpu_base: GpuPtr<u8>,
    pub(crate) size: u64,
}

/// CPU base address -> the allocation mapped there.
pub(crate) type MappedAllocations = BTreeMap<usize, MappedAllocation>;

/// Translate a CPU address inside some mapped allocation to the matching GPU address.
///
/// `None` when `ptr` is not inside any of them: the greatest base at or below `ptr` is the only
/// candidate, and it only matches if `ptr` lands before that allocation's end.
pub(crate) fn resolve_mapped_pointer(
    allocations: &MappedAllocations,
    ptr: usize,
) -> Option<GpuPtr<u8>> {
    let (&base, allocation) = allocations.range(..=ptr).next_back()?;
    let offset = (ptr - base) as u64;
    (offset < allocation.size).then(|| allocation.gpu_base.offset(offset))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_uses_the_predecessor_and_respects_the_end() {
        let mut allocations = MappedAllocations::new();
        allocations.insert(
            0x1000,
            MappedAllocation {
                gpu_base: GpuPtr::from_addr(0x8000),
                size: 0x20,
            },
        );
        allocations.insert(
            0x2000,
            MappedAllocation {
                gpu_base: GpuPtr::from_addr(0x9000),
                size: 0x10,
            },
        );

        assert_eq!(
            resolve_mapped_pointer(&allocations, 0x100f),
            Some(GpuPtr::from_addr(0x800f))
        );
        assert_eq!(
            resolve_mapped_pointer(&allocations, 0x200f),
            Some(GpuPtr::from_addr(0x900f))
        );
        // Past the end of the first allocation, before the start of the second.
        assert_eq!(resolve_mapped_pointer(&allocations, 0x1020), None);
        assert_eq!(resolve_mapped_pointer(&allocations, 0x1fff), None);
    }
}
