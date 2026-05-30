//! Headless memory + allocator tests (timed).
//!
//! Black-box successors to the unit tests that used to live in `src/memory.rs`.
//! Every test reports exact runtimes; run with `cargo test -- --nocapture` to see them.

mod common;

use kiln_rhi::{GpuAllocatorDesc, MemoryType};

/// `Default` memory is CPU-mapped GPU memory: a write through the mapped pointer must read
/// straight back (the dual-pointer model the whole RHI is built on).
#[test]
fn default_memory_is_cpu_mapped_roundtrip() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };

    const N: usize = 4096;
    let mut allocation = common::timed("malloc 4 KiB (Default)", || {
        device
            .malloc(N as u64, MemoryType::Default)
            .expect("malloc(Default) should succeed")
    });

    common::timed("CPU write+read 4 KiB roundtrip", || {
        for (i, b) in allocation
            .as_mut_slice::<u8>()
            .expect("mapped slice")
            .iter_mut()
            .enumerate()
        {
            *b = (i as u8).wrapping_mul(7);
        }
        for (i, &b) in allocation
            .as_slice::<u8>()
            .expect("mapped slice")
            .iter()
            .enumerate()
        {
            assert_eq!(b, (i as u8).wrapping_mul(7), "byte {i} mismatch");
        }
    });

    device.free(allocation);
}

/// `host_to_device_pointer` translates a mapped CPU pointer (and an offset into it) back to
/// the matching GPU address.
#[test]
fn host_to_device_pointer_translates_with_offset() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };

    let allocation = device
        .malloc(256, MemoryType::Default)
        .expect("malloc(Default) should succeed");
    let cpu = allocation
        .cpu()
        .expect("Default memory must expose a CPU-mapped pointer");

    let base = common::timed("host_to_device_pointer", || {
        device
            .host_to_device_pointer(cpu)
            .expect("base CPU pointer should translate to a GPU address")
    });
    assert_eq!(base, allocation.gpu());

    // `host_to_device_pointer` is intentionally a raw-pointer bridge, so exercising an
    // offset translation requires raw pointer arithmetic — the one place `unsafe` is
    // inherent to what's under test.
    let offset = device
        .host_to_device_pointer(unsafe { cpu.add(64) })
        .expect("offset CPU pointer should translate to a GPU address");
    assert_eq!(offset, allocation.gpu().offset(64));

    device.free(allocation);
}

/// malloc/free throughput — a feel for raw allocation cost.
#[test]
fn malloc_free_throughput() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };

    common::bench("malloc(64 KiB) + free", 256, || {
        let a = device
            .malloc(64 * 1024, MemoryType::Default)
            .expect("malloc");
        device.free(a);
    });
}

/// A sub-allocation honours the requested alignment in its absolute GPU address.
#[test]
fn suballocation_address_is_aligned() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };

    let mut allocator = device.create_gpu_allocator(GpuAllocatorDesc {
        block_size: 1 << 20,
        memory: MemoryType::Default,
        label: None,
    });

    for align in [16u64, 64, 256, 4096] {
        let sub = allocator
            .alloc(16, align)
            .expect("allocation within a fresh block should fit");
        assert!(
            sub.gpu().is_aligned_to(align),
            "address {:#x} not aligned to {align}",
            sub.gpu()
        );
    }

    common::bench("GpuAllocator.alloc(256, 16)", 4096, || {
        let _ = allocator.alloc(256, 16).expect("suballoc");
    });
}

/// Freed ranges coalesce: after freeing every sub-allocation in a block, a single allocation
/// spanning (almost) the whole block fits *without* pulling in a second backing block.
#[test]
fn freed_ranges_coalesce_for_reuse() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };

    let mut allocator = device.create_gpu_allocator(GpuAllocatorDesc {
        block_size: 256,
        memory: MemoryType::Default,
        label: None,
    });

    common::timed("alloc x3 · free x3 · coalesced realloc", || {
        let a = allocator.alloc(64, 16).expect("a fits");
        let b = allocator.alloc(64, 16).expect("b fits");
        let c = allocator.alloc(64, 16).expect("c fits");
        assert_eq!(
            allocator.block_count(),
            1,
            "three 64B allocs fit in one block"
        );

        allocator.free(a);
        allocator.free(b);
        allocator.free(c);
        assert_eq!(allocator.used(), 0, "freeing everything frees all bytes");

        let big = allocator
            .alloc(192, 16)
            .expect("coalesced free space should fit 192 bytes");
        assert_eq!(
            allocator.block_count(),
            1,
            "coalesced reuse must not allocate a second block"
        );
        allocator.free(big);
    });
}
