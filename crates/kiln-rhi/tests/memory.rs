//! Headless memory and allocator contract tests.

mod common;

use kiln_rhi::{AllocationDesc, BumpAllocator, MemoryType};

/// `Default` memory is CPU-mapped GPU memory: a write through the mapped pointer must read
/// straight back (the dual-pointer model the whole RHI is built on).
#[test]
fn default_memory_is_cpu_mapped_roundtrip() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };

    const N: usize = 4096;
    let mut allocation = common::timed("allocate 4 KiB (Default)", || {
        device
            .allocate(N as u64, MemoryType::Default)
            .expect("allocate(Default) should succeed")
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
        .allocate(256, MemoryType::Default)
        .expect("allocate(Default) should succeed");
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

fn bump(device: &kiln_rhi::Device, size: u64) -> BumpAllocator {
    let buffer = device
        .create_allocation(&AllocationDesc {
            size,
            memory: MemoryType::Default,
            label: Some("bump".into()),
        })
        .expect("create_buffer(Default)");
    BumpAllocator::new(buffer)
}

/// Allocations honour alignment and accounting advances exactly: each chunk lands at
/// a distinct, aligned, non-overlapping address and `used()` tracks the bumped offset.
#[test]
fn bump_alloc_aligns_and_accounts() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };
    let mut bump = bump(&device, 64 * 1024);

    assert_eq!(bump.capacity(), 64 * 1024, "capacity is the backing size");
    assert_eq!(bump.used(), 0, "fresh allocator has used nothing");

    let mut prev_end = bump.gpu().addr();
    for align in [16u64, 64, 256, 4096] {
        let used_before = bump.used();
        let a = bump.alloc(100, align).expect("fits in a 64 KiB block");
        assert!(
            a.gpu.is_aligned_to(align),
            "gpu address {:#x} not aligned to {align}",
            a.gpu.addr()
        );
        assert!(!a.cpu.is_null(), "Default memory must be CPU-mapped");
        assert!(a.gpu.addr() >= prev_end, "allocations must not overlap");
        // used() == bumped offset == (aligned start - base) + size.
        let start = a.gpu.addr() - bump.gpu().addr();
        assert_eq!(bump.used(), start + 100, "used() tracks the bump offset");
        assert!(bump.used() > used_before);
        prev_end = a.gpu.addr() + 100;
    }

    device.destroy_allocation(bump.into_allocation());
}

/// The dual-pointer invariant: `cpu` and `gpu` from one allocation name the *same*
/// memory. `host_to_device_pointer(cpu)` must return exactly `gpu`, and a CPU write
/// through `cpu` must read straight back.
#[test]
fn bump_alloc_cpu_gpu_correspond() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };
    let mut bump = bump(&device, 4096);

    // Bump past offset 0 so the correspondence is exercised at a non-base address.
    let _pad = bump.alloc(48, 16).expect("pad");
    let a = bump.alloc(256, 16).expect("alloc");

    let translated = device
        .host_to_device_pointer(a.cpu)
        .expect("mapped cpu pointer should translate to a gpu address");
    assert_eq!(translated, a.gpu, "cpu and gpu must name the same memory");
    assert_eq!(
        a.gpu,
        bump.gpu().offset(a.gpu.addr() - bump.gpu().addr()),
        "gpu address is the base plus the bumped offset"
    );

    // CPU write/read roundtrip through the mapped pointer.
    a.upload(&0xDEAD_BEEFu32).expect("upload");
    let read_back = unsafe { std::ptr::read_unaligned(a.cpu as *const u32) };
    assert_eq!(
        read_back, 0xDEAD_BEEF,
        "write through cpu pointer must persist"
    );

    device.destroy_allocation(bump.into_allocation());
}

/// A full allocator returns `None` rather than panicking or overrunning: both an
/// oversized request and exhausting the capacity fail gracefully.
#[test]
fn bump_full_returns_none() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };
    let mut bump = bump(&device, 256);

    assert!(
        bump.alloc(512, 16).is_none(),
        "a request larger than capacity must return None"
    );

    // Drain the allocator; once it can't fit another chunk it keeps returning None.
    let mut count = 0;
    while bump.alloc(64, 16).is_some() {
        count += 1;
        assert!(
            count <= 4,
            "256 bytes can't yield more than four 64B chunks"
        );
    }
    assert_eq!(count, 4, "exactly four 64B chunks fit in 256 bytes");
    assert!(bump.alloc(1, 1).is_none(), "exhausted allocator stays full");

    device.destroy_allocation(bump.into_allocation());
}

/// `reset()` rewinds to the start: post-reset allocations reuse the same addresses,
/// which is the whole point of a per-frame bump allocator.
#[test]
fn bump_reset_reuses_space() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };
    let mut bump = bump(&device, 4096);

    let first = bump.alloc(128, 16).expect("first");
    let (first_cpu, first_gpu) = (first.cpu, first.gpu);
    let _second = bump.alloc(128, 16).expect("second");
    assert_eq!(bump.used(), 256, "two 128B allocs used 256 bytes");

    bump.reset();
    assert_eq!(bump.used(), 0, "reset rewinds the offset");

    let reused = bump.alloc(128, 16).expect("post-reset alloc");
    assert_eq!(reused.cpu, first_cpu, "reset reuses the same cpu pointer");
    assert_eq!(reused.gpu, first_gpu, "reset reuses the same gpu address");

    device.destroy_allocation(bump.into_allocation());
}
