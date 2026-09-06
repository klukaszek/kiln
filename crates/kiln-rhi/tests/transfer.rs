//! Headless GPU transfer tests.

mod common;

use kiln_rhi::{MemoryType, StageFlags};

/// Write a pattern into a CPU-mapped `Default` buffer, GPU-copy it into a `Readback`
/// buffer, and verify the bytes came through.
#[test]
fn gpu_memcpy_roundtrip() {
    let (device, _gpu) = common::device();

    const SIZE: u64 = 1 << 16; // 64 KiB

    let mut src = device.allocate(SIZE, MemoryType::Upload).expect("src");
    let dst = device.allocate(SIZE, MemoryType::Readback).expect("dst");

    for (i, b) in src
        .as_mut_slice::<u8>()
        .expect("src slice")
        .iter_mut()
        .enumerate()
    {
        *b = (i as u8).wrapping_mul(13).wrapping_add(1);
    }

    common::timed("memcpy 64 KiB · record+submit+wait", || {
        let mut cmd = device.create_command_buffer().expect("cmd");
        cmd.memcpy(dst.gpu(), src.gpu(), SIZE);
        cmd.barrier(StageFlags::TRANSFER, StageFlags::ALL_COMMANDS);
        cmd.end();
        let queue = device.queue();
        queue.submit(cmd).expect("submit");
        queue.wait_idle();
    });

    for (i, &b) in dst.as_slice::<u8>().expect("dst slice").iter().enumerate() {
        let expected = (i as u8).wrapping_mul(13).wrapping_add(1);
        assert_eq!(b, expected, "byte {i} mismatch");
    }

    device.destroy(src);
    device.destroy(dst);
}

/// Copy round-trips across several allocation sizes.
#[test]
fn gpu_memcpy_size_sweep() {
    let (device, _gpu) = common::device();

    for &kib in &[4u64, 64, 1024, 16 * 1024] {
        let size = kib * 1024;
        let src = device.allocate(size, MemoryType::Upload).expect("src");
        let dst = device.allocate(size, MemoryType::GpuOnly).expect("dst");

        common::timed(&format!("memcpy {kib} KiB → GpuOnly"), || {
            let mut cmd = device.create_command_buffer().expect("cmd");
            cmd.memcpy(dst.gpu(), src.gpu(), size);
            cmd.barrier(StageFlags::TRANSFER, StageFlags::ALL_COMMANDS);
            cmd.end();
            let queue = device.queue();
            queue.submit(cmd).expect("submit");
            queue.wait_idle();
        });

        device.destroy(src);
        device.destroy(dst);
    }
}

/// Destroying a buffer that an in-flight submission still reads must not release its storage.
///
/// The allocation right after the destroy is the trap: with immediate release, the suballocator
/// hands back the same block, and writing a fresh pattern into it corrupts the copy still running
/// on the GPU. `Device::destroy` therefore holds the storage until the submission retires.
#[test]
fn destroy_while_in_flight_keeps_storage_alive() {
    let (device, _gpu) = common::device();

    const SIZE: u64 = 1 << 20; // 1 MiB, big enough that the copy is still running at destroy time
    const LIVE: u8 = 0xC3;
    const SQUATTER: u8 = 0x5A;

    let mut src = device
        .allocate(SIZE, MemoryType::Upload)
        .expect("source allocation");
    let dst = device
        .allocate(SIZE, MemoryType::Readback)
        .expect("readback allocation");
    src.as_mut_slice::<u8>().expect("src slice").fill(LIVE);

    let mut cmd = device.create_command_buffer().expect("cmd");
    cmd.memcpy(dst.gpu(), src.gpu(), SIZE);
    cmd.barrier(StageFlags::TRANSFER, StageFlags::ALL_COMMANDS);
    cmd.end();
    device.queue().submit(cmd).expect("submit");

    // No fence of any kind between the submit and the destroy.
    device.destroy(src);

    // Try to reclaim the freed range and scribble over it.
    let mut squatter = device
        .allocate(SIZE, MemoryType::Upload)
        .expect("squatter allocation");
    squatter
        .as_mut_slice::<u8>()
        .expect("squatter slice")
        .fill(SQUATTER);

    device.queue().wait_idle();

    for (i, &b) in dst.as_slice::<u8>().expect("dst slice").iter().enumerate() {
        assert_eq!(b, LIVE, "byte {i} was overwritten by a reused allocation");
    }

    device.destroy(squatter);
    device.destroy(dst);
}
