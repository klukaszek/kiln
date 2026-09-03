//! Headless GPU transfer tests.

mod common;

use kiln_rhi::{MemoryType, StageFlags};

/// Write a pattern into a CPU-mapped `Default` buffer, GPU-copy it into a `Readback`
/// buffer, and verify the bytes came through.
#[test]
fn gpu_memcpy_roundtrip() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };

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
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };

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
