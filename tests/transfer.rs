//! Headless GPU transfer tests (timed): record → submit → wait → readback.

mod common;

use spectradio_rhi::{MemoryType, StageFlags};

/// Write a pattern into a CPU-mapped `Default` buffer, GPU-copy it into a `Readback`
/// buffer, and verify the bytes came through. Reports the full submit→wait latency.
#[test]
fn gpu_memcpy_roundtrip() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };

    const SIZE: u64 = 1 << 16; // 64 KiB

    let mut src = device.malloc(SIZE, MemoryType::Default).expect("src");
    let dst = device.malloc(SIZE, MemoryType::Readback).expect("dst");

    for (i, b) in src.as_mut_slice::<u8>().expect("src slice").iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(13).wrapping_add(1);
    }

    common::timed("memcpy 64 KiB · record+submit+wait", || {
        let mut cmd = device.create_command_buffer().expect("cmd");
        cmd.memcpy(dst.gpu_address(), src.gpu_address(), SIZE);
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

    device.free(src);
    device.free(dst);
}

/// Copy-bandwidth sweep across sizes. Each size reports submit→wait time for the copy.
#[test]
fn gpu_memcpy_size_sweep() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };

    for &kib in &[4u64, 64, 1024, 16 * 1024] {
        let size = kib * 1024;
        let src = device.malloc(size, MemoryType::Default).expect("src");
        let dst = device.malloc(size, MemoryType::GpuOnly).expect("dst");

        common::timed(&format!("memcpy {kib} KiB → GpuOnly"), || {
            let mut cmd = device.create_command_buffer().expect("cmd");
            cmd.memcpy(dst.gpu_address(), src.gpu_address(), size);
            cmd.barrier(StageFlags::TRANSFER, StageFlags::ALL_COMMANDS);
            cmd.end();
            let queue = device.queue();
            queue.submit(cmd).expect("submit");
            queue.wait_idle();
        });

        device.free(src);
        device.free(dst);
    }
}
