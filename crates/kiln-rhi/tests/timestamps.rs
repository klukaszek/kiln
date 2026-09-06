//! Exercise timestamp readback and reuse around real GPU work.
mod common;

use kiln_rhi::{MemoryType, StageFlags};
use std::time::Instant;

#[test]
fn timestamps_bracket_transfers() {
    let (device, _gpu) = common::device();
    const SIZE: u64 = 16 * 1024 * 1024;
    let src = device.allocate(SIZE, MemoryType::Upload).expect("src");
    let dst = device.allocate(SIZE, MemoryType::GpuOnly).expect("dst");
    let pool = device.create_query_pool(2).expect("query pool");
    let mut valid_samples = 0;
    let mut previous_end = 0;
    for _ in 0..32 {
        let mut cmd = device.create_command_buffer().expect("cmd");
        cmd.reset_queries(&pool);
        cmd.write_timestamp(&pool, 0);
        for _ in 0..64 {
            cmd.memcpy(dst.gpu(), src.gpu(), SIZE);
            cmd.barrier(StageFlags::TRANSFER, StageFlags::ALL_COMMANDS);
        }
        cmd.write_timestamp(&pool, 1);
        cmd.end();
        let start = Instant::now();
        device.queue().submit(cmd).expect("submit");
        device.queue().wait_idle();
        let wall_ms = start.elapsed().as_secs_f64() * 1e3;
        let ticks = device.read_timestamps(&pool).expect("timestamps");
        let gpu_ms = device.gpu_elapsed_ms(&pool, 0, 1).expect("elapsed");
        eprintln!(
            "ticks={ticks:?}, period={} ns, GPU={gpu_ms:?} ms, wall={wall_ms:.3} ms",
            device.timestamp_period_ns()
        );
        // Metal can omit a sample. Missing data must remain unavailable rather than
        // pairing a stale timestamp from the previous submission with a fresh one.
        assert!(
            ticks[0] == 0 || ticks[0] > previous_end,
            "stale start timestamp"
        );
        if let Some(gpu_ms) = gpu_ms {
            valid_samples += 1;
            assert!(gpu_ms.is_finite() && gpu_ms > 0.0);
            assert!(
                gpu_ms <= wall_ms + 1.0,
                "GPU duration exceeds submit-to-completion time"
            );
        } else {
            assert!(ticks.contains(&0), "nonzero timestamps must be ordered");
        }
        previous_end = previous_end.max(ticks[0]).max(ticks[1]);
    }
    assert!(valid_samples > 0, "no usable GPU timing samples");
    device.destroy(pool);
    device.destroy(src);
    device.destroy(dst);
}
