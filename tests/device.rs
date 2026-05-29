//! Headless device + submission-path tests (timed).

mod common;

use spectradio_rhi::{Device, DeviceDesc};

/// Time device creation and report the backend's reported properties.
#[test]
fn device_creation_and_properties() {
    let start = std::time::Instant::now();
    let device = match Device::new(&DeviceDesc {
        validation: false,
        label: Some("rhi-timing".into()),
        ..Default::default()
    }) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no headless GPU device available ({e})");
            return;
        }
    };
    eprintln!("    ⏱  Device::new: {}", common::fmt_dur(start.elapsed()));
    eprintln!(
        "    backend={}  bindless={:?}  clip_space_y={:?}",
        device.backend_name(),
        device.bindless_mode(),
        device.clip_space_y()
    );
    assert!(!device.backend_name().is_empty());
}

/// Round-trip latency of submitting an empty command buffer and waiting for the GPU.
/// This is the floor cost of the record→submit→wait path.
#[test]
fn empty_submit_latency() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };

    common::bench("empty cmd · record+submit+wait_idle", 64, || {
        let mut cmd = device.create_command_buffer().expect("cmd");
        cmd.end();
        let queue = device.queue();
        queue.submit(cmd).expect("submit");
        queue.wait_idle();
    });
}

/// Command-buffer creation cost on its own (no submit).
#[test]
fn command_buffer_creation_throughput() {
    let Some((device, _gpu)) = common::device_or_skip() else {
        return;
    };

    common::bench("create_command_buffer", 64, || {
        let mut cmd = device.create_command_buffer().expect("cmd");
        cmd.end();
        let queue = device.queue();
        queue.submit(cmd).expect("submit");
    });
    device.queue().wait_idle();
}
