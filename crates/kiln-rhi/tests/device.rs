//! Headless device contract test.

mod common;

use kiln_rhi::{Device, DeviceDesc};

/// Device creation exposes a usable backend and bindless mode.
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
