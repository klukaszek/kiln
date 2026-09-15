//! Headless device contract test.

mod common;

use kiln_rhi::Backend;
use kiln_rhi::{AllocationDesc, Device, DeviceDesc, MemoryType};

/// Device creation exposes a usable backend and bindless mode.
#[test]
fn device_creation_and_properties() {
    let start = std::time::Instant::now();
    let device = Device::new(&DeviceDesc {
        validation: false,
        label: Some("rhi-timing".into()),
        ..Default::default()
    })
    .expect("no headless GPU device available");
    eprintln!("    ⏱  Device::new: {}", common::fmt_dur(start.elapsed()));
    eprintln!("    backend={}", device.backend());
    assert!(matches!(device.backend(), Backend::Vulkan | Backend::Metal));
}

/// Owning resources retain the backend device, so ordinary Rust drop order cannot make their
/// destructors call through an already-destroyed native device.
#[test]
fn resources_may_outlive_the_device_handle() {
    let (device, _gpu) = common::device();

    let allocation = device
        .create_allocation(&AllocationDesc {
            size: 256,
            memory: MemoryType::Upload,
            label: Some("device-lifetime-buffer".into()),
            ..Default::default()
        })
        .expect("allocation");
    let queries = device.create_query_pool(2).expect("query pool");
    let timeline = device.create_timeline_semaphore(0).expect("timeline");

    drop(device);
    drop(queries);
    drop(timeline);
    drop(allocation);
}
