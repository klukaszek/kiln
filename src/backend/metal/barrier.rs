// Metal barrier utilities.
//
// Metal needs no explicit image-layout transitions (unlike Vulkan), so there is no
// stage→layout mapping table here. The actual barrier translation lives in
// `command.rs`: RHI stage/hazard barriers are encoded as MTL4 queue barriers with
// `MTLStages` + `MTL4VisibilityOptions` (see `PendingQueueBarrier` / `barrier*`).
// This file is intentionally empty of logic.
