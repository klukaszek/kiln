// Metal barrier utilities.
// Metal does not require explicit layout transitions like Vulkan.
// Barriers in the traditional Metal API are handled via MTLFence or
// texture/buffer hazard tracking which is mostly automatic.
// The RHI barrier calls are no-ops on Metal for now.
