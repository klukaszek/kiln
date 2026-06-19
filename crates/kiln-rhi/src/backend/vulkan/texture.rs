use ash::vk;

/// Vulkan texture stored in the bindless heap.
pub struct VulkanTexture {
    pub(crate) image: vk::Image,
    pub(crate) image_view: vk::ImageView,
    /// True when this entry is a view into another texture's image.
    /// On destruction, only `image_view` is freed; `image` belongs to the source.
    pub(crate) is_view: bool,
}
