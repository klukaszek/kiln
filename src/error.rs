use thiserror::Error;

#[derive(Error, Debug)]
pub enum RhiError {
    #[error("Device creation failed: {0}")]
    DeviceCreation(String),

    #[error("Surface creation failed: {0}")]
    SurfaceCreation(String),

    #[error("Swapchain creation failed: {0}")]
    SwapchainCreation(String),

    #[error("Swapchain out of date")]
    SwapchainOutOfDate,

    #[error("Memory allocation failed: {0}")]
    AllocationFailed(String),

    #[error("Buffer creation failed: {0}")]
    BufferCreation(String),

    #[error("Texture creation failed: {0}")]
    TextureCreation(String),

    #[error("Shader compilation failed: {0}")]
    ShaderCompilation(String),

    #[error("Pipeline creation failed: {0}")]
    PipelineCreation(String),

    #[error("Command buffer error: {0}")]
    CommandBuffer(String),

    #[error("Queue submit failed: {0}")]
    QueueSubmit(String),

    #[error("Present failed: {0}")]
    PresentFailed(String),

    #[error("Synchronization error: {0}")]
    SyncError(String),

    #[error("No suitable GPU found")]
    NoSuitableGpu,

    #[error("Feature not supported: {0}")]
    FeatureNotSupported(String),

    #[error("Unsupported: {0}")]
    Unsupported(String),

    #[error("Backend error: {0}")]
    Backend(String),
}

pub type RhiResult<T> = Result<T, RhiError>;
