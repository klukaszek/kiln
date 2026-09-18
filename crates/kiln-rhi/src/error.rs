//! Error and result types for the RHI.

use std::error::Error;
use std::fmt;

use thiserror::Error;

/// A message plus the backend error behind it, so a driver failure survives as an object rather
/// than as text: the message is for humans, [`Error::source`] is for code.
///
/// Built from a `String`/`&str`, or with [`with_source`](Self::with_source).
#[derive(Debug)]
pub struct ErrorDetail {
    message: String,
    source: Option<Box<dyn Error + Send + Sync + 'static>>,
}

impl ErrorDetail {
    /// Attach the backend error that caused this failure.
    pub fn with_source<E>(message: impl Into<String>, source: E) -> Self
    where
        E: Error + Send + Sync + 'static,
    {
        Self {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }
}

impl fmt::Display for ErrorDetail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for ErrorDetail {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_deref().map(|source| source as _)
    }
}

impl From<String> for ErrorDetail {
    fn from(message: String) -> Self {
        Self {
            message,
            source: None,
        }
    }
}

impl From<&str> for ErrorDetail {
    fn from(message: &str) -> Self {
        Self::from(message.to_owned())
    }
}

/// A raw `VkResult` converts with itself as the source, so `map_err(|e| RhiError::X(e.into()))`
/// keeps the code the driver actually returned instead of flattening it into text.
#[cfg(feature = "vulkan")]
impl From<ash::vk::Result> for ErrorDetail {
    fn from(result: ash::vk::Result) -> Self {
        Self::with_source(result.to_string(), result)
    }
}

/// Same for the `io::Error`s the shader compiler runs into.
impl From<std::io::Error> for ErrorDetail {
    fn from(error: std::io::Error) -> Self {
        Self::with_source(error.to_string(), error)
    }
}

/// Errors the RHI reports.
///
/// The variant is the category, and is meant to be matched on;
/// [`SwapchainOutOfDate`](Self::SwapchainOutOfDate) in particular is a normal part of a resize.
/// `#[non_exhaustive]` because backends gain failure modes over time.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum RhiError {
    #[error("Device creation failed: {0}")]
    DeviceCreation(#[source] ErrorDetail),

    #[error("Surface creation failed: {0}")]
    SurfaceCreation(#[source] ErrorDetail),

    #[error("Swapchain creation failed: {0}")]
    SwapchainCreation(#[source] ErrorDetail),

    #[error("Swapchain out of date")]
    SwapchainOutOfDate,

    #[error("Memory allocation failed: {0}")]
    AllocationFailed(#[source] ErrorDetail),

    #[error("Buffer creation failed: {0}")]
    BufferCreation(#[source] ErrorDetail),

    #[error("Texture creation failed: {0}")]
    TextureCreation(#[source] ErrorDetail),

    #[error("Shader compilation failed: {0}")]
    ShaderCompilation(#[source] ErrorDetail),

    #[error("Pipeline creation failed: {0}")]
    PipelineCreation(#[source] ErrorDetail),

    #[error("Command buffer error: {0}")]
    CommandBuffer(#[source] ErrorDetail),

    #[error("Queue submit failed: {0}")]
    QueueSubmit(#[source] ErrorDetail),

    #[error("Present failed: {0}")]
    PresentFailed(#[source] ErrorDetail),

    #[error("Synchronization error: {0}")]
    SyncError(#[source] ErrorDetail),

    #[error("No suitable GPU found")]
    NoSuitableGpu,

    #[error("Unsupported: {0}")]
    Unsupported(#[source] ErrorDetail),

    #[error("Backend error: {0}")]
    Backend(#[source] ErrorDetail),
}

pub type RhiResult<T> = Result<T, RhiError>;
