use opencl3::error_codes::ClError;

#[derive(Debug, thiserror::Error)]
pub enum CleError {
    #[error("OpenCL error: {0}")]
    OpenCL(String),
    #[error("Null array: {0}")]
    NullArray(&'static str),
    #[error("Invalid dtype")]
    InvalidDtype,
    #[error("Dimension mismatch")]
    DimensionMismatch,
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Backend not initialized")]
    BackendNotInitialized,
    #[error("No OpenCL devices found")]
    NoDevicesFound,
    #[error("Array not allocated")]
    NotAllocated,
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, CleError>;

impl From<ClError> for CleError {
    fn from(e: ClError) -> Self {
        CleError::OpenCL(format!("{:?}", e))
    }
}
