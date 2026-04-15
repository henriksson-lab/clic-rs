/// Data type of GPU array elements — mirrors CLIc's `dType` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DType {
    Int8,
    Uint8,
    Int16,
    Uint16,
    Int32,
    Uint32,
    Float,
    Complex,
    Unknown,
}

// Semantic aliases matching CLIc
pub const INT: DType = DType::Int32;
pub const INDEX: DType = DType::Uint32;
pub const LABEL: DType = DType::Uint32;
pub const BINARY: DType = DType::Uint8;

impl DType {
    /// OpenCL type string (e.g. "float", "int", "uchar")
    pub fn to_ocl_str(self) -> &'static str {
        match self {
            DType::Float => "float",
            DType::Int32 => "int",
            DType::Uint32 => "uint",
            DType::Int8 => "char",
            DType::Uint8 => "uchar",
            DType::Int16 => "short",
            DType::Uint16 => "ushort",
            DType::Complex => "float",
            DType::Unknown => "unknown",
        }
    }

    /// Short type suffix used in kernel macro names (e.g. "f", "i", "uc")
    pub fn to_short_str(self) -> &'static str {
        match self {
            DType::Float => "f",
            DType::Int32 => "i",
            DType::Uint32 => "ui",
            DType::Int8 => "c",
            DType::Uint8 => "uc",
            DType::Int16 => "s",
            DType::Uint16 => "us",
            DType::Complex => "f",
            DType::Unknown => "?",
        }
    }

    /// Size in bytes
    pub fn byte_size(self) -> usize {
        match self {
            DType::Float => 4,
            DType::Int32 => 4,
            DType::Uint32 => 4,
            DType::Int8 => 1,
            DType::Uint8 => 1,
            DType::Int16 => 2,
            DType::Uint16 => 2,
            DType::Complex => 4,
            DType::Unknown => 0,
        }
    }

    /// The USE_* `#define` that CLIc prepends for this dtype
    pub fn dimension_define(self) -> &'static str {
        match self {
            DType::Int8 => "#define USE_CHAR",
            DType::Uint8 => "#define USE_UCHAR",
            DType::Int16 => "#define USE_SHORT",
            DType::Uint16 => "#define USE_USHORT",
            DType::Int32 => "#define USE_INT",
            DType::Uint32 => "#define USE_UINT",
            DType::Float => "#define USE_FLOAT",
            DType::Complex => "#define USE_FLOAT",
            DType::Unknown => "",
        }
    }
}

/// Memory type — mirrors CLIc's `mType` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MType {
    Buffer,
    Image,
}

/// Marker trait for types that map to a DType at compile time.
pub trait GpuScalar: Copy + Send + Sync + 'static {
    fn dtype() -> DType;
}

impl GpuScalar for f32 {
    fn dtype() -> DType { DType::Float }
}
impl GpuScalar for i8 {
    fn dtype() -> DType { DType::Int8 }
}
impl GpuScalar for u8 {
    fn dtype() -> DType { DType::Uint8 }
}
impl GpuScalar for i16 {
    fn dtype() -> DType { DType::Int16 }
}
impl GpuScalar for u16 {
    fn dtype() -> DType { DType::Uint16 }
}
impl GpuScalar for i32 {
    fn dtype() -> DType { DType::Int32 }
}
impl GpuScalar for u32 {
    fn dtype() -> DType { DType::Uint32 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_ocl_str() {
        assert_eq!(DType::Float.to_ocl_str(), "float");
        assert_eq!(DType::Int32.to_ocl_str(), "int");
        assert_eq!(DType::Uint8.to_ocl_str(), "uchar");
    }

    #[test]
    fn dtype_byte_size() {
        assert_eq!(DType::Float.byte_size(), 4);
        assert_eq!(DType::Uint8.byte_size(), 1);
        assert_eq!(DType::Int16.byte_size(), 2);
    }

    #[test]
    fn dtype_short_str() {
        assert_eq!(DType::Float.to_short_str(), "f");
        assert_eq!(DType::Uint32.to_short_str(), "ui");
        assert_eq!(DType::Int8.to_short_str(), "c");
    }
}
