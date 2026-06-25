use std::any::TypeId;

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

/// Memory type — mirrors CLIc's `mType` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MType {
    Buffer,
    Image,
}

pub fn to_string(dtype: DType) -> &'static str {
    match dtype {
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

pub fn to_short_string(dtype: DType) -> &'static str {
    match dtype {
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

pub fn to_bytes(dtype: DType) -> usize {
    match dtype {
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

pub fn to_mtype_string(mtype: MType) -> &'static str {
    match mtype {
        MType::Buffer => "Buffer",
        MType::Image => "Image",
    }
}

/// Marker trait for types that map to a DType at compile time.
pub trait GpuScalar: Copy + Send + Sync + 'static {}

impl GpuScalar for f32 {}
impl GpuScalar for i8 {}
impl GpuScalar for u8 {}
impl GpuScalar for i16 {}
impl GpuScalar for u16 {}
impl GpuScalar for i32 {}
impl GpuScalar for u32 {}

/// Canonical equivalent of CLIc's `toType<T>()`.
pub fn to_type<T: GpuScalar>() -> DType {
    let type_id = TypeId::of::<T>();
    if type_id == TypeId::of::<f32>() {
        DType::Float
    } else if type_id == TypeId::of::<i32>() {
        DType::Int32
    } else if type_id == TypeId::of::<u32>() {
        DType::Uint32
    } else if type_id == TypeId::of::<i16>() {
        DType::Int16
    } else if type_id == TypeId::of::<u16>() {
        DType::Uint16
    } else if type_id == TypeId::of::<i8>() {
        DType::Int8
    } else if type_id == TypeId::of::<u8>() {
        DType::Uint8
    } else {
        DType::Unknown
    }
}

/// Cast a numeric value to the common scalar representation used by CLIc's
/// `castTo()` helper.
pub fn cast_to<T: Into<f64> + Copy>(value: T, dtype: DType) -> f64 {
    match dtype {
        DType::Float => value.into(),
        DType::Int32 => value.into(),
        DType::Uint32 => value.into(),
        DType::Int8 => value.into(),
        DType::Uint8 => value.into(),
        DType::Int16 => value.into(),
        DType::Uint16 => value.into(),
        DType::Complex | DType::Unknown => value.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_ocl_str() {
        assert_eq!(to_string(DType::Float), "float");
        assert_eq!(to_string(DType::Int32), "int");
        assert_eq!(to_string(DType::Uint8), "uchar");
    }

    #[test]
    fn dtype_byte_size() {
        assert_eq!(to_bytes(DType::Float), 4);
        assert_eq!(to_bytes(DType::Uint8), 1);
        assert_eq!(to_bytes(DType::Int16), 2);
    }

    #[test]
    fn dtype_short_str() {
        assert_eq!(to_short_string(DType::Float), "f");
        assert_eq!(to_short_string(DType::Uint32), "ui");
        assert_eq!(to_short_string(DType::Int8), "c");
    }

    #[test]
    fn canonical_dtype_aliases() {
        assert_eq!(to_string(DType::Uint8), "uchar");
        assert_eq!(to_short_string(DType::Uint16), "us");
        assert_eq!(to_bytes(DType::Float), 4);
        assert_eq!(to_type::<f32>(), DType::Float);
        assert_eq!(to_type::<i32>(), DType::Int32);
        assert_eq!(to_type::<u32>(), DType::Uint32);
        assert_eq!(to_type::<i16>(), DType::Int16);
        assert_eq!(to_type::<u16>(), DType::Uint16);
        assert_eq!(to_type::<i8>(), DType::Int8);
        assert_eq!(to_type::<u8>(), DType::Uint8);
        assert_eq!(cast_to(7_u8, DType::Float), 7.0);
        assert_eq!(cast_to(-3_i16, DType::Uint16), -3.0);
    }

    #[test]
    fn mtype_string_matches_clic() {
        assert_eq!(to_mtype_string(MType::Buffer), "Buffer");
        assert_eq!(to_mtype_string(MType::Image), "Image");
    }
}
