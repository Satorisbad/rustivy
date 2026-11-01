//! Element types. `DType` is a runtime tag; the typed tensor carries the same
//! information in its type parameters so most checking happens at compile time.

use serde::{Deserialize, Serialize};

/// Runtime element type tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DType {
    Bool,
    U8,
    U32,
    I8,
    I16,
    I32,
    I64,
    F16,
    Bf16,
    F32,
    F64,
    F8E4M3,
    F8E5M2,
}

impl DType {
    /// Every dtype the runtime knows about, in promotion rank order.
    pub const ALL: &'static [DType] = &[
        DType::Bool,
        DType::U8,
        DType::U32,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::F16,
        DType::Bf16,
        DType::F32,
        DType::F64,
        DType::F8E4M3,
        DType::F8E5M2,
    ];

    /// Size in bytes of a single element.
    pub const fn size_bytes(self) -> usize {
        match self {
            DType::Bool | DType::U8 | DType::I8 | DType::F8E4M3 | DType::F8E5M2 => 1,
            DType::I16 | DType::F16 | DType::Bf16 => 2,
            DType::U32 | DType::I32 | DType::F32 => 4,
            DType::I64 | DType::F64 => 8,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            DType::Bool => "bool",
            DType::U8 => "uint8",
            DType::U32 => "uint32",
            DType::I8 => "int8",
            DType::I16 => "int16",
            DType::I32 => "int32",
            DType::I64 => "int64",
            DType::F16 => "float16",
            DType::Bf16 => "bfloat16",
            DType::F32 => "float32",
            DType::F64 => "float64",
            DType::F8E4M3 => "float8_e4m3fn",
            DType::F8E5M2 => "float8_e5m2",
        }
    }

    pub const fn is_float(self) -> bool {
        matches!(
            self,
            DType::F16 | DType::Bf16 | DType::F32 | DType::F64 | DType::F8E4M3 | DType::F8E5M2
        )
    }

    pub const fn is_signed(self) -> bool {
        matches!(self, DType::I8 | DType::I16 | DType::I32 | DType::I64) || self.is_float()
    }

    pub const fn is_int(self) -> bool {
        matches!(
            self,
            DType::U8 | DType::U32 | DType::I8 | DType::I16 | DType::I32 | DType::I64
        )
    }

    /// Accumulator type for reductions — what a sum over this dtype is computed in.
    pub const fn accumulator(self) -> DType {
        match self {
            DType::F8E4M3 | DType::F8E5M2 | DType::F16 | DType::Bf16 => DType::F32,
            DType::Bool | DType::U8 | DType::U32 | DType::I8 | DType::I16 | DType::I32 => {
                DType::I64
            }
            other => other,
        }
    }

    /// Result type of a binary op between `self` and `other`, following torch's
    /// rules for tensors (Python-scalar special cases are handled at the op layer,
    /// not here).
    pub fn promote(self, other: DType) -> DType {
        use DType::*;
        if self == other {
            return self;
        }
        // float32 is the meeting point for the half formats, exactly as in torch.
        let half = |d: DType| matches!(d, F16 | Bf16);
        if (half(self) && other == F32) || (half(other) && self == F32) {
            return F32;
        }
        if half(self) && half(other) {
            return F32;
        }
        if self == F64 || other == F64 {
            return F64;
        }
        if self.is_float() || other.is_float() {
            // Any float meeting an int widens to that float's width or wider.
            let (f, i) = if self.is_float() {
                (self, other)
            } else {
                (other, self)
            };
            return match (f, i) {
                (F16 | Bf16, I64) => F64,
                _ => f,
            };
        }
        // Both integers: wider wins, and signedness only matters at equal width.
        let (a, b) = (self.size_bytes(), other.size_bytes());
        match a.cmp(&b) {
            core::cmp::Ordering::Equal => {
                if self.is_signed() {
                    self
                } else {
                    other
                }
            }
            core::cmp::Ordering::Less => other,
            core::cmp::Ordering::Greater => self,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DType;

    #[test]
    fn promotion_is_reflexive() {
        for &d in DType::ALL {
            assert_eq!(d.promote(d), d);
        }
    }

    #[test]
    fn halves_meet_at_f32() {
        assert_eq!(DType::F16.promote(DType::Bf16), DType::F32);
        assert_eq!(DType::Bf16.promote(DType::F32), DType::F32);
    }

    #[test]
    fn ints_widen_by_rank() {
        assert_eq!(DType::I8.promote(DType::I32), DType::I32);
        assert_eq!(DType::U8.promote(DType::U8), DType::U8);
        assert_eq!(DType::U8.promote(DType::I8), DType::I8);
    }
}
