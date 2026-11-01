//! Compile-time-shaped tensors.
//!
//! `TypedTensor<N, T>` carries rank and element type in the type system. This is
//! the audit's cheapest large win: broadcasting and device mismatches stop being
//! runtime surprises. The dynamic [`crate::Tensor`] stays available for torch
//! compatibility, and both are the same storage underneath.

use core::marker::PhantomData;

use ivy_core::Shape;
use ivy_core::{DType, Device, Result};

use crate::Tensor;

/// Element type marker. Implemented for the dtypes that have typed storage.
pub trait Elem: Copy + 'static {
    const DTYPE: DType;
}

impl Elem for f32 {
    const DTYPE: DType = DType::F32;
}
impl Elem for f64 {
    const DTYPE: DType = DType::F64;
}
impl Elem for F16Bits {
    const DTYPE: DType = DType::F16;
}

/// Storage representation of IEEE binary16, kept as bits so `f16` needs no
/// dependency. Arithmetic goes through a cast helper in the CPU backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct F16Bits(pub u16);

/// A tensor whose rank is known to the compiler.
///
/// ```ignore
/// let t: TypedTensor<2, f32> = TypedTensor::zeros([4, 8])?;
/// let t2 = t.transpose();          // rank still checked
/// ```
pub struct TypedTensor<const N: usize, T: Elem> {
    inner: Tensor,
    _elem: PhantomData<T>,
}

impl<const N: usize, T: Elem> TypedTensor<N, T> {
    pub fn zeros(dims: [usize; N]) -> Result<TypedTensor<N, T>> {
        let inner = Tensor::zeros(dims, T::DTYPE)?;
        Ok(TypedTensor {
            inner,
            _elem: PhantomData,
        })
    }

    pub fn shape(&self) -> [usize; N] {
        let mut out = [0usize; N];
        out.copy_from_slice(self.inner.shape().dims());
        out
    }

    pub fn dtype(&self) -> DType {
        T::DTYPE
    }

    pub fn device(&self) -> &Device {
        self.inner.device()
    }

    pub fn numel(&self) -> usize {
        self.inner.numel()
    }

    pub fn as_dynamic(&self) -> &Tensor {
        &self.inner
    }

    pub fn into_dynamic(self) -> Tensor {
        self.inner
    }

    pub fn from_dynamic(t: Tensor) -> Result<TypedTensor<N, T>> {
        if t.rank() != N {
            return Err(ivy_core::Error::new(
                ivy_core::ErrorKind::ShapeMismatch,
                format!(
                    "rank {} tensor passed where rank {N} was required",
                    t.rank()
                ),
            ));
        }
        if t.dtype() != T::DTYPE {
            return Err(ivy_core::Error::new(
                ivy_core::ErrorKind::DTypeMismatch,
                format!(
                    "{} tensor passed where {} was required",
                    t.dtype().name(),
                    T::DTYPE.name()
                ),
            ));
        }
        Ok(TypedTensor {
            inner: t,
            _elem: PhantomData,
        })
    }

    pub fn shape_of(&self) -> Shape {
        self.inner.shape().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_zeros_hold_their_rank() {
        let t: TypedTensor<2, f32> = TypedTensor::zeros([4, 8]).expect("zeros");
        assert_eq!(t.shape(), [4, 8]);
        assert_eq!(t.numel(), 32);
        assert_eq!(t.dtype(), DType::F32);
    }

    #[test]
    fn from_dynamic_rejects_wrong_rank() {
        let d = Tensor::zeros([2usize, 3], DType::F32).expect("zeros");
        assert!(TypedTensor::<3, f32>::from_dynamic(d.clone()).is_err());
        assert!(TypedTensor::<2, f32>::from_dynamic(d).is_ok());
    }
}
