//! Shapes and the stride algebra that sits under every view operation.

use crate::error::{Error, ErrorKind, Result};

/// A tensor shape. Owned and boxed: shapes are small and cloning one should be a
/// single allocation, not a reference-counted graph node.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(Box<[usize]>);

impl Shape {
    pub fn new(dims: impl Into<Box<[usize]>>) -> Shape {
        Shape(dims.into())
    }

    pub fn scalar() -> Shape {
        Shape(Box::new([]))
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Number of elements. A zero-sized dimension yields zero, as in torch.
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    pub fn is_scalar(&self) -> bool {
        self.0.is_empty()
    }

    /// Row-major strides for a densely packed tensor of this shape.
    ///
    /// A dimension of length 1 gets stride equal to the next stride (torch does the
    /// same in `empty_strided`) so that broadcasting a size-1 dim is stride 0.
    pub fn contiguous_strides(&self) -> Box<[isize]> {
        let mut strides = vec![0isize; self.rank()];
        let mut acc: usize = 1;
        for i in (0..self.rank()).rev() {
            // Size-1 dims get the running stride, so broadcasting is stride 0 and a
            // `[3,1]` view of `[3,4]` stays contiguous-adjacent, as in torch.
            strides[i] = acc as isize;
            acc *= self.0[i].max(1);
        }
        strides.into_boxed_slice()
    }

    /// Element count of a single logical row if this shape were flattened into
    /// `rows x width`.
    pub fn numel_of(&self, from: usize) -> usize {
        self.0[from..].iter().product()
    }

    pub fn dim(&self, axis: usize) -> Result<usize> {
        let rank = self.rank();
        let idx = if axis < rank {
            axis
        } else {
            return Err(self.axis_error(axis));
        };
        Ok(self.0[idx])
    }

    fn axis_error(&self, axis: usize) -> Error {
        Error::new(
            ErrorKind::AxisOutOfRange,
            format!("axis {axis} out of range for shape {self:?}"),
        )
        .with_hint(format!("valid axis range is 0..{}", self.rank()))
    }
}

impl core::fmt::Display for Shape {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("(")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{d}")?;
        }
        f.write_str(")")
    }
}

impl From<&[usize]> for Shape {
    fn from(v: &[usize]) -> Shape {
        Shape::new(v.to_vec())
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(v: [usize; N]) -> Shape {
        Shape::new(v.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::Shape;

    #[test]
    fn contiguous_strides_are_row_major() {
        let s = Shape::from([2usize, 3, 4]);
        assert_eq!(&*s.contiguous_strides(), &[12, 4, 1]);
        assert_eq!(s.numel(), 24);
    }

    #[test]
    fn scalar_has_no_dims() {
        let s = Shape::scalar();
        assert_eq!(s.rank(), 0);
        assert_eq!(s.numel(), 1);
    }
}
