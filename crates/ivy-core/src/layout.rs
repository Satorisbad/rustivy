//! `Layout` = shape + strides + storage offset. Views are layouts pointing into
//! the same storage; no copy is implied by anything in this type.

use crate::error::{Error, ErrorKind, Result};
use crate::shape::Shape;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Layout {
    shape: Shape,
    strides: Box<[isize]>,
    offset: usize,
}

impl Layout {
    pub fn contiguous(shape: Shape) -> Layout {
        let strides = shape.contiguous_strides();
        Layout {
            shape,
            strides,
            offset: 0,
        }
    }

    pub fn new(shape: Shape, strides: Box<[isize]>, offset: usize) -> Result<Layout> {
        if strides.len() != shape.rank() {
            return Err(Error::new(
                ErrorKind::StrideRankMismatch,
                format!("{} strides for rank-{} shape", strides.len(), shape.rank()),
            ));
        }
        Ok(Layout {
            shape,
            strides,
            offset,
        })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn is_contiguous(&self) -> bool {
        let mut expect: isize = 1;
        for i in (0..self.shape.rank()).rev() {
            let dim = self.shape.dims()[i];
            if dim != 1 {
                if self.strides[i] != expect {
                    return false;
                }
                expect *= dim as isize;
            }
        }
        true
    }

    /// Highest storage index this layout can touch, plus one.
    pub fn span(&self) -> usize {
        let mut max = self.offset;
        for (i, dim) in self.shape.dims().iter().enumerate() {
            if *dim > 0 {
                max += (*dim as isize - 1).max(0) as usize * self.strides[i].unsigned_abs();
            }
        }
        max + if self.shape.numel() == 0 { 0 } else { 1 }
    }
}

#[cfg(test)]
mod tests {
    use super::Layout;
    use crate::shape::Shape;

    #[test]
    fn contiguous_layout_reports_contiguous() {
        let l = Layout::contiguous(Shape::from([4usize, 8]));
        assert!(l.is_contiguous());
        assert_eq!(l.span(), 32);
    }

    #[test]
    fn transposed_is_not_contiguous() {
        let l = Layout::new(Shape::from([8usize, 4]), Box::new([1, 8]), 0).expect("layout");
        assert!(!l.is_contiguous());
    }
}
