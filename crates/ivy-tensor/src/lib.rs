//! `ivy-tensor` — storage, views and the two tensor types.
//!
//! Two surfaces, one implementation:
//! - [`Tensor`] is dynamic: torch-compatible shapes, runtime dtype, runtime errors.
//! - [`Tensor`]`<N>` (see [`typed`]) carries rank in the type and dtype in a
//!   type parameter, so broadcasting and device mistakes fail at compile time.
//!
//! Burst #1 implements host storage only; device storage arrives with the first
//! backend crate.
#![deny(unsafe_code)]

pub mod storage;
pub mod typed;

pub use ivy_core::{DType, Device, Error, ErrorKind, Layout, Result, Shape};
pub use storage::Storage;
pub use typed::TypedTensor;

/// Dynamic, torch-compatible tensor: shape, dtype and device are runtime values.
#[derive(Clone)]
pub struct Tensor {
    storage: Storage,
    layout: Layout,
}

impl Tensor {
    /// Allocate zeroed host storage. Device dispatch is a burst-#2 concern.
    pub fn zeros(shape: impl Into<Shape>, dtype: DType) -> Result<Tensor> {
        let shape = shape.into();
        let storage = Storage::zeros(shape.numel(), dtype, Device::Cpu)?;
        Ok(Tensor {
            storage,
            layout: Layout::contiguous(shape),
        })
    }

    /// Build from f32 data. Shape must match `data.len()`.
    pub fn from_f32(data: &[f32], shape: impl Into<Shape>) -> Result<Tensor> {
        let shape = shape.into();
        if shape.numel() != data.len() {
            return Err(Error::new(
                ErrorKind::ShapeMismatch,
                format!(
                    "shape {} holds {} elements but {} values given",
                    shape,
                    shape.numel(),
                    data.len()
                ),
            ));
        }
        let storage = Storage::from_bytes(bytemuck_cast(data), 4, DType::F32, Device::Cpu)?;
        Ok(Tensor {
            storage,
            layout: Layout::contiguous(shape),
        })
    }

    pub fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    pub fn strides(&self) -> &[isize] {
        self.layout.strides()
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    pub fn device(&self) -> &Device {
        self.storage.device()
    }

    pub fn numel(&self) -> usize {
        self.shape().numel()
    }

    pub fn rank(&self) -> usize {
        self.shape().rank()
    }

    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    /// Contiguous f32 view. Errors rather than silently copying: an implicit copy
    /// is exactly the kind of hidden cost this project exists to remove.
    pub fn as_f32(&self) -> Result<&[f32]> {
        self.storage.as_f32()
    }

    /// Reshape a contiguous tensor. Non-contiguous input is an error, not a copy.
    pub fn reshape(&self, shape: impl Into<Shape>) -> Result<Tensor> {
        let shape = shape.into();
        if !self.is_contiguous() {
            return Err(Error::new(
                ErrorKind::ShapeMismatch,
                "reshape requires a contiguous tensor",
            )
            .with_hint("call .contiguous() first, which copies explicitly"));
        }
        if shape.numel() != self.numel() {
            return Err(Error::new(
                ErrorKind::ShapeMismatch,
                format!(
                    "cannot reshape {} -> {}: {} vs {} elements",
                    self.shape(),
                    shape,
                    self.numel(),
                    shape.numel()
                ),
            ));
        }
        Ok(Tensor {
            storage: self.storage.clone(),
            layout: Layout::contiguous(shape),
        })
    }
}

fn bytemuck_cast(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for v in data {
        out.extend_from_slice(&v.to_ne_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_have_the_right_extent() {
        let t = Tensor::zeros([2usize, 3], DType::F32).expect("zeros");
        assert_eq!(t.numel(), 6);
        assert!(t.is_contiguous());
        assert_eq!(t.as_f32().expect("f32").len(), 6);
    }

    #[test]
    fn from_f32_checks_element_count() {
        assert!(Tensor::from_f32(&[1.0, 2.0, 3.0], [2usize, 2]).is_err());
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], [2usize, 2]).expect("tensor");
        assert_eq!(t.as_f32().expect("f32"), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn reshape_rejects_wrong_count() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], [4usize]).expect("tensor");
        assert!(t.reshape([3usize]).is_err());
        assert_eq!(t.reshape([2usize, 2]).expect("reshape").shape().rank(), 2);
    }
}
