//! Owned byte buffer plus its dtype and device.
//!
//! RustIVY allocates explicitly: no global caching allocator, no refcount cycle
//! collector, no "reserved but not releasable" bytes. Dropping the last handle
//! frees the buffer, and that is the whole memory story.

use std::sync::Arc;

use ivy_core::{DType, Device, Error, ErrorKind, Result};

#[derive(Clone)]
pub struct Storage {
    inner: Arc<Buffer>,
}

pub struct Buffer {
    bytes: Vec<u8>,
    len: usize,
    dtype: DType,
    device: Device,
}

impl Storage {
    pub fn zeros(len: usize, dtype: DType, device: Device) -> Result<Storage> {
        if device.is_accelerator() {
            return Err(Error::new(
                ErrorKind::DeviceUnavailable,
                format!("no allocator for {device} in this build"),
            )
            .with_hint("device storage is burst #2; use Device::Cpu"));
        }
        let bytes = vec![
            0u8;
            len.checked_mul(dtype.size_bytes()).ok_or_else(|| {
                Error::new(ErrorKind::Internal, "allocation size overflow")
            })?
        ];
        Ok(Storage {
            inner: Arc::new(Buffer {
                bytes,
                len,
                dtype,
                device,
            }),
        })
    }

    pub fn from_bytes(
        bytes: Vec<u8>,
        elem_size: usize,
        dtype: DType,
        device: Device,
    ) -> Result<Storage> {
        if elem_size != dtype.size_bytes() {
            return Err(Error::new(
                ErrorKind::DTypeMismatch,
                format!("{elem_size}-byte elements passed for {}", dtype.name()),
            ));
        }
        if !bytes.len().is_multiple_of(elem_size) {
            return Err(Error::new(
                ErrorKind::IoCorrupt,
                "buffer length is not a multiple of elem size",
            ));
        }
        let len = bytes.len() / elem_size;
        Ok(Storage {
            inner: Arc::new(Buffer {
                bytes,
                len,
                dtype,
                device,
            }),
        })
    }

    pub fn dtype(&self) -> DType {
        self.inner.dtype
    }

    pub fn device(&self) -> &Device {
        &self.inner.device
    }

    pub fn len(&self) -> usize {
        self.inner.len
    }

    pub fn is_empty(&self) -> bool {
        self.inner.len == 0
    }

    pub fn bytes(&self) -> &[u8] {
        &self.inner.bytes
    }

    pub fn as_f32(&self) -> Result<&[f32]> {
        if self.dtype() != DType::F32 {
            return Err(Error::new(
                ErrorKind::DTypeMismatch,
                format!("tensor is {}, not float32", self.dtype().name()),
            )
            .with_hint("call .to(DType::F32)"));
        }
        // Chunks of 4 little-endian bytes reinterpreted as f32.
        Ok(unsafe_f32_slice(&self.inner.bytes))
    }
}

// `#![deny(unsafe_code)]` covers this crate; the reinterpretation lives behind
// this one audited helper instead.
#[allow(unsafe_code)]
fn unsafe_f32_slice(bytes: &[u8]) -> &[f32] {
    // SAFETY: f32 has no invalid bit patterns, alignment is satisfied because the
    // buffer comes from a Vec<u8> whose allocation is at least 4-byte aligned by
    // the allocator's minimum alignment, and length is an exact multiple of 4.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), bytes.len() / 4) }
}
