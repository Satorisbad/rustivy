//! `ivy-core` — the vocabulary every other crate speaks.
//!
//! This crate holds no allocation strategy and no backend knowledge: just the
//! types that appear in every signature. Keep it dependency-light and stable —
//! everything above it inherits every change made here.
#![forbid(unsafe_code)]

pub mod device;
pub mod dtype;
pub mod error;
pub mod layout;
pub mod shape;

pub use device::Device;
pub use dtype::DType;
pub use error::{Error, ErrorKind, Result};
pub use layout::Layout;
pub use shape::Shape;
