//! `rustivy` — umbrella crate. Re-exports the RustIVY stack under one name.
//!
//! Layers are re-exported in dependency order; depend on the narrowest crate you
//! need, and reach for this only when you want everything.

pub use ivy_core as core;
pub use ivy_tensor as tensor;

#[cfg(feature = "autograd")]
pub use ivy_autograd as autograd;
#[cfg(feature = "nn")]
pub use ivy_nn as nn;
#[cfg(feature = "optim")]
pub use ivy_optim as optim;
