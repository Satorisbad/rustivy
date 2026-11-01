//! `ivy-kernel` — one source of kernels, many backends.
//!
//! The IR is deliberately small and *not* SSA-by-construction: a program is a
//! named list of buffers plus a statement list, which every backend lowers to its
//! own model. Tiling, fusion and autotuning live here; instruction selection
//! lives in the backend crates.
//!
//! A [`reference`] interpreter runs every program on the host in scalar form. It
//! is slow and completely trustworthy, which makes it the oracle the fast
//! backends are checked against — including when torch is unavailable.
#![forbid(unsafe_code)]

pub mod autotune;
pub mod ir;
pub mod reference;

pub use autotune::{Config, Space};
pub use ir::{
    Access, BinaryOp, BufferDecl, BufferKind, Emitter, Program, Scalar, Stmt, Target, UnaryOp,
};
