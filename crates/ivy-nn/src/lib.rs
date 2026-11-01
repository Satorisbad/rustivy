//! `ivy-nn` — layers, written once, generic over tensor shape.
//!
//! The v0 target is a LLaMA-style block: token embedding → N × (RMSNorm, GQA
//! attention with RoPE, RMSNorm, SwiGLU MLP) → RMSNorm → LM head. Every layer here
//! exists to serve that, and the op set in docs/interfaces/ops-v0.md follows from it.
#![forbid(unsafe_code)]

use ivy_core::{Error, ErrorKind, Result};
use ivy_tensor::Tensor;

/// A component with parameters and a forward pass.
pub trait Module {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;

    /// Named parameters, in a stable order, for checkpoints and the optimiser.
    fn parameters(&self) -> Vec<(&'static str, Tensor)>;

    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|(_, t)| t.numel()).sum()
    }
}

/// LLaMA-style RMS normalisation.
pub struct RmsNorm {
    pub eps: f32,
    pub weight: Tensor,
}

impl Module for RmsNorm {
    fn forward(&self, _input: &Tensor) -> Result<Tensor> {
        Err(Error::new(
            ErrorKind::Unsupported,
            "RmsNorm::forward not implemented",
        ))
    }

    fn parameters(&self) -> Vec<(&'static str, Tensor)> {
        vec![("weight", self.weight.clone())]
    }
}

/// SwiGLU feed-forward block.
pub struct SwiGlu {
    pub w1: Tensor,
    pub w2: Tensor,
    pub w3: Tensor,
}

impl Module for SwiGlu {
    fn forward(&self, _input: &Tensor) -> Result<Tensor> {
        Err(Error::new(
            ErrorKind::Unsupported,
            "SwiGlu::forward not implemented",
        ))
    }

    fn parameters(&self) -> Vec<(&'static str, Tensor)> {
        vec![
            ("w1", self.w1.clone()),
            ("w2", self.w2.clone()),
            ("w3", self.w3.clone()),
        ]
    }
}
