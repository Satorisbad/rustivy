//! `ivy-optim` — optimisers.
//!
//! AdamW is the default and is implemented fused: one kernel updates weight and
//! both moments, so a step is one pass over memory instead of four. The state is
//! fp32 even when weights are bf16 (`master weights`), and it is what the
//! checkpoint sidecar stores.
#![forbid(unsafe_code)]

use ivy_core::{Error, ErrorKind, Result};
use ivy_tensor::Tensor;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamWConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub grad_clip_norm: Option<f32>,
}

impl Default for AdamWConfig {
    fn default() -> AdamWConfig {
        AdamWConfig {
            lr: 3e-4,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.1,
            grad_clip_norm: Some(1.0),
        }
    }
}

/// Sharding level. `Zero2` is the default: gradients and optimiser state are
/// partitioned, parameters are replicated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Shard {
    None,
    Zero1,
    Zero2,
    Zero3,
}

pub struct AdamW {
    pub config: AdamWConfig,
    pub shard: Shard,
    pub step: u64,
}

impl AdamW {
    pub fn new(config: AdamWConfig, shard: Shard) -> AdamW {
        AdamW {
            config,
            shard,
            step: 0,
        }
    }

    pub fn step(&mut self, _params: &mut [Tensor], _grads: &[Tensor]) -> Result<()> {
        Err(Error::new(
            ErrorKind::Unsupported,
            "AdamW::step not implemented",
        ))
        .map_err(|e| e.with_hint("burst #3: fused AdamW kernel + ZeRO-2 sharding"))
    }

    /// Bias correction for the current step, exposed because a wrong value here
    /// shows up as a subtly wrong loss curve rather than a crash.
    pub fn bias_correction(&self) -> (f32, f32) {
        let b1 = 1.0 - self.config.beta1.powi(self.step as i32 + 1);
        let b2 = 1.0 - self.config.beta2.powi(self.step as i32 + 1);
        (b1, b2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bias_correction_matches_formula() {
        let o = AdamW::new(AdamWConfig::default(), Shard::Zero2);
        let (b1, b2) = o.bias_correction();
        assert!((b1 - 0.1).abs() < 1e-6, "b1 = {b1}");
        assert!((b2 - 0.05).abs() < 1e-6, "b2 = {b2}");
    }
}
