//! Run state: everything needed to resume, not just the weights.
//!
//! A checkpoint is a directory: `model.safetensors` for the tensors, and
//! `run_state.json` for the things that make a resume actually a resume — optimiser
//! moments, RNG state, dataloader position, LR schedule and the shard layout, so a
//! run resumed onto a different world size can re-map instead of restarting.

use std::path::Path;

use ivy_core::{Error, ErrorKind, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunState {
    pub global_step: u64,
    pub tokens_seen: u64,
    /// Opaque RNG state per stream, so a resumed run draws the same batches.
    pub rng: Vec<u8>,
    /// Byte offset into the token stream.
    pub data_position: u64,
    pub lr_schedule: Vec<(u64, f32)>,
    /// Parallel plan the checkpoint was written under, so a resume can re-shard.
    pub plan: (u32, u32, u32),
    pub world_size: u32,
    pub sha256_manifest: String,
}

impl RunState {
    pub fn new(plan: (u32, u32, u32), world_size: u32) -> RunState {
        RunState {
            global_step: 0,
            tokens_seen: 0,
            rng: Vec::new(),
            data_position: 0,
            lr_schedule: Vec::new(),
            plan,
            world_size,
            sha256_manifest: String::new(),
        }
    }

    /// A resume is allowed onto a different world size as long as the *product* is
    /// unchanged — that is the whole point of shard-agnostic state.
    pub fn can_resume_onto(&self, new_world_size: u32) -> Result<()> {
        let (dp, tp, pp) = self.plan;
        let old = dp * tp * pp;
        if old != new_world_size {
            return Err(Error::new(
                ErrorKind::WorldSizeMismatch,
                format!(
                    "checkpoint written for {old} ranks, resume requested with {new_world_size}"
                ),
            )
            .with_hint("shard-agnostic checkpoints still require the same total world size"));
        }
        Ok(())
    }

    pub fn write(&self, dir: impl AsRef<Path>) -> Result<()> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        let json = serde_json::to_vec_pretty(self)
            .map_err(|e| Error::new(ErrorKind::IoFormat, e.to_string()))?;
        std::fs::write(dir.join("run_state.json"), json)?;
        Ok(())
    }

    pub fn read(dir: impl AsRef<Path>) -> Result<RunState> {
        let bytes = std::fs::read(dir.as_ref().join("run_state.json"))?;
        serde_json::from_slice(&bytes)
            .map_err(|e| Error::new(ErrorKind::IoCorrupt, format!("run_state.json: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resume_requires_matching_world_size() {
        let s = RunState::new((2, 2, 2), 8);
        assert!(s.can_resume_onto(8).is_ok());
        assert!(s.can_resume_onto(4).is_err());
    }

    #[test]
    fn round_trips_through_disk() {
        let dir = std::env::temp_dir().join("ivy-run-state-test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut s = RunState::new((1, 1, 1), 1);
        s.global_step = 42;
        s.write(&dir).unwrap();
        let back = RunState::read(&dir).unwrap();
        assert_eq!(back.global_step, 42);
    }
}
