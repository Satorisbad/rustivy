//! Run state: everything needed to resume, not just the weights.
//!
//! A checkpoint is a directory: `model.safetensors` for the tensors, and
//! `run_state.json` for the things that make a resume actually a resume — optimiser
//! moments, RNG state, dataloader position, LR schedule and the shard layout, so a
//! run resumed onto a different world size can re-map instead of restarting.
//!
//! Two halves:
//! - [`RunState`] is the JSON sidecar (scalars, RNG bytes, the plan it was written under).
//! - [`ShardedMatrix`] is the optimiser-state layout, and [`ShardedMatrix::remap`] is
//!   the data move a plan change needs. `can_resume_onto` only *validates* a resume;
//!   the re-map is what makes it work.

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

    /// Total ranks the recorded plan describes.
    pub fn plan_world_size(&self) -> Result<u32> {
        plan_product(self.plan)
    }

    /// A resume is allowed onto a different world size as long as the *product* is
    /// unchanged — that is the whole point of shard-agnostic state.
    pub fn can_resume_onto(&self, new_world_size: u32) -> Result<()> {
        let recorded = self.plan_world_size()?;
        if recorded != self.world_size {
            return Err(Error::new(
                ErrorKind::WorldSizeMismatch,
                format!(
                    "run_state is inconsistent: plan {:?} is {recorded} ranks but world_size is {}",
                    self.plan, self.world_size
                ),
            )
            .with_hint("a checkpoint written by a buggy writer; re-shard it deliberately"));
        }
        if recorded != new_world_size {
            return Err(Error::new(
                ErrorKind::WorldSizeMismatch,
                format!(
                    "checkpoint written for {recorded} ranks, resume requested with {new_world_size}"
                ),
            )
            .with_hint("shard-agnostic checkpoints still require the same total world size"));
        }
        Ok(())
    }

    /// Re-map optimiser state written under this checkpoint's plan onto `new_plan`.
    ///
    /// This is the data move behind `can_resume_onto`: the plan may change shape
    /// (for example `(2, 2, 2)` → `(4, 1, 2)`) as long as the total rank count is
    /// preserved, because the union of the shards is the same global buffer.
    pub fn remap_optimiser_state(
        &self,
        state: &ShardedMatrix,
        new_plan: (u32, u32, u32),
    ) -> Result<ShardedMatrix> {
        if state.plan != self.plan {
            return Err(Error::new(
                ErrorKind::WorldSizeMismatch,
                format!(
                    "optimiser state was sharded for plan {:?} but the checkpoint records {:?}",
                    state.plan, self.plan
                ),
            )
            .with_hint("re-map the state that was written alongside this run_state"));
        }
        self.can_resume_onto(plan_product(new_plan)?)?;
        state.remap(new_plan)
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

/// Optimiser state for one parameter, sharded as a 2-D block grid over `(dp, tp)`.
///
/// A parameter's optimiser moments are a single global matrix; each rank owns one
/// contiguous row-block × column-block of it. That is the layout ZeRO/TP resume
/// has to move when the plan changes shape.
///
/// `pp` splits *layers* between pipeline stages, not a single tensor, so a given
/// parameter lives on the `dp × tp` ranks of exactly one stage. Hence
/// `shards.len() == dp * tp`, and a re-map requires `pp` to be unchanged: moving a
/// parameter to a different stage needs the layer map, which `ivy-dist` owns.
///
/// Rank order inside `shards` is `rank = d * tp + t`.
#[derive(Debug, Clone, PartialEq)]
pub struct ShardedMatrix {
    /// Global row count.
    pub rows: usize,
    /// Global column count.
    pub cols: usize,
    /// Plan the shards were cut under.
    pub plan: (u32, u32, u32),
    /// One row-major block per `(dp, tp)` rank, in rank order.
    pub shards: Vec<Vec<f32>>,
}

impl ShardedMatrix {
    /// Cut a global row-major matrix into the `(dp, tp)` block grid of `plan`.
    pub fn shard(
        global: &[f32],
        rows: usize,
        cols: usize,
        plan: (u32, u32, u32),
    ) -> Result<ShardedMatrix> {
        let (dp, tp, pp) = plan;
        if dp == 0 || tp == 0 || pp == 0 {
            return Err(Error::new(
                ErrorKind::WorldSizeMismatch,
                format!("plan {plan:?} has a zero dimension"),
            )
            .with_hint("every parallel degree must be at least 1"));
        }
        let want = rows.checked_mul(cols).ok_or_else(|| {
            Error::new(
                ErrorKind::ShapeMismatch,
                format!("{rows}×{cols} overflows usize"),
            )
        })?;
        if want != global.len() {
            return Err(Error::new(
                ErrorKind::ShapeMismatch,
                format!(
                    "{rows}×{cols} is {want} values but {} were given",
                    global.len()
                ),
            ));
        }

        let mut shards = Vec::with_capacity((dp as usize) * (tp as usize));
        for d in 0..dp {
            let (r0, rn) = split(rows, dp, d);
            for t in 0..tp {
                let (c0, cn) = split(cols, tp, t);
                let mut block = Vec::with_capacity(rn * cn);
                for r in r0..r0 + rn {
                    let row = r * cols;
                    block.extend_from_slice(&global[row + c0..row + c0 + cn]);
                }
                shards.push(block);
            }
        }
        Ok(ShardedMatrix {
            rows,
            cols,
            plan,
            shards,
        })
    }

    /// Gather the blocks back into the global row-major matrix.
    pub fn gather(&self) -> Result<Vec<f32>> {
        let (dp, tp, _) = self.plan;
        self.check_layout()?;
        let mut out = vec![0f32; self.rows * self.cols];
        for d in 0..dp {
            let (r0, rn) = split(self.rows, dp, d);
            for t in 0..tp {
                let (c0, cn) = split(self.cols, tp, t);
                let block = &self.shards[(d * tp + t) as usize];
                for i in 0..rn {
                    let src = i * cn;
                    let row = (r0 + i) * self.cols;
                    out[row + c0..row + c0 + cn].copy_from_slice(&block[src..src + cn]);
                }
            }
        }
        Ok(out)
    }

    /// Re-map onto `new_plan`, which must describe the same total world size.
    ///
    /// The global buffer is unchanged by a plan change, so this gathers and
    /// re-cuts; no value is dropped and none is duplicated.
    pub fn remap(&self, new_plan: (u32, u32, u32)) -> Result<ShardedMatrix> {
        let old = plan_product(self.plan)?;
        let new = plan_product(new_plan)?;
        if old != new {
            return Err(Error::new(
                ErrorKind::WorldSizeMismatch,
                format!(
                    "cannot re-shard plan {:?} ({old} ranks) onto {:?} ({new} ranks)",
                    self.plan, new_plan
                ),
            )
            .with_hint("a re-map preserves the total rank count; the product must match"));
        }
        if self.plan.2 != new_plan.2 {
            return Err(Error::new(
                ErrorKind::WorldSizeMismatch,
                format!(
                    "cannot re-shard across a pipeline change: pp {} -> {}",
                    self.plan.2, new_plan.2
                ),
            )
            .with_hint(
                "pp assigns layers to stages, so a parameter moving stage needs the layer map; \
                 re-map with pp unchanged",
            ));
        }
        let global = self.gather()?;
        ShardedMatrix::shard(&global, self.rows, self.cols, new_plan)
    }

    /// Shape of the block owned by `rank`, as `(row_start, row_len, col_start, col_len)`.
    pub fn block_bounds(&self, rank: usize) -> Result<(usize, usize, usize, usize)> {
        let (dp, tp, _) = self.plan;
        let grid = (dp as usize) * (tp as usize);
        if rank >= grid {
            return Err(Error::new(
                ErrorKind::WorldSizeMismatch,
                format!("rank {rank} is outside the {grid}-rank (dp×tp) grid"),
            ));
        }
        let d = rank as u32 / tp;
        let t = rank as u32 % tp;
        let (r0, rn) = split(self.rows, dp, d);
        let (c0, cn) = split(self.cols, tp, t);
        Ok((r0, rn, c0, cn))
    }

    /// The shards must be exactly as many as the plan says, each the size its
    /// block bounds imply. A checkpoint that fails this is corrupt, not re-mappable.
    fn check_layout(&self) -> Result<()> {
        let (dp, tp, _) = self.plan;
        let grid = (dp as usize) * (tp as usize);
        if self.shards.len() != grid {
            return Err(Error::new(
                ErrorKind::WorldSizeMismatch,
                format!(
                    "plan {:?} needs {grid} shards but {} are present",
                    self.plan,
                    self.shards.len()
                ),
            )
            .with_hint("optimiser state does not match the plan recorded beside it"));
        }
        for rank in 0..grid {
            let (_, rn, _, cn) = self.block_bounds(rank)?;
            let want = rn * cn;
            if self.shards[rank].len() != want {
                return Err(Error::new(
                    ErrorKind::WorldSizeMismatch,
                    format!(
                        "shard {rank} holds {} values but its {rn}×{cn} block needs {want}",
                        self.shards[rank].len()
                    ),
                )
                .with_hint("optimiser state does not match the plan recorded beside it"));
            }
        }
        Ok(())
    }
}

/// Split `n` items into `parts` contiguous blocks, remainder on the first blocks.
///
/// `split(7, 2, 0) == (0, 4)`, `split(7, 2, 1) == (4, 3)`.
fn split(n: usize, parts: u32, index: u32) -> (usize, usize) {
    let p = parts as usize;
    let i = index as usize;
    let base = n / p;
    let rem = n % p;
    let start = i * base + i.min(rem);
    let len = base + usize::from(i < rem);
    (start, len)
}

fn plan_product(plan: (u32, u32, u32)) -> Result<u32> {
    let (dp, tp, pp) = plan;
    if dp == 0 || tp == 0 || pp == 0 {
        return Err(Error::new(
            ErrorKind::WorldSizeMismatch,
            format!("plan {plan:?} has a zero dimension"),
        )
        .with_hint("every parallel degree must be at least 1"));
    }
    let product = u64::from(dp) * u64::from(tp) * u64::from(pp);
    u32::try_from(product).map_err(|_| {
        Error::new(
            ErrorKind::WorldSizeMismatch,
            format!("plan {plan:?} describes more ranks than a world can hold"),
        )
    })
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

    #[test]
    fn split_balances_the_remainder() {
        assert_eq!(split(7, 2, 0), (0, 4));
        assert_eq!(split(7, 2, 1), (4, 3));
        assert_eq!(split(0, 3, 2), (0, 0));
        assert_eq!(split(6, 2, 1), (3, 3));
    }

    /// The ticket's case: (2,2,2) → (4,1,2) keeps the product, and every value
    /// survives the move.
    #[test]
    fn reshard_moves_every_value() {
        let rows = 8usize;
        let cols = 6usize;
        let global: Vec<f32> = (0..(rows * cols)).map(|i| i as f32).collect();

        let old = ShardedMatrix::shard(&global, rows, cols, (2, 2, 2)).expect("shard");
        assert_eq!(old.shards.len(), 4);
        assert_eq!(old.shards[0].len(), 4 * 3);

        let new = old.remap((4, 1, 2)).expect("remap");
        assert_eq!(new.plan, (4, 1, 2));
        assert_eq!(new.shards.len(), 4);
        assert_eq!(new.shards[0].len(), 2 * 6);

        // Values, not just shapes: rank 0 of the new plan owns rows 0..2, all cols.
        assert_eq!(new.shards[0], global[0..12]);
        // Rank 3 owns rows 6..8, all cols.
        assert_eq!(new.shards[3], global[36..48]);

        // And the whole thing gathers back bit-for-bit.
        assert_eq!(new.gather().expect("gather"), global);
        assert_eq!(old.gather().expect("gather"), global);
    }

    #[test]
    fn reshard_back_to_the_original_plan_is_a_round_trip() {
        let global: Vec<f32> = (0..35).map(|i| i as f32 * 0.5).collect();
        let old = ShardedMatrix::shard(&global, 7, 5, (2, 2, 2)).expect("shard");
        let there = old.remap((4, 1, 2)).expect("remap out");
        let back = there.remap((2, 2, 2)).expect("remap home");
        assert_eq!(back, old);
    }

    #[test]
    fn reshard_rejects_a_different_world_size() {
        let global: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let s = ShardedMatrix::shard(&global, 4, 3, (2, 2, 2)).expect("shard");
        let err = s.remap((4, 2, 2)).expect_err("8 -> 16 ranks");
        assert_eq!(err.code(), 6002, "{err}");
    }

    #[test]
    fn reshard_rejects_a_pipeline_change() {
        let global: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let s = ShardedMatrix::shard(&global, 4, 3, (2, 2, 2)).expect("shard");
        // Same product (8), different pp.
        let err = s.remap((4, 2, 1)).expect_err("pp changed");
        assert_eq!(err.code(), 6002, "{err}");
        assert!(err.hint().contains("layer map"), "{err}");
    }

    #[test]
    fn run_state_drives_the_remap() {
        let global: Vec<f32> = (0..48).map(|i| i as f32).collect();
        let state = ShardedMatrix::shard(&global, 8, 6, (2, 2, 2)).expect("shard");
        let run = RunState::new((2, 2, 2), 8);

        let moved = run.remap_optimiser_state(&state, (4, 1, 2)).expect("remap");
        assert_eq!(moved.gather().expect("gather"), global);

        // A plan whose product moved is refused before any data is touched.
        assert!(run.remap_optimiser_state(&state, (4, 1, 1)).is_err());
        // State written under another plan is refused too.
        let other = ShardedMatrix::shard(&global, 8, 6, (4, 1, 2)).expect("shard");
        let err = run
            .remap_optimiser_state(&other, (4, 1, 2))
            .expect_err("plan mismatch");
        assert_eq!(err.code(), 6002, "{err}");
    }

    #[test]
    fn a_short_shard_is_corrupt_not_remappable() {
        let global: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let mut s = ShardedMatrix::shard(&global, 4, 3, (2, 2, 2)).expect("shard");
        s.shards[1].pop();
        assert_eq!(s.gather().expect_err("short shard").code(), 6002);
        assert_eq!(s.remap((4, 1, 2)).expect_err("short shard").code(), 6002);
    }

    #[test]
    fn shard_rejects_a_size_mismatch() {
        let global = vec![0f32; 11];
        assert_eq!(
            ShardedMatrix::shard(&global, 4, 3, (2, 2, 2))
                .expect_err("11 != 12")
                .code(),
            1001
        );
    }
}
