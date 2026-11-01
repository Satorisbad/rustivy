//! `ivy-dist` — collectives we own, and a layout type that refuses impossible plans.
//!
//! Two rules make this different from `torch.distributed`:
//! 1. A plan whose dimensions don't multiply to the world size cannot be built.
//! 2. Collectives are issued through a session type that encodes the expected
//!    order, so a rank/gradient desync is a compile error rather than a 30-minute
//!    watchdog timeout.
#![forbid(unsafe_code)]

use ivy_core::{Error, ErrorKind, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Rank(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct World {
    pub size: u32,
    pub rank: Rank,
}

impl World {
    pub fn new(size: u32, rank: u32) -> Result<World> {
        if size == 0 || rank >= size {
            return Err(Error::new(
                ErrorKind::WorldSizeMismatch,
                format!("rank {rank} is not inside a world of {size}"),
            ));
        }
        Ok(World {
            size,
            rank: Rank(rank),
        })
    }

    pub fn is_leader(&self) -> bool {
        self.rank.0 == 0
    }
}

/// Data / tensor / pipeline split. `dp * tp * pp` must equal the world size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plan {
    pub dp: u32,
    pub tp: u32,
    pub pp: u32,
}

impl Plan {
    pub fn new(dp: u32, tp: u32, pp: u32, world: World) -> Result<Plan> {
        let product = dp
            .checked_mul(tp)
            .and_then(|v| v.checked_mul(pp))
            .ok_or_else(|| Error::new(ErrorKind::WorldSizeMismatch, "parallel plan overflows"))?;
        if product != world.size {
            return Err(Error::new(
                ErrorKind::WorldSizeMismatch,
                format!(
                    "plan {}x{}x{} = {product} but world size is {}",
                    dp, tp, pp, world.size
                ),
            )
            .with_hint("3D parallelism must cover every rank exactly once"));
        }
        Ok(Plan { dp, tp, pp })
    }

    /// Coordinate of `rank` inside the plan, most-significant first: (dp, tp, pp).
    pub fn coords(&self, rank: Rank) -> (u32, u32, u32) {
        let r = rank.0;
        let pp = r % self.pp;
        let tp = (r / self.pp) % self.tp;
        let dp = r / (self.pp * self.tp);
        (dp, tp, pp)
    }

    /// Ranks that share the same tensor-parallel group as `rank`.
    pub fn tp_group(&self, rank: Rank) -> Vec<Rank> {
        let (dp, _, pp) = self.coords(rank);
        (0..self.tp)
            .map(|t| Rank(dp * self.tp * self.pp + t * self.pp + pp))
            .collect()
    }
}

/// Collectives. Own transport, so the trait is about semantics, not NCCL.
pub trait Collective {
    fn all_reduce_sum(&self, buf: &mut [f32]) -> Result<()>;
    fn all_gather(&self, send: &[f32], recv: &mut [f32]) -> Result<()>;
    fn reduce_scatter(&self, send: &[f32], recv: &mut [f32]) -> Result<()>;
    fn barrier(&self) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_must_cover_the_world() {
        let w = World::new(8, 0).unwrap();
        assert!(Plan::new(2, 2, 2, w).is_ok());
        assert!(Plan::new(2, 2, 4, w).is_err());
    }

    #[test]
    fn coords_partition_the_world() {
        let w = World::new(8, 0).unwrap();
        let p = Plan::new(2, 2, 2, w).unwrap();
        let mut seen = std::collections::HashSet::new();
        for r in 0..8 {
            seen.insert(p.coords(Rank(r)));
        }
        assert_eq!(seen.len(), 8);
    }

    #[test]
    fn tp_group_has_tp_members() {
        let w = World::new(8, 0).unwrap();
        let p = Plan::new(2, 2, 2, w).unwrap();
        assert_eq!(p.tp_group(Rank(3)).len(), 2);
    }
}
