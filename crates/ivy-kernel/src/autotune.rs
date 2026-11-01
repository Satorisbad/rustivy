//! Autotuning: the search space a program declares and the config it is run with.
//!
//! Deliberately simple to start — exhaustive over a bounded space, cached by
//! (program, shape-class, device). No ML heuristics until the exhaustive path is
//! measurable.

use serde::{Deserialize, Serialize};

/// One point in the search space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Config {
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
    pub threads: u32,
    pub vectorize: u32,
    pub stage_k: bool,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            tile_m: 128,
            tile_n: 128,
            tile_k: 32,
            threads: 256,
            vectorize: 4,
            stage_k: true,
        }
    }
}

/// The bounded space a kernel searches over.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Space {
    pub tile_m: Vec<u32>,
    pub tile_n: Vec<u32>,
    pub tile_k: Vec<u32>,
    pub threads: Vec<u32>,
}

impl Space {
    /// Every point in the space, in a stable order so benchmark caches stay valid.
    pub fn exhaustive(&self) -> Vec<Config> {
        let mut out = Vec::with_capacity(self.len());
        for &m in &self.tile_m {
            for &n in &self.tile_n {
                for &k in &self.tile_k {
                    for &t in &self.threads {
                        out.push(Config {
                            tile_m: m,
                            tile_n: n,
                            tile_k: k,
                            threads: t,
                            ..Config::default()
                        });
                    }
                }
            }
        }
        out
    }

    pub fn len(&self) -> usize {
        self.tile_m.len() * self.tile_n.len() * self.tile_k.len() * self.threads.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::Space;

    #[test]
    fn exhaustive_covers_the_cartesian_product() {
        let s = Space {
            tile_m: vec![64, 128],
            tile_n: vec![64, 128],
            tile_k: vec![16],
            threads: vec![128, 256],
        };
        assert_eq!(s.len(), 8);
        assert_eq!(s.exhaustive().len(), 8);
    }

    #[test]
    fn configs_are_unique() {
        let s = Space {
            tile_m: vec![64, 128],
            tile_n: vec![64],
            tile_k: vec![16],
            threads: vec![256],
        };
        let mut v = s.exhaustive();
        v.sort_by_key(|c| (c.tile_m, c.tile_n, c.tile_k, c.threads));
        v.dedup();
        assert_eq!(v.len(), 2);
    }
}
