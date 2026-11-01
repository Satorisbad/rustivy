//! `ivy-compile` — the graph IR and its shape-bucket cache.
//!
//! Open decision O1 in docs/plan.md: whether programs are recorded from eager
//! execution (recommended) or declared up front. Burst #1 commits only to the IR
//! *type*, so that decision stays cheap.
//!
//! What is already decided: shapes are *keys*, not guards. A shape that has not
//! been seen starts a new bucket; it never silently falls back to eager and never
//! recompiles the program it already had.
#![forbid(unsafe_code)]

use ivy_core::{DType, Result, Shape};
use ivy_tensor::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

/// Operations the graph can hold. Grows with the op program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Op {
    Add,
    Mul,
    Matmul,
    RmsNorm,
    Softmax,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub id: NodeId,
    pub op: Op,
    pub inputs: Vec<NodeId>,
    pub shape: Shape,
    pub dtype: DType,
}

/// A recorded computation, in topological order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Graph {
    pub nodes: Vec<Node>,
}

impl Graph {
    pub fn push(&mut self, op: Op, inputs: Vec<NodeId>, shape: Shape, dtype: DType) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(Node {
            id,
            op,
            inputs,
            shape,
            dtype,
        });
        id
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// The shapes a compilation is specialised for. Two inputs with the same bucket
/// reuse one compiled program; a different bucket compiles a new one.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShapeBucket {
    pub dims: Vec<usize>,
    pub dtype: DType,
}

impl ShapeBucket {
    pub fn new(dims: impl Into<Vec<usize>>, dtype: DType) -> ShapeBucket {
        ShapeBucket {
            dims: dims.into(),
            dtype,
        }
    }

    /// Bucket from a real tensor. Sequence length is bucketed to the next multiple
    /// of 128 so that a 17-token batch and a 128-token batch share a program,
    /// which is the whole point of bucketing rather than exact-shape caching.
    pub fn from_tensor(t: &Tensor, bucket_seq_to: usize) -> ShapeBucket {
        let mut dims: Vec<usize> = t.shape().dims().to_vec();
        if let Some(last) = dims.last_mut()
            && bucket_seq_to > 1
        {
            *last = last.div_ceil(bucket_seq_to) * bucket_seq_to;
        }
        ShapeBucket {
            dims,
            dtype: t.dtype(),
        }
    }
}

/// Compiled artifacts keyed by bucket. Bounded, so a training run with many shapes
/// cannot grow it without limit — the failure mode torch has as a recompile storm.
pub struct Cache {
    entries: Vec<(ShapeBucket, Vec<u8>)>,
    max_entries: usize,
}

impl Cache {
    pub fn new(max_entries: usize) -> Cache {
        Cache {
            entries: Vec::new(),
            max_entries,
        }
    }

    pub fn get(&self, bucket: &ShapeBucket) -> Option<&[u8]> {
        self.entries
            .iter()
            .find(|(b, _)| b == bucket)
            .map(|(_, v)| v.as_slice())
    }

    pub fn insert(&mut self, bucket: ShapeBucket, program: Vec<u8>) -> Result<()> {
        if self.entries.len() >= self.max_entries {
            return Err(ivy_core::Error::new(
                ivy_core::ErrorKind::CompileFailed,
                format!("shape cache full at {} entries", self.max_entries),
            )
            .with_hint(
                "increase the bucket limit or coarsen bucket_seq_to; never fall back silently",
            ));
        }
        self.entries.push((bucket, program));
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ivy_tensor::Tensor;

    #[test]
    fn sequence_length_is_bucketed() {
        let t = Tensor::zeros([2usize, 17], DType::F32).unwrap();
        let b = ShapeBucket::from_tensor(&t, 128);
        assert_eq!(b.dims, vec![2, 128]);
        let t = Tensor::zeros([2usize, 300], DType::F32).unwrap();
        assert_eq!(ShapeBucket::from_tensor(&t, 128).dims, vec![2, 384]);
    }

    #[test]
    fn cache_refuses_to_grow_without_bound() {
        let mut c = Cache::new(2);
        c.insert(ShapeBucket::new(vec![1], DType::F32), vec![0])
            .unwrap();
        c.insert(ShapeBucket::new(vec![2], DType::F32), vec![0])
            .unwrap();
        assert!(
            c.insert(ShapeBucket::new(vec![3], DType::F32), vec![0])
                .is_err()
        );
    }
}
