//! `ivy-autograd` — reverse-mode differentiation over a typed tape.
//!
//! Design commitments (see docs/plan.md):
//! - The tape is a `Vec` of nodes, not a graph of `Rc<RefCell<..>>`; no cycles, no
//!   refcount churn, and backward is one linear pass in reverse.
//! - Every derivative is registered explicitly. There is no silent
//!   `once_differentiable` escape hatch: an op either has a vjp or it cannot
//!   participate in a graph that requires grad.
//! - Double-backward is a supported path, not a hole.
//!
//! Burst #1 is interface only.
#![forbid(unsafe_code)]

use ivy_core::{Error, ErrorKind, Result};
use ivy_tensor::Tensor;

/// Index into a [`Tape`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

/// One recorded operation: inputs, output, and how to differentiate it.
pub struct Node {
    pub id: NodeId,
    pub op: &'static str,
    pub inputs: Vec<NodeId>,
    pub output: Tensor,
    pub requires_grad: bool,
}

/// Append-only tape. Node ids are stable and ordered by creation.
#[derive(Default)]
pub struct Tape {
    nodes: Vec<Node>,
}

impl Tape {
    pub fn new() -> Tape {
        Tape { nodes: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn push(
        &mut self,
        op: &'static str,
        inputs: Vec<NodeId>,
        output: Tensor,
        requires_grad: bool,
    ) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(Node {
            id,
            op,
            inputs,
            output,
            requires_grad,
        });
        id
    }

    pub fn get(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id.0 as usize)
    }

    /// Reverse-order walk from `root`, visiting every node that needs a gradient.
    pub fn backward_order(&self, root: NodeId) -> Vec<NodeId> {
        let mut order = Vec::new();
        let mut stack = vec![root];
        let mut seen = vec![false; self.nodes.len()];
        while let Some(id) = stack.pop() {
            let i = id.0 as usize;
            if seen[i] {
                continue;
            }
            seen[i] = true;
            order.push(id);
            if let Some(node) = self.get(id) {
                for &input in &node.inputs {
                    stack.push(input);
                }
            }
        }
        order.reverse();
        order
    }
}

/// Executes the backward pass. Implemented in burst #2 alongside the first real ops.
pub fn backward(_tape: &Tape, _root: NodeId, _seed: Tensor) -> Result<()> {
    Err(Error::new(
        ErrorKind::Unsupported,
        "backward pass is not implemented yet",
    )
    .with_hint("burst #2: tape + elementwise vjps first, then matmul and norm"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ivy_tensor::DType;

    #[test]
    fn tape_records_and_orders_backward() {
        let mut t = Tape::new();
        let a = t.push(
            "leaf",
            vec![],
            Tensor::zeros([2usize], DType::F32).unwrap(),
            true,
        );
        let b = t.push(
            "leaf",
            vec![],
            Tensor::zeros([2usize], DType::F32).unwrap(),
            true,
        );
        let c = t.push(
            "add",
            vec![a, b],
            Tensor::zeros([2usize], DType::F32).unwrap(),
            true,
        );
        let order = t.backward_order(c);
        assert_eq!(order, vec![a, b, c]);
    }
}
