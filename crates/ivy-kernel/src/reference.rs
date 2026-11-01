//! Scalar reference interpreter — the trusted oracle for every backend.
//!
//! Slow by construction: one element at a time, no vectorisation, no fusion. If a
//! fast backend disagrees with this, the backend is wrong.

use ivy_core::{Error, ErrorKind, Result};

use crate::ir::{BinaryOp, Program, Scalar, Stmt, UnaryOp};

/// Run `program` over `n` elements, reading and writing `f32` slices laid out in
/// buffer declaration order. Buffers are matched positionally to `buffers`.
pub fn run_f32(program: &Program, n: usize, buffers: &mut [&mut [f32]]) -> Result<()> {
    if buffers.len() != program.buffers.len() {
        return Err(Error::new(
            ErrorKind::ShapeMismatch,
            format!(
                "program declares {} buffers, {} supplied",
                program.buffers.len(),
                buffers.len()
            ),
        ));
    }
    for (i, decl) in program.buffers.iter().enumerate() {
        if decl.scalar != Scalar::F32 {
            return Err(Error::new(
                ErrorKind::UnsupportedDType,
                format!(
                    "reference interpreter handles float32 only, buffer `{}` is {:?}",
                    decl.name, decl.scalar
                ),
            ));
        }
        if buffers[i].len() < n {
            return Err(Error::new(
                ErrorKind::ShapeMismatch,
                format!(
                    "buffer `{}` has {} elements, needs {n}",
                    decl.name,
                    buffers[i].len()
                ),
            ));
        }
    }

    let mut regs = vec![0.0f32; program.registers];
    let mut idx = vec![0usize; program.registers];

    for e in 0..n {
        for stmt in &program.body {
            match *stmt {
                Stmt::GlobalIndex { dst } => idx[dst] = e,
                Stmt::Load {
                    buf,
                    idx: idx_reg,
                    dst,
                } => regs[dst] = buffers[buf][idx[idx_reg]],
                Stmt::Bin { op, a, b, dst } => regs[dst] = binop(op, regs[a], regs[b]),
                Stmt::Un { op, a, dst } => regs[dst] = unop(op, regs[a]),
                Stmt::Store {
                    buf,
                    idx: idx_reg,
                    src,
                } => buffers[buf][idx[idx_reg]] = regs[src],
            }
        }
    }
    Ok(())
}

fn binop(op: BinaryOp, a: f32, b: f32) -> f32 {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        BinaryOp::Max => a.max(b),
        BinaryOp::Min => a.min(b),
    }
}

fn unop(op: UnaryOp, a: f32) -> f32 {
    match op {
        UnaryOp::Neg => -a,
        UnaryOp::Exp => a.exp(),
        UnaryOp::Sqrt => a.sqrt(),
        UnaryOp::Rsqrt => 1.0 / a.sqrt(),
        UnaryOp::Silu => a / (1.0 + (-a).exp()),
        UnaryOp::Relu => a.max(0.0),
        UnaryOp::Gelu => 0.5 * a * (1.0 + (0.797_884_6 * (a + 0.044_715 * a * a * a)).tanh()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Program;

    #[test]
    fn elementwise_add_matches_hand_arithmetic() {
        let p = Program::elementwise_binary("add", Scalar::F32, BinaryOp::Add);
        let mut a = [1.0f32, 2.0, 3.0];
        let mut b = [10.0f32, 20.0, 30.0];
        let mut out = [0.0f32; 3];
        {
            let mut bufs: [&mut [f32]; 3] = [&mut a, &mut b, &mut out];
            run_f32(&p, 3, &mut bufs).expect("run");
        }
        assert_eq!(out, [11.0, 22.0, 33.0]);
    }

    #[test]
    fn wrong_buffer_count_is_an_error() {
        let p = Program::elementwise_binary("add", Scalar::F32, BinaryOp::Add);
        let mut a = [1.0f32];
        let mut bufs: [&mut [f32]; 1] = [&mut a];
        assert!(run_f32(&p, 1, &mut bufs).is_err());
    }
}
