//! `ivy-parity` — the oracle harness.
//!
//! PyTorch lives here and nowhere else: dev-dependency on a Python venv, never in
//! shipped code. Two levels, per docs/plan.md §7:
//!
//! - **op level** (this crate, run on every commit): a few hundred ops, f32/f16/bf16,
//!   random and fuzzed shapes, relative tolerance.
//! - **model level** (`ivy-cli parity-model`, run on Colab): same seed, same data,
//!   loss curves overlaid.
//!
//! A parity failure is a failure with a code (`E8001`), not a note in a log.
#![forbid(unsafe_code)]

use ivy_core::{Error, ErrorKind, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Tolerance {
    pub rel: f32,
    pub abs: f32,
}

impl Tolerance {
    /// float32 parity against torch. Tighter than torch's own CUDA-vs-CPU drift.
    pub const F32: Tolerance = Tolerance {
        rel: 1e-4,
        abs: 1e-5,
    };
    /// Reduced precision: bf16 has 8 mantissa bits, so 1e-2 relative is honest.
    pub const BF16: Tolerance = Tolerance {
        rel: 1e-2,
        abs: 1e-2,
    };
    pub const F16: Tolerance = Tolerance {
        rel: 1e-2,
        abs: 1e-2,
    };
}

/// One comparison.
#[derive(Debug, Clone, Serialize)]
pub struct Case {
    pub op: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub tol: Tolerance,
    pub max_rel_error: f32,
    pub max_abs_error: f32,
    pub passed: bool,
}

/// Compare two float slices elementwise, ignoring positions where either side is NaN.
pub fn compare(
    op: &str,
    shape: &[usize],
    dtype: &str,
    tol: Tolerance,
    ours: &[f32],
    theirs: &[f32],
) -> Result<Case> {
    if ours.len() != theirs.len() {
        return Err(Error::new(
            ErrorKind::ParityMismatch,
            format!(
                "{op}: {} elements from us, {} from torch",
                ours.len(),
                theirs.len()
            ),
        ));
    }
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (i, (&a, &b)) in ours.iter().zip(theirs).enumerate() {
        if a.is_nan() && b.is_nan() {
            continue;
        }
        if a.is_nan() != b.is_nan() {
            return Err(Error::new(
                ErrorKind::ParityMismatch,
                format!("{op}: NaN on one side only at index {i} ({a} vs {b})"),
            ));
        }
        let abs = (a - b).abs();
        let rel = if b == 0.0 { abs } else { abs / b.abs() };
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
    }
    let passed = max_abs <= tol.abs || max_rel <= tol.rel;
    let case = Case {
        op: op.to_string(),
        shape: shape.to_vec(),
        dtype: dtype.to_string(),
        tol,
        max_rel_error: max_rel,
        max_abs_error: max_abs,
        passed,
    };
    if !passed {
        return Err(Error::new(
            ErrorKind::ParityMismatch,
            format!(
                "{op}: max rel {max_rel:e} > {:e}, max abs {max_abs:e} > {:e}",
                tol.rel, tol.abs
            ),
        ));
    }
    Ok(case)
}

/// Runs expressions against a real torch install through a subprocess.
pub struct Oracle {
    pub python: String,
    pub torch_available: bool,
}

impl Oracle {
    /// Discover the operator's python env. Reports availability rather than
    /// failing, so CPU-only CI still builds and only the oracle tests skip.
    pub fn discover(python: impl Into<String>) -> Oracle {
        let python = python.into();
        let out = std::process::Command::new(&python)
            .args(["-c", "import torch; print(torch.__version__)"])
            .output();
        let torch_available = matches!(&out, Ok(o) if o.status.success());
        Oracle {
            python,
            torch_available,
        }
    }

    /// Evaluate a torch expression and return its flattened float32 values.
    pub fn eval_f32(&self, expr: &str) -> Result<Vec<f32>> {
        if !self.torch_available {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "torch oracle unavailable in this python",
            )
            .with_hint("point IVY_PYTHON at the venv with torch installed"));
        }
        let script = format!(
            "import torch, json, sys\nt=({expr})\nif t.dtype!=torch.float32: t=t.float()\nsys.stdout.write(json.dumps(t.reshape(-1).tolist()))"
        );
        let out = std::process::Command::new(&self.python)
            .args(["-c", &script])
            .output()
            .map_err(|e| Error::new(ErrorKind::Internal, format!("failed to run oracle: {e}")))?;
        if !out.status.success() {
            return Err(Error::new(
                ErrorKind::ParityMismatch,
                format!(
                    "oracle expression failed: {}",
                    String::from_utf8_lossy(&out.stderr)
                ),
            ));
        }
        serde_json::from_slice(&out.stdout).map_err(|e| {
            Error::new(
                ErrorKind::Internal,
                format!("oracle produced non-JSON: {e}"),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_slices_pass() {
        let c = compare("add", &[2], "f32", Tolerance::F32, &[1.0, 2.0], &[1.0, 2.0]).unwrap();
        assert!(c.passed);
    }

    #[test]
    fn drift_beyond_tolerance_fails_with_e8001() {
        let e = compare("add", &[2], "f32", Tolerance::F32, &[1.0, 2.0], &[1.0, 2.1]).unwrap_err();
        assert_eq!(e.code(), 8001);
    }

    #[test]
    fn nan_on_one_side_is_a_failure() {
        assert!(compare("x", &[1], "f32", Tolerance::F32, &[f32::NAN], &[1.0]).is_err());
        assert!(compare("x", &[1], "f32", Tolerance::F32, &[f32::NAN], &[f32::NAN]).is_ok());
    }
}
