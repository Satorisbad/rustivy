//! `ivy-cpu` — the always-available backend.
//!
//! Burst #1 executes every program through the scalar reference interpreter. That
//! is the correctness floor: whatever the SIMD codegen (burst #2) produces must
//! agree with this, bit for bit, and `ivy-parity` checks it.
#![forbid(unsafe_code)]

use ivy_core::Result;
use ivy_kernel::{Program, reference};

/// Detected host capabilities, used to pick an ISA when codegen lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostCaps {
    pub avx2: bool,
    pub avx512f: bool,
    pub neon: bool,
    pub cores: usize,
}

impl HostCaps {
    pub fn detect() -> HostCaps {
        HostCaps {
            #[cfg(target_arch = "x86_64")]
            avx2: std::arch::is_x86_feature_detected!("avx2"),
            #[cfg(not(target_arch = "x86_64"))]
            avx2: false,
            #[cfg(target_arch = "x86_64")]
            avx512f: std::arch::is_x86_feature_detected!("avx512f"),
            #[cfg(not(target_arch = "x86_64"))]
            avx512f: false,
            #[cfg(target_arch = "aarch64")]
            neon: std::arch::is_aarch64_feature_detected!("neon"),
            #[cfg(not(target_arch = "aarch64"))]
            neon: false,
            cores: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
        }
    }
}

/// Execute a program over `n` elements using host memory.
pub fn execute_f32(program: &Program, n: usize, buffers: &mut [&mut [f32]]) -> Result<()> {
    reference::run_f32(program, n, buffers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ivy_kernel::{BinaryOp, Program, Scalar};

    #[test]
    fn caps_detect_without_panicking() {
        let c = HostCaps::detect();
        assert!(c.cores >= 1);
    }

    #[test]
    fn executes_elementwise_mul() {
        let p = Program::elementwise_binary("mul", Scalar::F32, BinaryOp::Mul);
        let (mut a, mut b, mut o) = ([2.0f32, 3.0], [4.0f32, 5.0], [0.0f32, 0.0]);
        let mut bufs: [&mut [f32]; 3] = [&mut a, &mut b, &mut o];
        execute_f32(&p, 2, &mut bufs).expect("exec");
        assert_eq!(o, [8.0, 15.0]);
    }
}
