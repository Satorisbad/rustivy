//! `ivy-spirv` — portable backend.
//!
//! Not started. Kept as a crate from day one so the emitter trait is exercised by
//! four real targets rather than two, which is what stops the trait quietly
//! becoming NVIDIA-shaped.
#![forbid(unsafe_code)]

use ivy_core::{Error, ErrorKind, Result};
use ivy_kernel::{Emitter, Program, Target};

#[derive(Debug, Clone, Copy, Default)]
pub struct SpirvEmitter;

impl Emitter for SpirvEmitter {
    fn target(&self) -> Target {
        Target::Spirv
    }

    fn emit(&self, program: &Program) -> Result<String> {
        Err(Error::new(
            ErrorKind::KernelUnsupported,
            format!(
                "SPIR-V backend does not exist yet (program `{}`)",
                program.name
            ),
        )
        .with_hint("scheduled after Metal; see docs/plan.md §3 decision 13"))
    }

    fn supports(&self, _program: &Program) -> bool {
        false
    }
}
