//! One error type, with stable numeric codes.
//!
//! Every error carries a `kind()`, whose `code()` never changes: agents match on
//! the number, humans read the message, and `hint()` says what to do next. This is
//! the audit's "typed failure modes" win, made concrete.

use core::fmt;

pub type Result<T, E = Error> = core::result::Result<T, E>;

/// Coarse failure category. Stable; extend, never renumber.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    // 1xxx — shape
    ShapeMismatch,
    AxisOutOfRange,
    StrideRankMismatch,
    BroadcastIncompatible,
    // 2xxx — dtype
    DTypeMismatch,
    UnsupportedDType,
    // 3xxx — device
    DeviceMismatch,
    DeviceParse,
    DeviceUnavailable,
    // 4xxx — kernel / compile
    KernelUnsupported,
    EmitFailed,
    CompileFailed,
    // 5xxx — io
    IoFormat,
    IoCorrupt,
    // 6xxx — distributed
    CollectiveDesync,
    WorldSizeMismatch,
    // 7xxx — autograd
    GradMissing,
    InplaceOnGrad,
    // 8xxx — parity
    ParityMismatch,
    // 9xxx — internal
    Internal,
    Unsupported,
}

impl ErrorKind {
    /// Stable numeric code. Format: `E` + 4 digits, category in the first digit.
    pub const fn code(self) -> u16 {
        match self {
            ErrorKind::ShapeMismatch => 1001,
            ErrorKind::AxisOutOfRange => 1002,
            ErrorKind::StrideRankMismatch => 1003,
            ErrorKind::BroadcastIncompatible => 1004,
            ErrorKind::DTypeMismatch => 2001,
            ErrorKind::UnsupportedDType => 2002,
            ErrorKind::DeviceMismatch => 3001,
            ErrorKind::DeviceParse => 3002,
            ErrorKind::DeviceUnavailable => 3003,
            ErrorKind::KernelUnsupported => 4001,
            ErrorKind::EmitFailed => 4002,
            ErrorKind::CompileFailed => 4003,
            ErrorKind::IoFormat => 5001,
            ErrorKind::IoCorrupt => 5002,
            ErrorKind::CollectiveDesync => 6001,
            ErrorKind::WorldSizeMismatch => 6002,
            ErrorKind::GradMissing => 7001,
            ErrorKind::InplaceOnGrad => 7002,
            ErrorKind::ParityMismatch => 8001,
            ErrorKind::Internal => 9001,
            ErrorKind::Unsupported => 9002,
        }
    }

    /// One line an agent can act on without reading the message.
    pub const fn hint(self) -> &'static str {
        match self {
            ErrorKind::ShapeMismatch => "check operand shapes; RustIVY never broadcasts silently",
            ErrorKind::AxisOutOfRange => "axis must be < rank()",
            ErrorKind::StrideRankMismatch => "strides length must equal shape rank",
            ErrorKind::BroadcastIncompatible => "broadcast requires each dim to be equal or 1",
            ErrorKind::DTypeMismatch => "cast explicitly with .to(dtype)",
            ErrorKind::UnsupportedDType => "this op does not implement the requested dtype yet",
            ErrorKind::DeviceMismatch => "move both operands to the same device explicitly",
            ErrorKind::DeviceParse => "expected cpu, cuda[:N], metal, vulkan, tpu[:N]",
            ErrorKind::DeviceUnavailable => "the requested backend is not compiled in",
            ErrorKind::KernelUnsupported => "this op/dtype/backend combination has no kernel yet",
            ErrorKind::EmitFailed => "inspect the emitted program with `ivy-cli kernel --dump`",
            ErrorKind::CompileFailed => "run with --eager to compare against interpretable results",
            ErrorKind::IoFormat => "check the file magic and extension",
            ErrorKind::IoCorrupt => "checksum mismatch; re-download the checkpoint",
            ErrorKind::CollectiveDesync => {
                "ranks disagreed on collective order; check the session trace"
            }
            ErrorKind::WorldSizeMismatch => {
                "checkpoint shard layout does not match this world size"
            }
            ErrorKind::GradMissing => "the tensor does not require grad or its graph was detached",
            ErrorKind::InplaceOnGrad => "in-place op on a tensor needed by the backward pass",
            ErrorKind::ParityMismatch => {
                "our result differs from the torch oracle beyond tolerance"
            }
            ErrorKind::Internal => "bug in RustIVY; please report with the reproduction",
            ErrorKind::Unsupported => "not implemented yet; see the coverage dashboard",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Error {
    kind: ErrorKind,
    message: String,
    hint: Option<String>,
}

impl Error {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Error {
        Error {
            kind,
            message: message.into(),
            hint: None,
        }
    }

    pub fn with_hint(mut self, hint: impl Into<String>) -> Error {
        self.hint = Some(hint.into());
        self
    }

    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    pub fn code(&self) -> u16 {
        self.kind.code()
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn hint(&self) -> &str {
        self.hint.as_deref().unwrap_or_else(|| self.kind.hint())
    }

    /// Stable machine-readable form, consumed by `ivy-cli` and by agents.
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"code":"E{:04}","kind":"{:?}","message":{},"hint":{}}}"#,
            self.code(),
            self.kind,
            json_escape(&self.message),
            json_escape(self.hint())
        )
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push(' '),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "E{:04} {:?}: {}\n  hint: {}",
            self.code(),
            self.kind,
            self.message,
            self.hint()
        )
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Error {
        Error::new(ErrorKind::IoFormat, e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::{Error, ErrorKind};

    #[test]
    fn codes_are_stable_and_categorised() {
        assert_eq!(ErrorKind::ShapeMismatch.code(), 1001);
        assert_eq!(ErrorKind::ParityMismatch.code(), 8001);
    }

    #[test]
    fn json_is_well_formed_ish() {
        let e = Error::new(ErrorKind::DeviceMismatch, "cpu vs cuda:0");
        let j = e.to_json();
        assert!(j.starts_with(r#"{"code":"E3001""#));
        assert!(j.contains("\"hint\":\"move both operands"));
    }
}
