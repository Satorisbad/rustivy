//! Kernel IR.

use ivy_core::{Error, ErrorKind, Result};

/// Backend the program is being emitted for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Target {
    Ptx { sm: u16 },
    CpuSimd,
    Msl,
    Spirv,
}

impl Target {
    pub fn name(self) -> String {
        match self {
            Target::Ptx { sm } => format!("ptx-sm{sm}"),
            Target::CpuSimd => "cpu-simd".into(),
            Target::Msl => "metal".into(),
            Target::Spirv => "spirv".into(),
        }
    }
}

/// Scalar type inside a kernel. Deliberately narrower than [`ivy_core::DType`]:
/// a dtype with no arithmetic support is not a kernel scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Scalar {
    F32,
    F16,
    Bf16,
    I32,
    U32,
    Bool,
}

impl Scalar {
    pub fn size_bytes(self) -> usize {
        match self {
            Scalar::F32 | Scalar::I32 | Scalar::U32 => 4,
            Scalar::F16 | Scalar::Bf16 => 2,
            Scalar::Bool => 1,
        }
    }

    pub fn from_dtype(d: ivy_core::DType) -> Result<Scalar> {
        use ivy_core::DType;
        Ok(match d {
            DType::F32 => Scalar::F32,
            DType::F16 => Scalar::F16,
            DType::Bf16 => Scalar::Bf16,
            DType::I32 => Scalar::I32,
            DType::U32 => Scalar::U32,
            DType::Bool => Scalar::Bool,
            other => {
                return Err(Error::new(
                    ErrorKind::UnsupportedDType,
                    format!("{} is not a supported kernel scalar", other.name()),
                ));
            }
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferKind {
    Input,
    Output,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    Read,
    Write,
    ReadWrite,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferDecl {
    pub name: String,
    pub scalar: Scalar,
    pub kind: BufferKind,
    pub access: Access,
}

impl BufferDecl {
    pub fn input(name: impl Into<String>, scalar: Scalar) -> BufferDecl {
        BufferDecl {
            name: name.into(),
            scalar,
            kind: BufferKind::Input,
            access: Access::Read,
        }
    }

    pub fn output(name: impl Into<String>, scalar: Scalar) -> BufferDecl {
        BufferDecl {
            name: name.into(),
            scalar,
            kind: BufferKind::Output,
            access: Access::Write,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Exp,
    Sqrt,
    Rsqrt,
    Silu,
    Relu,
    Gelu,
}

/// Statement list. Element indices are expressed as [`Stmt::Index`] so backends
/// can fold them into address arithmetic without a separate lowering pass.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    /// `dst = scalar_load(buf, idx)`
    Load { buf: usize, idx: usize, dst: usize },
    /// `dst = scalar_binop(a, b)`
    Bin {
        op: BinaryOp,
        a: usize,
        b: usize,
        dst: usize,
    },
    /// `dst = scalar_unop(a)`
    Un { op: UnaryOp, a: usize, dst: usize },
    /// `store(buf, idx, src)`
    Store { buf: usize, idx: usize, src: usize },
    /// Element index for the current invocation, written into a register slot.
    GlobalIndex { dst: usize },
}

/// A kernel program: buffers, launch shape, and a statement list.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub name: String,
    pub buffers: Vec<BufferDecl>,
    /// Workgroup dimensions. `[x, y, z]`; elementwise programs use `[256, 1, 1]`.
    pub workgroup: [u32; 3],
    pub body: Vec<Stmt>,
    /// Highest register slot index used by `body`, plus one.
    pub registers: usize,
}

impl Program {
    /// The canonical elementwise binary program: `out[i] = a[i] <op> b[i]`.
    pub fn elementwise_binary(name: impl Into<String>, scalar: Scalar, op: BinaryOp) -> Program {
        Program {
            name: name.into(),
            buffers: vec![
                BufferDecl::input("a", scalar),
                BufferDecl::input("b", scalar),
                BufferDecl::output("out", scalar),
            ],
            workgroup: [256, 1, 1],
            body: vec![
                Stmt::GlobalIndex { dst: 0 },
                Stmt::Load {
                    buf: 0,
                    idx: 0,
                    dst: 1,
                },
                Stmt::Load {
                    buf: 1,
                    idx: 0,
                    dst: 2,
                },
                Stmt::Bin {
                    op,
                    a: 1,
                    b: 2,
                    dst: 3,
                },
                Stmt::Store {
                    buf: 2,
                    idx: 0,
                    src: 3,
                },
            ],
            registers: 4,
        }
    }

    /// Elements launched per workgroup, as a flat count.
    pub fn workgroup_size(&self) -> u32 {
        self.workgroup[0] * self.workgroup[1] * self.workgroup[2]
    }

    pub fn output_buffers(&self) -> impl Iterator<Item = &BufferDecl> {
        self.buffers.iter().filter(|b| b.kind == BufferKind::Output)
    }
}

/// Everything a backend crate implements.
pub trait Emitter {
    fn target(&self) -> Target;

    /// Lower a program to backend source (PTX text, MSL source, SPIR-V words, asm).
    fn emit(&self, program: &Program) -> Result<String>;

    /// Whether this backend can run the program at all.
    fn supports(&self, program: &Program) -> bool {
        program
            .buffers
            .iter()
            .all(|b| matches!(b.scalar, Scalar::F32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elementwise_program_is_well_formed() {
        let p = Program::elementwise_binary("ivy_add_f32", Scalar::F32, BinaryOp::Add);
        assert_eq!(p.buffers.len(), 3);
        assert_eq!(p.output_buffers().count(), 1);
        assert_eq!(p.workgroup_size(), 256);
        assert_eq!(p.body.len(), 5);
    }
}
