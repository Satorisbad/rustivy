# Frozen interfaces — burst #1

The scaffold in `crates/` is the interface. This document says which parts of it are **frozen**
(agents may not change without an operator-approved ticket) and what each crate is allowed to grow
into.

Rule: if a signature here is wrong, the agent stops and says so. An interface changed silently
costs a week of merges.

## `ivy-core` — frozen

```rust
pub enum DType { Bool, U8, U32, I8, I16, I32, I64, F16, Bf16, F32, F64, F8E4M3, F8E5M2 }
impl DType { size_bytes, name, is_float, is_signed, is_int, accumulator, promote }

pub enum Device { Cpu, Cuda { index: u8 }, Metal, Vulkan, Tpu { index: u8 } }
impl Device { parse(&str) -> Result<Device>, is_accelerator, index }

pub struct Shape(Box<[usize]>);   // numel, rank, dims, contiguous_strides, dim(axis)
pub struct Layout { shape, strides, offset }  // contiguous, new, is_contiguous, span

pub struct Error;                 // new(kind, msg), with_hint, kind(), code(), to_json()
pub enum ErrorKind { /* 1xxx shape, 2xxx dtype, 3xxx device, 4xxx kernel/compile,
                       5xxx io, 6xxx dist, 7xxx autograd, 8xxx parity, 9xxx internal */ }
pub type Result<T> = core::result::Result<T, Error>;
```

Non-negotiable:
- `ErrorKind::code()` values never change and are never reused.
- `promote` follows torch's tensor rules. Python-scalar special cases live at the op layer.
- `Shape::contiguous_strides` gives size-1 dims the running stride, matching torch's `empty_strided`.

May grow: more `ErrorKind` variants (new numbers only), `Shape` helpers.

## `ivy-tensor` — frozen surface, open implementation

```rust
pub struct Tensor;                       // zeros, from_f32, shape, strides, dtype, device,
                                         // numel, rank, is_contiguous, as_f32, reshape
pub struct TypedTensor<const N: usize, T: Elem>;  // zeros, shape, dtype, as_dynamic, from_dynamic
pub struct Storage;                      // zeros, from_bytes, dtype, device, bytes, as_f32
```

Non-negotiable:
- `reshape` on a non-contiguous tensor is an **error**, never an implicit copy.
- `as_f32` on a non-f32 tensor is an error with `ErrorKind::DTypeMismatch`.
- `TypedTensor::from_dynamic` checks rank and dtype and returns `E1001`/`E2001`.

Open: device storage (burst #2), views beyond reshape, `contiguous()`.

## `ivy-kernel` — frozen IR, open op set

```rust
pub enum Target { Ptx { sm: u16 }, CpuSimd, Msl, Spirv }
pub enum Scalar { F32, F16, Bf16, I32, U32, Bool }
pub struct Program { name, buffers, workgroup, body, registers }
pub enum Stmt { GlobalIndex, Load, Bin, Un, Store }
pub trait Emitter { target(); emit(&Program) -> Result<String>; supports(&Program) -> bool }
pub fn reference::run_f32(&Program, n, &mut [&mut [f32]]) -> Result<()>
```

Non-negotiable:
- `reference::run_f32` is the oracle for every backend. A backend that disagrees with it is wrong,
  even if it agrees with torch.
- Every `Emitter` must return `E4001 KernelUnsupported` rather than emitting garbage.

## `ivy-cuda` / `ivy-cpu` / `ivy-metal` / `ivy-spirv` — open

`PtxEmitter` and `MslEmitter` emit text for f32 elementwise binary programs in burst #1.
`ivy-cpu::execute_f32` routes to the reference interpreter until SIMD codegen lands.

## `ivy-autograd`, `ivy-nn`, `ivy-optim`, `ivy-dist`, `ivy-compile` — skeleton

Types exist so the workspace has its final shape. `ivy-dist::Plan` and
`ivy-compile::ShapeBucket`/`Cache` are **implemented and tested** because they encode decisions
(world-size validation, shape-key bucketing) that later code must not re-litigate.

## `ivy-io` — frozen surface

```rust
pub fn detect(path) -> Result<Format>              // Safetensors | Gguf | Onnx | Unknown
safetensors::read_header(path) -> Result<Header>   // names + shapes, no tensor bytes yet
gguf::read_meta(path) -> Result<GgufMeta>
run_state::RunState { global_step, tokens_seen, rng, data_position, lr_schedule, plan,
                      world_size, sha256_manifest }  // write/read/can_resume_onto
```

Non-negotiable: a checkpoint is a **directory** (`model.safetensors` + `run_state.json`), and
`can_resume_onto` refuses a world size whose product differs from the recorded one (`E6002`).

## `ivy-parity` — frozen surface

```rust
pub const Tolerance::F32 / BF16 / F16
pub fn compare(op, shape, dtype, tol, ours, theirs) -> Result<Case>
pub struct Oracle { eval_f32(expr) -> Result<Vec<f32>> }
```

Non-negotiable: a parity failure is `E8001`, never a warning.

## `ivy-cli` — frozen contract

One JSON object on stdout per invocation; errors go to stderr as `{"code":"Exxxx",...}` with exit
code 1. Subcommands: `coverage`, `kernel [--dump]`, `bench`, `run`, `checkpoint`.
