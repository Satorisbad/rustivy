# RustIVY

A from-scratch Rust reimplementation of PyTorch — tensors, autograd, kernels, scheduler,
collectives, packaging — with a torch-shaped API so ports are mechanical.

Status: **burst #1 (foundation)**. Nothing trains yet. See `docs/plan.md` for the full design of
record and `docs/interfaces/` for the frozen interfaces.

## What it claims

- **Torch-shaped** — `Tensor`, `nn`, `optim` under torch's names, so `torch.foo` → `ivy::foo`.
- **Own kernels** — one kernel IR, emitted to PTX, CPU SIMD, MSL and SPIR-V. Tiling, fusion and
  autotuning are ours, not a vendor library's.
- **Agent-operable** — every error carries a stable numeric code and a hint; `ivy` prints JSON.
- **Training-first** — throughput, stability, resumability and instrumentation are design
  constraints, not later features.
- **Small** — one static binary. No libpython, no `abi3` problem, no 2.3 GB wheel.
- **Framework-compatible** — safetensors, GGUF, ONNX and HF `config.json` load without Python.

It does **not** claim to be faster than PyTorch yet. That is measured (`Gate K` in the plan) and
reported, never assumed.

## Quick start

```sh
cargo test --workspace          # 41 tests, no GPU, no network
cargo run -p ivy-cli -- kernel --name ivy_add_f32 --sm 75 --dump    # PTX out of the IR
cargo run -p ivy-cli -- coverage                                    # op coverage as JSON
```

## Crates

| Crate | Role |
|---|---|
| `ivy-core` | dtype, device, shape, layout, error codes |
| `ivy-tensor` | storage, views, dynamic `Tensor`, typed `Tensor<N>` |
| `ivy-kernel` | kernel IR, tiling, autotune, reference interpreter |
| `ivy-cuda` / `ivy-cpu` / `ivy-metal` / `ivy-spirv` | backends |
| `ivy-autograd` | tape, vjp/jvp registry |
| `ivy-nn` / `ivy-optim` | layers, LLaMA blocks, fused AdamW, ZeRO |
| `ivy-dist` | own collectives, 3D parallel plan |
| `ivy-compile` | graph IR, shape buckets, fusion cache |
| `ivy-io` | safetensors, GGUF, run state |
| `ivy-parity` | PyTorch oracle (dev only) |
| `ivy-cli` | `ivy` binary: coverage, kernel, bench, run, checkpoint |
| `rustivy` | umbrella |

Dependencies point **downward only** in that list. Enforced in CI.

## Working on it

Read `AGENTS.md` first — it has the crate-ownership rules, the dependency policy and the machine
limits. Short version: one crate per agent, interfaces frozen before implementation, `va...` no
ML crates, and no heavy builds on the operator's Mac.

## Licence

MIT OR Apache-2.0.
