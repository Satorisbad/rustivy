# RustIVY — plan of record

Working name: **RustIVY** (iron II + corrosion). Crates use the `ivy-*` prefix, umbrella crate `rustivy`.
Repo: `github.com/rustivy/rustivy`, local at `~/WorkBuddy AI/rustivy/`.
Status: design settled 2026-11-01. Compile architecture deliberately open (see Open Decisions).

---

## 1. Thesis

PyTorch is ~1.4M lines of Python over 1.25M lines of C++/CUDA, and its weaknesses are
**structural**, not cosmetic: a 2.3 GB wheel per CPython version, ~2 µs of interpreter before
every op, silent broadcasting and silent device fallbacks, and a compiler (Dynamo) that exists
only because Python can't be traced safely.

RustIVY is a from-scratch reimplementation that owns the whole path: tensor, autograd, kernels,
scheduler, collectives and packaging. No ML framework dependency anywhere.

## 2. The claim

What it promises:

- **Torch-shaped** — `Tensor`, `nn`, `optim` with torch's names and semantics, so ports are mechanical.
- **Framework-compatible** — safetensors / GGUF / ONNX / HF `config.json` load without Python.
- **Agent-operable** — machine-readable error codes, one obvious way per op, a JSON CLI that drives runs.
- **Training-optimised** — throughput, stability, resumability and built-in metrics are design
  constraints, not features added later.
- **Own kernels** — a Rust kernel DSL → one source emitting PTX, CPU SIMD, Metal, SPIR-V (TPU later).
- **Small** — one static binary. No libpython, no ABI pinning, no `pickle`.

What it does **not** claim (yet): that it is faster than PyTorch at training. That is measured and
reported, never assumed. See Gate K.

## 3. Settled decisions

| # | Decision |
|---|---|
| 1 | Rust-native core with a torch-shaped API; Python shim (`pyo3`) is a later phase, designed for from day one |
| 2 | Consumers: the Operator, AI agents first, then humans; NyxAI joins after interfaces exist |
| 3 | Scope: core five **plus** compile, distributed, GGUF/ONNX interop, Python bindings |
| 4 | Backends: CPU, CUDA, Metal, SPIR-V; TPU designed for, built later |
| 5 | v0 proof: LLaMA-arch 1B demonstrates training (see §8) |
| 6 | Own the kernel DSL **and** the scheduler; one source, all backends |
| 7 | Own the collectives (not NCCL) |
| 8 | `Tensor` (dynamic, torch-compatible) **and** `Tensor<N>` (const-generic, typed) |
| 9 | Shipping surface is Rust; PyTorch lives only in the test path as the oracle |
| 10 | One cargo workspace, one crate per agent, interfaces frozen before any agent starts |
| 11 | Public repo, trunk-based, PR per agent, CI must be green |
| 12 | Parity: op-level locally on CPU torch, model-level on Colab |
| 13 | GPU order: sm_75 (T4) → sm_80/86 → sm_89 (L4) → sm_90 (H100); then CPU SIMD, Metal, SPIR-V, TPU |
| 14 | Checkpoints: safetensors tensors + typed run-state sidecar (optimiser, RNG, dataloader position) |
| 15 | Op coverage: all 1,904 kernel-bearing aten ops before v0 (see §6 for ordering) |
| 16 | Compile: graph IR type defined in burst #1; pass design deferred |
| 17 | Compute: free tiers only, no exceptions (Colab + Kaggle + CPU) |
| 18 | Dependencies: zero ML crates; driver-level only (`cudarc`, `metal-rs`, `rayon`, `serde`, `memmap2`) |
| 19 | Licence: MIT OR Apache-2.0 |
| 20 | Tracker: GitHub issues are the work queue; agents pull tickets, not chat |

## 4. What we are building against (inventory, 2026-11-01)

| Metric | Count |
|---|---|
| aten ops | 2,590 |
| pure compositions (free once the base exists) | 686 (26.5%) |
| ops needing a real kernel | 1,904 |
| ops with an autograd formula (`derivatives.yaml`) | 688 |
| dispatch keys | 141 (30 codegen-supported) |
| Inductor (scheduler + lowerings) | ~325k LOC |
| Dynamo (incl. 7,305 lines of guards) | ~155k LOC |
| distributed + DTensor + FSDP | ~153k LOC |
| kernel source files (.cu/.cpp/.metal) | 223 / 452 / 61 |

Two facts that shape everything downstream:

- **Only 688 of 2,590 ops have a derivative.** Backward coverage is a separate, smaller, and much
  harder-earned surface than forward coverage.
- **26.5% of the op surface is composition**, so the first ~100 real kernels buy far more than
  100 ops of surface.

## 5. Architecture

```
ivy-core      dtype, device, shape, error codes, layout traits        (no deps)
ivy-tensor    storage, strides, dynamic Tensor, typed Tensor<N>       (→ core)
ivy-kernel    kernel DSL: IR, tiling, fusion hooks, autotune          (→ core)
ivy-cuda      PTX emitter + driver bindings (cudarc)                  (→ kernel)
ivy-cpu       SIMD emitter (AVX2/AVX-512/NEON)                        (→ kernel)
ivy-metal     MSL emitter + metal-rs                                  (→ kernel)
ivy-spirv     SPIR-V emitter                                          (→ kernel)
ivy-autograd  tape, typed graph nodes, vjp/jvp registry               (→ tensor)
ivy-nn        Module-shaped layers, LLaMA-arch blocks                 (→ autograd)
ivy-optim     SGD/AdamW, fused optimiser kernels, ZeRO sharding       (→ nn)
ivy-dist      collectives (own ring/tree), 3D parallelism, typestate  (→ optim)
ivy-compile   graph IR, shape buckets, fusion, cache                  (→ autograd)
ivy-io        safetensors, GGUF, ONNX, config.json, run-state         (→ core)
ivy-parity    torch oracle harness, fixtures, tolerance policy        (dev-only)
ivy-cli       run control, metrics as JSON, resume                    (→ all)
```

Rule: a crate may only depend **downward** in that list. No cycles, enforced in CI.

The typed/dynamic duality: `Tensor<N>` carries shape in the type and dtype as a parameter, so
broadcast and device errors are compile errors. `Tensor` is dynamic and torch-compatible. Every
`nn` layer is written once, generic over both.

## 6. The op program (1,904 kernels, ordered)

Ordering is not alphabet: it is **transformer-first, dependency-sorted**, because it costs nothing
to order it this way and everything to order it another way.

1. **W1 — the 68 ops a LLaMA block executes**: elementwise set, `matmul`/`bmm`/`addmm`,
   `scaled_dot_product_attention`, `rms_norm`/`layer_norm`, `softmax`/`log_softmax`, `embedding`,
   `cross_entropy`/`nll_loss`, `cat`/`stack`/`split`/`reshape`/`transpose`/`contiguous`, casting,
   `index_select`/`gather`, `topk`/`sort`/`argmax`, `masked_fill`, dropout, optimiser ops.
2. **W2 — reductions and stencils**: `sum`/`mean`/`var` family, pooling, conv1d/2d/3d, attention variants.
3. **W3 — the long tail**, ordered by: (a) anything with a derivative, (b) anything a HuggingFace
   model touches when we import it, (c) alphabetical for the rest.
4. **W4 — sparse, complex, fft, quantised** — the 30%-of-surface, 2%-of-use tail. Last, explicitly.

Coverage is a generated dashboard (`cargo run -p ivy-cli -- coverage`) comparing implemented ops
against the 2,590-op list, split by kernel-needed vs composite, forward vs backward.

## 7. Parity

- **Op-level, local, every commit**: a tiny CPU-only torch venv. ~200 ops × dtypes {f32, f16, bf16},
  random shapes including fuzz. Relative tolerance 1e-4 (f32) / 1e-2 (f16, bf16). Any drift fails CI.
- **Backward-level**: finite-difference check (our side) *and* torch oracle comparison (both sides),
  double-precision, on every op that has a derivative.
- **Model-level, on Colab**: same seed, same data, both frameworks, loss curves overlaid. Divergence
  beyond 1e-3 after 200 steps is a failure, not a note.
- PyTorch appears only in `ivy-parity` and CI. It never links into shipped code.

## 8. Milestones as signals (not phases)

| Signal | Meaning |
|---|---|
| S1 | `cargo test` green: tensor core, PTX elementwise, parity harness comparing 10 ops to torch |
| S2 | 68-op LLaMA set implemented (docs/interfaces/ops-v0.md), forward + backward, parity-clean |
| S3 | 10M LLaMA-arch trains on free Colab; loss curve matches torch within tolerance |
| S4 | 124M trains; checkpoint resumes across a forced disconnect |
| S5 | **The 1B**: builds, fits via ZeRO-3 + CPU/NVMe offload, trains N steps with falling loss, survives a Colab disconnect and resumes mid-run |
| S6 | 1,904/1,904 ops covered; `ivy-cli coverage` reports 100% |
| S7 | 3D parallelism correct on CPU multi-process over loopback (TP=2, PP=2, DP=2) |
| S8 | GGUF import: an existing open 1B loads and runs |

S5 is the honest version of "we trained a 1B" on free hardware. A **full pretrain** of a 1B is
not claimed, not scheduled, and not possible without a card — it stays on the ladder above S8.

## 9. Gate K — the kernel bet, falsifiable

The audit's evidence says wall-clock is set by cuBLAS/CuTe and that a rewrite cannot win there.
You overrode it. So the bet gets an early measurement, not a later argument:

**After the DSL's third iteration (elementwise fused → tiled matmul → fused attention), benchmark
our kernels against cuBLAS / FlashAttention on: 4096×4096 f16 matmul, 8×2048×4096×128 attention,
and a fused RMSNorm+SwiGLU.** Record GEMM TFLOPS and % of peak.

- Within 1.5× of vendor on all three → the bet is live; continue as planned.
- Beyond 1.5× after the fourth iteration → re-scope to owning the *scheduler* and calling vendor
  kernels for the three hottest ops, keeping the DSL for everything else.

This is written down now so it can be checked later without an argument.

## 10. Distributed

Own collectives, not NCCL: ring and tree allreduce in `ivy-dist`, behind a `Collective` trait with
typed buffers. A rank desync should be a **compile error** via session typestate, not a 30-minute
watchdog hang. Targets: DP (allreduce), TP (allgather/reduce-scatter), PP (send/recv with schedule).

Testing while compute is free: **N CPU processes over loopback**, which exercises the same trait as
NVLink. Single-GPU gradient accumulation is the degenerate DP=1 case. No bandwidth claims until real
multi-GPU exists.

## 11. Training-first properties (design constraints)

1. **Throughput**: steps/sec is the primary metric; fused optimiser, activation checkpointing default-on,
   ZeRO-2 as the default sharding, no sync points inside the step loop.
2. **Stability**: fp32 master weights, bf16 compute, gradient clipping built in, NaN/Inf detection
   always on, deterministic reductions by default (`--nondeterministic` to opt out).
3. **Resumability**: checkpoint every N steps; the sidecar carries optimiser moments, RNG, dataloader
   position, LR schedule and shard layout, so a resume onto a *different* world size works.
4. **Instrumentation**: `ivy-cli` prints MFU, memory headroom, tokens/sec and loss every step as JSON.

## 12. Burst #1 — foundation (7 agents, 7 crates)

Interfaces frozen and PR'd **before** any agent writes implementation.

| Crate | Delivers in burst #1 |
|---|---|
| `ivy-core` | `DType`, `Device`, `Shape`/stride algebra, `Error` with stable numeric codes + `kind()`, layout traits |
| `ivy-tensor` | storage + views, dynamic `Tensor`, `Tensor<N>` skeleton, basic allocation, no ops |
| `ivy-kernel` | DSL IR, elementwise lowering, tiling skeleton, autotune scaffold, PTX emitter for elementwise only |
| `ivy-parity` | torch oracle harness, fixtures, tolerance policy, 10-op comparison working end to end |
| `ivy-io` | safetensors + GGUF read (memory-mapped), `config.json` parsing |
| `ivy-cli` | `coverage`, `bench`, `run`, `resume` subcommands; JSON output |
| repo | workspace, CI (fmt, clippy, `cargo deny` for zero-ML-deps, parity on Linux CPU), issue templates |

Explicitly **not** in burst #1: autograd, nn, optim, dist, compile passes. They need a tensor core
that has been exercised first.

## 13. Open decisions

| # | Open | Unblocked by |
|---|---|---|
| O1 | **Compile architecture**: eager-recorded graph + static shape buckets (recommended) vs graph-first static whole-program | A spike comparing both on one fused op, or Operator decision. Burst #1 only commits to the IR *type*, so this stays cheap. |
| O2 | Full 1B pretrain | Compute that doesn't exist yet. Not scheduled. |
| O3 | Exact GPU budget when free tiers run out | Never, per decision 17 — S5 is the ceiling of what's claimed. |

## 14. Risks

1. **The kernel bet loses** — mitigated by Gate K, written before the work starts.
2. **1,904 ops is a year of surface** — mitigated by transformer-first ordering (S2/S3 arrive long
   before S6) and the coverage dashboard, so progress is visible weekly.
3. **Free compute limits throughput of everything** — parity and CPU work run locally; GPU work is
   batched per session; every GPU test must be resumable.
4. **Interface churn under parallel agents** — mitigated by frozen interfaces, downward-only deps,
   and one crate per agent.
5. **Scope creep into serving/vLLM territory** — not a goal. Inference exists only as S8 proof.
