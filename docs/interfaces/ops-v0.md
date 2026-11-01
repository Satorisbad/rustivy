# W1 — the ops a LLaMA block executes

This is the first kernel tranche. Everything else in the 1,904-op program is ordered *behind* it
(docs/plan.md §6) because until these exist there is no loss curve, and without a loss curve there
is no way to know whether any of the rest is right.

**68 ops.** Columns: `op` — what it does; `grad` — needs a vjp for the 1B milestone; `notes` — the
part that is easy to get wrong.

## Elementwise (18)

| op | grad | notes |
|---|---|---|
| `add`, `sub`, `mul`, `div` | yes | broadcasting must be explicit, never silent; `div` is `div.rn` (round-to-nearest) |
| `neg`, `abs` | yes | `abs` gradient at 0 is 0, matching torch |
| `exp`, `log`, `sqrt`, `rsqrt` | yes | use `__expf`-class approximations only where tolerance allows; parity is 1e-4 |
| `pow_scalar` | yes | integer exponents take the multiply-chain path |
| `maximum`, `minimum` | yes | NaN propagation follows IEEE, same as torch |
| `where_` (select) | yes | mask dtype `Bool`; gradient routes to both branches |
| `clamp` | yes | min/max bounds as scalars; gradient zero outside |
| `reciprocal` | yes | |
| `tanh`, `sigmoid` | yes | sigmoid reuses the tanh kernel where the backend supports it |

## Activations (3)

| op | grad | notes |
|---|---|---|
| `silu` | yes | `x * sigmoid(x)`; the fused form is one kernel, not three |
| `gelu_tanh` | yes | tanh approximation, exactly torch's `gelu(approximate='tanh')` |
| `relu` | yes | |

## Normalisation and reductions (10)

| op | grad | notes |
|---|---|---|
| `rms_norm` | yes | fp32 accumulation even for bf16 input; `eps` inside the rsqrt |
| `layer_norm` | yes | biased variance (`unbiased=False`), as torch's nn.LayerNorm |
| `softmax_lastdim` | yes | max-subtraction for stability; fused with the downstream matmul where possible |
| `log_softmax` | yes | must be numerically stable, not `log(softmax(x))` |
| `sum` | yes | axis + keepdim; fp32 accumulator for bf16 |
| `mean` | yes | |
| `var` | yes | biased by default |
| `max` | yes | values only in W1; indices in W2 |
| `argmax` | no | first-index tie-break, matching torch |
| `cumsum` | yes | exclusive/inclusive variants |

## Matmul (5)

| op | grad | notes |
|---|---|---|
| `mm` | yes | tiled, autotuned; Gate K benchmarks this against cuBLAS |
| `bmm` | yes | batched; batch dims never broadcast implicitly beyond torch's rules |
| `addmm` | yes | fused bias add; this is the training hot path |
| `matmul` | yes | full broadcasting semantics, lowered to `mm`/`bmm` |
| `linear` | yes | `x @ Wᵀ + b`, lowering to `addmm` |

## Attention (4)

| op | grad | notes |
|---|---|---|
| `sdpa_forward` | — | GQA: `n_heads != n_kv_heads`, no materialised `[B,H,S,S]` for the flash path |
| `sdpa_backward` | yes | recompute-based; this is where memory goes if it is wrong |
| `tril` / `causal_mask` | no | folded into the attention kernel, exposed separately for tests |
| `masked_fill` | yes | `-inf` fill; nan-safe under fp16 |

## Shape and memory (10)

`reshape`, `transpose`, `permute`, `contiguous`, `cat`, `stack`, `split`, `squeeze`, `unsqueeze`,
`narrow` — all **zero-copy views** except `contiguous` and `cat`, which copy and say so.

## Indexing (5)

| op | grad | notes |
|---|---|---|
| `embedding` | — | gather by index |
| `embedding_backward` | yes | `scatter_add` with duplicate indices; atomic on CUDA |
| `index_select` | yes | |
| `gather` | yes | |
| `scatter_add` | yes | determinism: CPU deterministic, CUDA opt-in via atomics ordering |

## Cast, loss, random (6)

`cast` (`to(dtype)`), `copy_`, `cross_entropy` (fused log-softmax + NLL), `nll_loss`,
`dropout` (Philox, matching torch's stream), `init_uniform` / `init_normal` (Philox).

## Optimiser (3)

`adamw_fused` (weight + both moments in one pass), `clip_grad_norm` (global norm, one reduction),
`zero_grad` (memset, not a kernel).

## Positional (3)

`rope_apply` (fused rotary, interleaved and half-split layouts), `sin`, `cos`.

---

## What "done" means for an op

1. Forward matches torch to `1e-4` relative (f32) on random and edge shapes.
2. Backward, where `grad` is yes, matches torch *and* passes a finite-difference check in fp64.
3. Works on at least: f32 (CPU, reference), f32 (PTX), bf16 (PTX).
4. Registers in `ivy-cli coverage` as implemented, so the dashboard never lies.

## What is deliberately absent

`conv2d`, pooling, `topk`, `sort`, sparse, `fft`, quantised formats, complex. They are W2/W3/W4 and
none of them appear in a LLaMA forward or backward pass.
