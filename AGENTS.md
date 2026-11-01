# AGENTS.md — rules for agents working in this repo

Read this before touching anything. These rules exist because the work is parallel: several
agents, one workspace, and no time for merge archaeology.

## 1. One crate, one agent

- You own **exactly one crate**. Do not edit files outside it, except the ticket that assigns it
  to you.
- If you believe another crate is wrong, open an issue against that crate. Do not "fix it while
  you're there".
- Shared files — `Cargo.toml` (workspace), `.github/`, `AGENTS.md`, `docs/plan.md` — are
  **operator-only**. Propose changes in a PR description instead of editing them.

## 2. Interfaces are frozen before implementation

Every crate has a spec in `docs/interfaces/<crate>.md`. Implement against the spec, not against
your own idea of what the API should be.

- Implement the spec as written.
- If the spec is wrong or under-specified, say so in the PR and stop; don't invent a different
  interface and carry on. A wrong frozen interface costs a day; a silently diverged one costs a
  week of merges.
- Every public type must be usable across the pyo3 boundary later: no lifetimes in public
  signatures, `Send + Sync` where the type is shareable, errors as `ivy_core::Error` with a
  `ErrorKind`, never panics on malformed input.

## 3. Dependency direction

Dependencies point downward only:

```
ivy-core → ivy-tensor → {ivy-kernel → backends} → ivy-autograd → {ivy-nn, ivy-optim} → ivy-dist → ivy-compile
                                                                                      ivy-io → ivy-cli
```

Two hard rules, enforced by `cargo deny` in CI:

1. **No ML framework dependencies.** Not `burn`, not `candle`, not `tch`, not `luminal`. Driver
   bindings only: `cudarc`, `metal-rs`, `rayon`, `serde`, `memmap2`, `clap`. If you need a crate
   that is not on that list, ask first — the answer is usually "write it".
2. **No new workspace dependency from a leaf crate without a ticket.**

## 4. Machine limits — the operator's Mac

- The Mac **compiles and unit-tests**. It never runs a model. Not a small one, not to "check".
- Use `cargo check -p <your-crate>` while iterating. `cargo build --release` and full-workspace
  test runs only at the end of a task.
- `CARGO_TARGET_DIR` is shared at `~/.cache/rustivy/target`. Don't override it per worktree; disk
  is tight and duplicate target dirs are how you fill an SSD.
- If the machine is hot or a build takes more than a few minutes, stop and report. Do not retry
  in a loop.

## 5. Definition of done (a crate is done when)

1. Its parity suite passes against the torch oracle (`ivy-parity`), or it has an explicit
   `#[ignore]`d test and a note saying why.
2. `docs/interfaces/<crate>.md` matches the code. If you changed behaviour, you changed the spec
   in the same PR.
3. `cargo fmt --all` and `cargo clippy --workspace --all-targets -- -D warnings` are clean.
4. No other crate's tests broke.

## 6. Tickets and PRs

- Every PR references one issue: `Closes #12`.
- Trunk-based: branch from `main`, one branch per crate per task, PR back to `main`.
- PR body: what changed, what the tests show (paste the `test result:` lines), and what you could
  not do.
- Never commit to `main` directly. Never force-push a branch someone else may have based on.

## 7. Errors are a product

Every user-facing failure goes through `ivy_core::Error` with a `ErrorKind` whose `code()` is
stable. Message says what happened; `hint()` says what to do. No `unwrap()` in library code on
anything reachable from a tensor shape, a file, or a device call.

## 8. Performance work

- Never guess. `ivy-cli bench` writes JSON; a change that is not measured is not an improvement.
- Gate K in `docs/plan.md` is the standing benchmark: matmul, attention and a fused RMSNorm+SwiGLU
  against cuBLAS/FlashAttention. Report TFLOPS and % of peak, not feelings.
