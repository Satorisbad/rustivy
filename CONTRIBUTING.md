# Contributing

Read `AGENTS.md` before opening anything. It is written for agents and applies to humans too.

## The short version

1. Pick an issue. Each one names the crate it touches.
2. Branch from `main`: `<crate>/<short-slug>`.
3. Implement against the frozen interface in `docs/interfaces/`.
4. `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`.
5. PR with `Closes #N` and the `test result:` lines pasted in.

## Interface changes

If a frozen interface is wrong, stop and say so in the PR. Do not silently redesign it — several
agents share a workspace, and the interfaces are the only thing keeping them from colliding.

## What we will not merge

- A dependency on an existing ML framework (`burn`, `candle`, `tch`, ...). `cargo deny` blocks it.
- An implicit copy or implicit device transfer. Those must be explicit calls.
- An `unwrap()` on anything reachable from a shape, a file or a device call.
- A parity test that is loosened instead of fixed.
