---
name: Crate ticket
about: One unit of work for one agent in one crate
labels: agent
---

## Crate
<!-- exactly one: ivy-core | ivy-tensor | ivy-kernel | ivy-cuda | ivy-cpu | ivy-metal | ivy-spirv |
     ivy-autograd | ivy-nn | ivy-optim | ivy-dist | ivy-compile | ivy-io | ivy-parity | ivy-cli -->

## Deliverable
<!-- what exists when this is done, in one or two sentences -->

## Frozen interface
<!-- link to the section of docs/interfaces/ that governs this, and quote the signatures that may
     not change -->

## Definition of done
- [ ] implemented against the frozen interface
- [ ] `cargo test -p <crate>` green
- [ ] parity against torch, or an `#[ignore]`d test plus a reason
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` clean
- [ ] no other crate's tests broken

## Blocked by
<!-- issue numbers, or "nothing" -->
