#!/usr/bin/env bash
# Enforce the dependency direction: a crate may only depend on crates below it in
# docs/plan.md §5. Cheap text check, no cargo metadata parsing.
set -euo pipefail

# Rank: lower may be depended on by anything above it.
rank() {
  case "$1" in
    ivy-core) echo 0 ;;
    ivy-tensor) echo 1 ;;
    ivy-kernel) echo 2 ;;
    ivy-cuda|ivy-cpu|ivy-metal|ivy-spirv) echo 3 ;;
    ivy-autograd) echo 4 ;;
    ivy-nn|ivy-optim) echo 5 ;;
    ivy-dist) echo 6 ;;
    ivy-compile) echo 7 ;;
    ivy-io) echo 8 ;;
    ivy-parity|ivy-cli) echo 9 ;;
    rustivy) echo 10 ;;
    *) echo -1 ;;
  esac
}

fail=0
for manifest in crates/*/Cargo.toml rustivy/Cargo.toml; do
  pkg=$(basename "$(dirname "$manifest")")
  mine=$(rank "$pkg")
  [ "$pkg" = "rustivy" ] && mine=10
  # every dependency line naming an ivy crate
  for dep in $(grep -oE '^ivy-[a-z]+' "$manifest" | sort -u); do
    theirs=$(rank "$dep")
    if [ "$theirs" -ge "$mine" ] && [ "$mine" -ne 10 ]; then
      echo "DEPENDENCY VIOLATION: $pkg (rank $mine) depends on $dep (rank $theirs)"
      fail=1
    fi
  done
done

[ "$fail" -eq 0 ] && echo "dependency direction OK"
exit "$fail"
