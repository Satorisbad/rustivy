# Parked: GitHub Actions workflow

`ci.yml` cannot be pushed by the current `gh` token — it has `repo`, `gist` and `read:org`, but not
`workflow`, and GitHub refuses any commit that creates `.github/workflows/*` without it.

To activate:

```sh
gh auth refresh -h github.com -s workflow
git mv docs/github/ci.yml .github/workflows/ci.yml
git commit -m "ci: enable the workflow now that the token has scope" && git push
```

Everything the workflow runs is reproducible locally:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
./scripts/check-deps.sh
cargo deny check bans
```
