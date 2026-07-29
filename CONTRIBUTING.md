# Contributing

VectorSeam is an early-stage project; issues and pull requests are welcome.

## Prerequisites

- Rust (stable toolchain; the MSRV is 1.85 and is checked in CI)
- [uv](https://docs.astral.sh/uv/) — used for everything Python; never raw
  `pip`/`python` calls
- Docker — for the database-backed test harness and the demo

## Building and testing

```sh
make test          # Rust workspace tests + Python unit tests, no database
make lint-rust     # rustfmt check + clippy with warnings denied
make doc-rust      # rustdoc with warnings denied
```

The database-backed tuner suites run against a Dockerized pgvector fixture:

```sh
make seam-f-pg-harness   # acceptance tests + Python anchor comparison
```

CI runs all of the above on every push and pull request, plus a clean-room
MSRV build (`make test-rust-msrv`). If you cannot run Docker locally, rely on
the CI harness job and say so in the PR.

## Style

- [AGENTS.md](AGENTS.md) is the binding coding guidance for this repository:
  Google's Python style guide, least-visibility Rust, and the Tokio rules for
  async code.
- Remove or wire up obsolete declarations, counters, configuration, and
  documentation in the same change; do not leave dead code behind.
- For new Python SDK hot-path functionality, consider a matching benchmark —
  see [benchmarks/README.md](benchmarks/README.md).

## Pull requests

- Branch from `master` and keep the change focused.
- Run `make test` and `make lint-rust` before opening the PR.
- Component behavior is specified in [docs/](docs/); if your change alters
  specified behavior, update the spec in the same PR.
- Benchmark result files under `python/ann-recall-latency/results/` are
  checked in deliberately so external writeups can reference stable
  artifacts; regenerate them only when that is the point of the change.

## Security

Do not report vulnerabilities in public issues — see
[SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contributions are licensed under the
Apache License 2.0 ([LICENSE](LICENSE)).
