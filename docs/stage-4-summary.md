# Stage 4 summary

## Passed

- A1–A5 run end-to-end against the deterministic F-pg fixture. The Python
  driver reuses the existing anchor modules, while the Rust side runs the real
  tuner over the ordered `.vseam` segment and compares the durable parquet
  observations and published round.
- The fixture generator removes prior tuner/anchor artifacts before each run,
  so cached intermediates cannot turn the anchor gate into a read-only
  comparison.
- Ordinary `cargo test` reports every F-pg/A test as ignored. The Docker
  targets explicitly run those ignored tests, and each test asserts the
  harness environment instead of silently returning. The anchor recipe
  deletes and then requires a fresh non-empty `comparison.json`.
- The `seam` binary accepts `--config` or `SEAM_CONFIG`, loads and validates
  YAML once, starts an immediate round, then schedules serialized periodic
  rounds. Tick boundaries crossed by a running round are skipped rather than
  queued.
- SIGINT/SIGTERM cancellation interrupts an inter-transaction cooldown or
  stops before remaining cohort work. An in-flight transaction retains the
  Stage 3 finish-and-observe behavior before owned database drivers shut down.
- CI keeps F-agg database-free on every push and runs F-pg plus A1–A5 in the
  Docker harness job.
- The Rust 1.85 gate now runs through `make test-rust-msrv`, which pins both
  Cargo and rustc to the MSRV, cleans an isolated target directory, checks
  every locked Rust target, and runs the locked workspace tests.
- `serde-saphyr` is pinned to `0.0.11`, the newest release verified to compile
  on Rust 1.85; newer releases use language and library features beyond the
  workspace MSRV.

## Red or deferred

- C7 remains the owner-approved manual-review exception; its required review
  was completed and approved on 2026-07-17.
- D2 remains the owner-approved removed concurrency criterion.
- Optional P1 `seam plan` was not implemented. It is outside A1–D3 and is
  flagged for the owner's product-priority call.
- The owner-approved A4 strengthening uses `value: 0.8`; both implementations
  must select the literal mid-grid ef `80`.

## Open questions

No A1–D3 specification question is open. The only remaining priority decision
is whether P1 plan mode should be scheduled as follow-up work.
