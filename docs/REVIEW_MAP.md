# Tuner review map

This map covers the complete Stage 1–4 tuner implementation.

## Tier 1 — semantically rich / critical

- `crates/seam/src/database.rs` — constructs the one-connection,
  `REPEATABLE READ` sample transaction, quotes identifiers, binds pgvector
  values, orders exact results by distance/key, deliberately omits the ANN
  key tie-break, applies `SET LOCAL`, owns commit/rollback sequencing, and
  classifies bounded client operations as durable or connection-level.
- `crates/seam/src/measure.rs` — parses `.vseam` records through
  vectorseam-core, performs first-occurrence exact-byte deduplication, hashes
  raw payloads, distinguishes durable sample failures, retryable connection
  loss, cancellation, and the table-smaller cohort abort, and builds durable
  rows.
- `crates/seam/src/pacer.rs` — enforces cooldown between whole sample
  transactions from each transaction's observed wall time, never sleeping
  inside an open transaction (owner decision, 2026-07-17); an error path must
  consume the same budget as success, and only the pre-transaction cooldown
  is cancellation-aware.
- `crates/seam/src/pipeline.rs` — owns window-scoped listing, source/header
  membership, incomplete/malformed/config-mismatched pair handling,
  truth-before-sweep durability, cancellation boundaries, Phase A abort
  precedence, and history-before-latest publication.
- `crates/seam/src/math.rs` — owns recall set semantics, exact FNV-1a and split
  membership, type-7 interpolation, ef selection, and beta-posterior
  confidence.
- `crates/seam/src/accounting.rs` — owns half-open rolling-window membership,
  listed-part accounting, coverage, and collector drop fractions.
- `crates/seam/src/population.rs` — owns cross-part survivor ordering,
  population summaries, train selection, holdout transfer, and confidence
  inputs.
- `crates/seam/src/aggregate.rs` — composes compatibility filtering,
  deterministic splitting, counters, typed Phase A abort precedence, and
  ordered round records from the focused Tier 1 primitives.
- `crates/seam/src/intermediate.rs` — validates the frozen parquet schemas and
  pair metadata, cross-checks `measured_count`, encodes zstd pairs, and joins
  authoritative stored sweep observations without reimplementing recall.
- `crates/seam/tests/acceptance_b_estimator.rs` and
  `crates/seam/tests/estimator_properties.rs` — literal acceptance assertions
  and estimator invariants.
- `python/seam_harness/anchor.py` — independent FNV split adapter that reuses
  the trusted anchor modules for all benchmark math.
- `crates/seam/tests/acceptance_a_anchor.rs` and
  `crates/seam/tests/support/anchor.rs` — run the real tuner once against the
  ordered F-pg segment, read its durable intermediate and round output, and
  compare A1–A5 against the trusted anchor artifact without duplicating
  anchor math.
- `crates/seam/tests/support/f_agg.rs` and
  `crates/seam/examples/f_pg_fixture.rs` — frozen-schema and seeded PostgreSQL
  fixtures, including query ordering, clean-run artifact reset, no ties,
  boundary gap, HNSW, and B2.

## Tier 2 — standard

- `crates/seam/src/tuner.rs` — sequential multi-cohort orchestration, one
  long-lived runtime per data source, once-per-round reconnect, connection
  abandonment, per-cohort failure isolation, and bounded connection-driver
  shutdown.
- `crates/seam/src/daemon.rs` — immediate/periodic single-flight scheduling,
  skipped-tick accounting, wall-clock projection, signal ownership, and
  ordered tuner shutdown.
- `crates/seam/src/model.rs` — pure aggregation contracts and ordered round
  JSON types.
- `crates/seam/tests/acceptance_c_durability.rs` — cached database-down,
  config, empty-round, abort, and reproducibility acceptance.
- `crates/seam/tests/f_agg_builders.rs` — schema, compression, metadata,
  crash-shape, ULID-order, and segment round-trip checks.
- `tests/seam/docker-compose.yml` — isolated pgvector service.
- `crates/seam/benches/phase_b.rs` and
  `docs/phase-b-benchmark-baseline.md` — fixed hot-path workloads and the
  Stage 3 reference measurement.

## Tier 3 — glue

- `crates/seam/src/config.rs` — YAML parsing, defaults, references,
  data-source uniqueness, conditional password-environment validation,
  duration/grid/window checks, and identifier validation.
- `crates/seam/src/main.rs` — `--config`/`SEAM_CONFIG`, logging setup,
  one-time config loading, and Tokio runtime entry.
- `crates/seam/src/lib.rs` — module wiring.
- `crates/seam/tests/support/mod.rs` — test support wiring.
- `Makefile` and `.github/workflows/ci.yml` — database-free and Docker gate
  entry points.
- `crates/seam/Cargo.toml` and workspace `Cargo.toml` — dependency wiring.
- `agents.md` — binding project and Tokio guidance.

## Confidence report

I am confident in the pure estimator primitives and their composition: all
B criteria are now machine-gated, including Phase A deduplication, PostgreSQL
tie handling, no-double-count resume behavior, and cross-window survivor
movement. Phase B remains synchronous and clock-free; aligned time and
`computed_at` are explicit inputs.

I am confident in the Stage 3 durability path. C1 uses a recording object
store to prove truth precedes sweep and round history precedes latest, repairs
an interrupted pair, measures only that part, and compares bytes with a clean
run. Source storage/parse errors abort without publication under the
owner-approved resolution; malformed intermediate pairs are remeasured.
Noncanonical `.vseam` names are skipped with a warning, while malformed
canonical parts remain fail-visible. Connection loss discards the in-progress
part and round publication, leaving it naturally retryable. Cancellation can
interrupt the cooldown before a transaction; an in-flight transaction is
never dropped, and a cancelled partial part and round are not published.

I am confident in the database resource path. One `DatabaseConnection` owns
one Tokio client and its supervised driver task; mutable single ownership and
sequential cohort orchestration serialize all statements for the data source,
while the pacer charges successes and failures. Every connect/protocol
operation has the shared client deadline; connection-level failure abandons
the driver and reconnects once at the next round start. F-pg tests prove B2,
C2 recovery, C6, and D3; D3 holds a dedicated fixture table lock so the
frozen 1 ms timeout is deterministic, and its transaction counter proves one
attempt per sample with no retries. D1 uses paused Tokio time and the frozen
50-unit bounds. Shutdown drops clients, observes both task-result layers, and
has a ten-second overall deadline whose forced fallback aborts remaining
driver handles.

I am confident in the Stage 4 anchor reproduction. The fixture is reset before
the run, Python computes its side through the existing
`ground_truth.py`/`sweep.py`/`analyze.py` functions, and the five Rust tests
measure through `Tuner`, read the durable parquet pair, and assert the frozen
A tolerances. The real Docker run passed all five criteria. The daemon keeps
one round future in the current task, advances past crossed tick boundaries
without queuing work, and uses an owned signal task to cancel cooldown or
remaining cohort work before shutting down the database drivers.
Database-backed tests are explicitly ignored outside their Docker targets,
and assert `SEAM_REQUIRE_F_PG` when forced to run. The anchor target deletes
the old comparison before generation and requires a non-empty replacement,
so a missing environment flag or stale comparison is fail-visible.

The highest-value human review remains `database.rs`, followed by the compact
`measure.rs`/`pipeline.rs` durability chain. The owner completed the C7 manual
review against that path on 2026-07-17.

Deliberate MVP corner cut: there is no connection pool or concurrent
statement execution. Each validated unique `(server, database)` pair owns
exactly one connection and pacer, and cohorts run sequentially. The owner
approved this simplification when removing D2 and `max_concurrent_queries`.

Changes from the first Stage 3 approach:

- Storage failure policy was initially underspecified. The owner chose:
  storage/source failures abort that cohort without publishing, while a
  malformed intermediate pair is remeasured; the spec now states this.
- Replacing an incompatible pair would have erased evidence of the mismatch
  before aggregation. `AggregationInput` now carries the Phase A count so the
  published round retains `incompatible_parts = 1` after successful
  remeasurement.
- C1 initially verified only final objects. It now records PUT order and
  asserts truth-before-sweep plus history-before-latest.
- Connection failure was initially represented as an ordinary per-sample
  error, which would have durably poisoned a part. It now has a typed path:
  discard the in-progress part, publish no cohort round, abandon the
  connection, and reconnect once next round. Server-confirmed errors on a
  usable connection, including D3 timeouts, remain durable.
- The source-header pass currently downloads and fully parses every in-scope
  segment and pending parts are read twice. This is now an explicit MVP
  corner cut; range-GET header parsing is the first recommended improvement,
  followed by an in-memory ULID-keyed header cache.
- Intermediate row-count conversions previously reused the unrelated
  `DuplicateCount` error. They now report dedicated `ResultCount` and
  `MeasuredCount` failures.
- Large outcome/runtime enum variants were boxed after clippy identified
  avoidable stack-size inflation; behavior and public JSON were unchanged.
- Local F-pg execution exposed a Colima host-forwarding issue. An explicit
  temporary SSH forward was used only to validate locally; the Docker CI
  target itself remains unchanged and all B2/C6/D3 assertions passed.
- A warm local PostgreSQL could occasionally complete D3's exact scan within
  1 ms. The F-pg generator now creates a dedicated timeout table and D3 locks
  it during the round, making the spec's every-sample timeout deterministic
  without changing the production query or tolerance.

Stage 4 deviations from the initial skeleton:

- The A-suite skeleton originally loaded only the Python artifact and left
  five ignored placeholders. It now uses one `OnceLock`-owned real tuner run
  shared by the five criterion tests; this avoids five competing measurement
  passes while preserving a separately named assertion for every A criterion.
- The daemon does not await a plain `tokio::time::Interval` after each round.
  An overdue interval tick would complete immediately and turn skipped work
  into a queued catch-up round. The implemented monotonic deadline calculation
  advances over every crossed boundary and schedules only the next future
  tick.

No unresolved A–D specification question remains. The owner approved the
stronger A4 fixture target `value: 0.8`; the anchor and tuner must now select
the literal mid-grid ef `80`. Optional P1 `seam plan` was not implemented in
Stage 4 and remains a product-priority call; it is not an A–D acceptance
criterion.

## C7 deferred manual review — Gate 3

Deferral approved by the owner on 2026-07-16. No pausing proxy and no C7-lite
instrumentation test are required. For Gate 3 approval, a human reviewed
`crates/seam/src/database.rs` and confirms:

1. One `DatabaseConnection.client` supplies every statement for one sample.
2. `build_transaction().isolation_level(RepeatableRead).start()` is the first
   sample statement.
3. Transaction-scoped settings use `SET LOCAL`, never session-level `SET`.
4. The ground-truth query completes before ascending ef sweep statements.
5. The borrowed transaction is neither committed, rolled back, nor
   reacquired between ground truth and the final sweep; commit occurs only
   after successful completion. Server-confirmed errors on a usable
   connection explicitly roll back; a connection-level/client-timeout error
   abandons the connection and never reuses uncertain protocol state.

Gate 3 sign-off record: owner, 2026-07-17, approved against the Stage 3
working tree before Stage 4 began.
