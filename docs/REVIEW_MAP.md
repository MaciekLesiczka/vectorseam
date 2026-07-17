# Tuner review map

This map covers the Stage 1 harness, Stage 2 estimator, and Stage 3
measurement/durability implementation.

## Tier 1 — semantically rich / critical

- `crates/seam/src/database.rs` — constructs the one-connection,
  `REPEATABLE READ` sample transaction, quotes identifiers, binds pgvector
  values, orders exact results by distance/key, deliberately omits the ANN
  key tie-break, applies `SET LOCAL`, and owns commit/rollback sequencing.
- `crates/seam/src/measure.rs` — parses `.vseam` records through
  vectorseam-core, performs first-occurrence exact-byte deduplication, hashes
  raw payloads, distinguishes semantic sample failures from the
  table-smaller cohort abort, and builds durable rows.
- `crates/seam/src/pacer.rs` — enforces cooldown from observed statement wall
  time after single-owner orchestration has serialized the statements; an
  error path must consume the same budget as success.
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
- `crates/seam/tests/support/f_agg.rs` and
  `crates/seam/examples/f_pg_fixture.rs` — frozen-schema and seeded PostgreSQL
  fixtures, including query ordering, no ties, boundary gap, HNSW, and B2.

## Tier 2 — standard

- `crates/seam/src/tuner.rs` — sequential multi-cohort orchestration, one
  long-lived runtime per data source, database-down degradation, per-cohort
  failure isolation, and bounded connection-driver shutdown.
- `crates/seam/src/model.rs` — pure aggregation contracts and ordered round
  JSON types.
- `crates/seam/tests/acceptance_c_durability.rs` — cached database-down,
  config, empty-round, abort, and reproducibility acceptance.
- `crates/seam/tests/f_agg_builders.rs` — schema, compression, metadata,
  crash-shape, ULID-order, and segment round-trip checks.
- `crates/seam/tests/acceptance_a_anchor.rs` and
  `crates/seam/tests/support/anchor.rs` — Stage 4 anchor comparison skeleton
  and artifact loader.
- `tests/seam/docker-compose.yml` — isolated pgvector service.
- `crates/seam/benches/phase_b.rs` and
  `docs/phase-b-benchmark-baseline.md` — fixed hot-path workloads and the
  Stage 3 reference measurement.

## Tier 3 — glue

- `crates/seam/src/config.rs` — YAML parsing, defaults, references,
  data-source uniqueness, conditional password-environment validation,
  duration/grid/window checks, and identifier validation.
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
Cancellation is checked between samples, never by dropping an in-flight
transaction, and a cancelled partial part and round are not published.

I am confident in the database resource path. One `DatabaseConnection` owns
one Tokio client and its supervised driver task; mutable single ownership and
sequential cohort orchestration serialize all statements for the data source,
while the pacer charges successes and failures. F-pg tests prove B2, C6, and
D3; D3 holds a dedicated fixture table lock so the frozen 1 ms timeout is
deterministic, and its transaction counter proves one attempt per sample with
no retries. D1 uses paused Tokio time and the frozen 50-statement bounds.
Shutdown drops clients, observes both task-result layers, and has a ten-second
overall deadline whose forced fallback aborts remaining driver handles.

The highest-value human review is `database.rs`, followed by the compact
`measure.rs`/`pipeline.rs` durability chain. C7 deliberately rests on that
manual database review rather than a pausing proxy. Please confirm that the
single borrowed `Transaction` and statement order satisfy the checklist
below.

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
- Connection failure is represented by the same `SampleMeasurer` boundary as
  per-sample SQL failure, allowing cached aggregation to publish and uncached
  valid samples to receive durable `failed_count` accounting without retries.
- Large outcome/runtime enum variants were boxed after clippy identified
  avoidable stack-size inflation; behavior and public JSON were unchanged.
- Local F-pg execution exposed a Colima host-forwarding issue. An explicit
  temporary SSH forward was used only to validate locally; the Docker CI
  target itself remains unchanged and all B2/C6/D3 assertions passed.
- A warm local PostgreSQL could occasionally complete D3's exact scan within
  1 ms. The F-pg generator now creates a dedicated timeout table and D3 locks
  it during the round, making the spec's every-sample timeout deterministic
  without changing the production query or tolerance.

No unresolved spec question remains in Stage 3.

## C7 deferred manual review — Gate 3

Deferral approved by the owner on 2026-07-16. No pausing proxy and no C7-lite
instrumentation test are required. Before Gate 3 approval, a human reviews
`crates/seam/src/database.rs` and confirms:

1. One `DatabaseConnection.client` supplies every statement for one sample.
2. `build_transaction().isolation_level(RepeatableRead).start()` is the first
   sample statement.
3. Transaction-scoped settings use `SET LOCAL`, never session-level `SET`.
4. The ground-truth query completes before ascending ef sweep statements.
5. The borrowed transaction is neither committed, rolled back, nor
   reacquired between ground truth and the final sweep; commit occurs only
   after successful completion, while every error explicitly rolls back.

Gate 3 sign-off record (reviewer, date, commit): pending.
