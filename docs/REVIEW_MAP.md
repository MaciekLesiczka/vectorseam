# Tuner review map

This map covers the completed Stage 1 harness and Stage 2 estimator. Stage 3
transaction, pacing, and object-store orchestration modules are not present
yet.

## Tier 1 — semantically rich / critical

- `crates/seam/src/math.rs` — owns recall set semantics, exact FNV-1a and split
  membership, type-7 interpolation, ef selection, and beta-posterior
  confidence.
- `crates/seam/src/accounting.rs` — owns half-open rolling-window membership,
  distinct listed-part accounting, coverage, and collector drop fractions.
- `crates/seam/src/population.rs` — owns cross-part survivor ordering,
  full-population summaries, train quantiles, ef selection, holdout transfer,
  and confidence inputs.
- `crates/seam/src/aggregate.rs` — composes compatibility filtering,
  deterministic population splitting, counters, typed Phase A abort
  precedence, insufficient output semantics, and ordered round records from
  the smaller Tier 1 primitives.
- `crates/seam/tests/acceptance_b_estimator.rs` — asserts every implemented B
  literal and tolerance, including independent FNV and SciPy reference sides.
- `crates/seam/tests/estimator_properties.rs` — guards monotone selection,
  stable split membership/fraction, bounded quantiles, and monotone
  confidence.
- `python/seam_harness/anchor.py` — owns the independent Python FNV split
  adapter while reusing the trusted anchor's ground-truth, recall, percentile,
  summary, and selection functions.
- `crates/seam/tests/support/f_agg.rs` — encodes frozen truth/sweep parquet
  schemas, metadata, object keys, and segment-header fixtures.
- `crates/seam/examples/f_pg_fixture.rs` — determines seed-0 fixture
  reproducibility, query ordering, pairwise distinctness, strengthened
  boundary-gap verification, HNSW data, and the B2 tie table.
- Stage 3 transaction-construction module (path to be recorded when added) —
  will own the single-connection snapshot invariant and must receive the C7
  manual review below.

## Tier 2 — standard

- `crates/seam/src/intermediate.rs` — synchronously validates and reads the
  frozen parquet field schemas and metadata, cross-checks `measured_count`
  against truth rows, then joins truth rows to the authoritative stored
  recall/latency rows. It contains IO but no estimator policy.
- `crates/seam/src/model.rs` — defines pure aggregation inputs and the ordered,
  serializable round JSON contract.
- `crates/seam/tests/f_agg_builders.rs` — validates fixture schemas, metadata,
  zstd compression, crash-shaped truth-only state, ULID ordering, and core
  segment round trips without a database.
- `crates/seam/tests/support/anchor.rs` — validates and loads the Python
  comparison artifact consumed in Stage 4.
- `crates/seam/tests/acceptance_a_anchor.rs` — defines the five anchor
  comparison tolerances.
- `crates/seam/tests/acceptance_c_durability.rs` — exercises Stage 2 config,
  compatibility, empty-round, parquet-reading, and reproducibility behavior;
  later durability halves remain explicit ignored tests.
- `crates/seam/tests/acceptance_d_resources.rs` — defines the instrumented
  resource ceilings for Stage 3.
- `tests/seam/docker-compose.yml` — provides the isolated pgvector test
  service.
- `.github/workflows/ci.yml` and `Makefile` — run the database-free Stage 2
  acceptance/property suite on every push and retain the dockerized pgvector
  harness job.

## Tier 3 — glue

- `crates/seam/src/config.rs` — typed YAML parsing, defaults, reference checks,
  credential rejection, minute-aligned storage/target-window validation,
  duration parsing, and quoted-identifier validation.
- `agents.md` — binding project guidance, including the least-possible Rust
  visibility rule added during Gate 2 review.
- `crates/seam/src/lib.rs` — package module wiring only.
- `crates/seam/tests/support/mod.rs` — shared pending-test marker and fixture
  module wiring.
- `python/seam_harness/__init__.py` — package marker.
- `python/vectorseam/tests/test_seam_anchor_harness.py` — database-free Python
  reference-hash smoke tests.
- `crates/seam/Cargo.toml` and workspace `Cargo.toml` — package and dependency
  wiring.

## Confidence report

I am confident in the exact B1, B3–B8, B10–B11, C4–C5, C8, and Phase B C6
behavior now machine-gated, and in the Phase B halves of B9, B12, and C3.
The estimator has no filesystem, async, or clock access: `computed_at` and the
aligned round end are explicit inputs. Population and part iteration are
normalized through ordered maps before floating-point reductions, and round
JSON uses struct field order rather than unordered maps.

The highest-value Stage 2 human review is the compact
`accounting.rs`/`population.rs`/`aggregate.rs` chain: confirm the half-open
part-membership predicate, lexicographic `(part_ulid, record_index)` survivor,
compatible-part counters, typed Phase A abort precedence, and the
owner-approved null/empty-split semantics.
In `math.rs`, confidence is evaluated as the mathematically
equivalent complementary regularized-beta expression
`I_(1-p)(n-m+1, m+1)`, avoiding cancellation from a literal `1 - I_p`; the
SciPy grid and B10 closed form pass at the frozen tolerances.

I am also confident that `intermediate.rs` preserves the intended seam: it
performs synchronous storage decoding, while `aggregate` consumes only owned
in-memory values. The reader validates frozen Arrow fields but ignores
non-contract Arrow schema metadata. It materializes only identity, dedup, and
stored recall/latency columns because §2.6 makes stored recall authoritative;
ground-truth and returned-key payloads remain part of the validated parquet
schema without being re-scored.

Changes from the first approach:

- The PostgreSQL fixture originally used seed 7. The owner-directed ascending
  search selected seed 0, whose minimum boundary gap is
  `1.3683911141981753e-6`.
- The parquet reader initially compared entire Arrow schemas, including
  implementation metadata not frozen by §2.4. It now compares the exact field
  list, matching the harness contract.
- The B12 fixture originally declared three records but only one received
  frame. Its header now correctly declares three received frames; assertions
  were not weakened.
- Inline `password` values are discarded during YAML deserialization rather
  than retained in the raw model; even `password: null` is rejected.
- The initial aggregation module combined membership, deduplication,
  selection, and orchestration in one file. It was split into focused
  accounting and population modules so Gate 2 Tier 1 review remains small.
- Gate 2 review exposed that a generic pass-through `error` could produce
  `status: "ok"` with a table-smaller-than-k error when cached intermediates
  met `min_samples`. The input is now a typed `PhaseAAbort`; this condition
  takes precedence over population size and forces the insufficient shape.
- The storage-window formatter initially truncated residual seconds. Startup
  validation now requires minute-aligned storage windows and target windows
  that are exact storage-window multiples.
- The first C8 assertion normalized both byte streams through
  `serde_json::Value`, which could hide field-order changes. It now normalizes
  only `computed_at` on the typed outputs and compares serialized bytes
  directly.
- The initial B4/B12 survivor-stability calls repeated the same pure helper.
  The replacement aggregates fixtures where the duplicate survivor genuinely
  moves to a different part and compares realized train/test counts.
- The parquet reader now rejects metadata `measured_count` values that differ
  from the decoded truth-row count.
- Dead no-op population handling, an unused alignment wrapper, and an unused
  segment adapter were removed. Remaining crate internals use the least
  visibility needed by their callers.

No Stage 2 behavior currently needs an additional owner decision. Tier 1
review is still requested before Gate 2 approval, as required by the staged
plan.

## C7 deferred manual review — Gate 3

Deferral approved by the owner on 2026-07-16. No pausing proxy and no C7-lite
instrumentation test are required. Before Gate 3 is approved, a human must
review the transaction-construction module and confirm all of the following:

1. One checked-out database connection owns every statement for one sample.
2. `BEGIN ISOLATION LEVEL REPEATABLE READ` establishes the transaction before
   the ground-truth statement.
3. Transaction-scoped settings use `SET LOCAL`, never session-level `SET`.
4. Ground truth completes before the ascending ef sweep begins.
5. There is no commit, rollback, or connection reacquisition between ground
   truth and the final sweep statement; the transaction commits only after the
   sweep completes.

Gate 3 sign-off record (reviewer, date, commit): pending.
