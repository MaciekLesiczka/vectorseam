# Tuner review map

This map covers the Stage 1 tuner additions. There is no tuner implementation
yet.

## Tier 1 — semantically rich / critical

- `python/seam_harness/anchor.py` — owns the independent FNV-1a split and the
  adapter that reuses the trusted ground-truth, recall, percentile, and
  selection functions without copying their math.
- `crates/seam/tests/support/f_agg.rs` — encodes the frozen truth/sweep parquet
  schemas, metadata, object keys, and segment-header fixtures.
- `crates/seam/examples/f_pg_fixture.rs` — determines fixture reproducibility,
  query ordering, no-tie verification, the HNSW dataset, and the B2 tie table.
- `crates/seam/tests/acceptance_b_estimator.rs` — transcribes every numerical
  estimator expectation that Stage 2 must satisfy.
- Stage 3 transaction-construction module (path to be recorded when added) —
  owns the single-connection, single-transaction snapshot invariant and must
  receive the C7 manual review below.

## Tier 2 — standard

- `crates/seam/tests/f_agg_builders.rs` — validates fixture schemas, metadata,
  zstd compression, crash-shaped truth-only state, and core segment round
  trips without a database.
- `crates/seam/tests/support/anchor.rs` — validates and loads the Python
  comparison artifact that the Rust A tests consume.
- `crates/seam/tests/acceptance_a_anchor.rs` — defines the five anchor
  comparison tolerances.
- `crates/seam/tests/acceptance_c_durability.rs` — defines durability and edge
  case outcomes across Stages 2–4.
- `crates/seam/tests/acceptance_d_resources.rs` — defines the instrumented
  resource ceilings for Stage 3.
- `tests/seam/docker-compose.yml` — provides the isolated pgvector test
  service.
- `.github/workflows/ci.yml` — runs database-free F-agg checks and the
  dockerized pgvector harness.
- `Makefile` — exposes the local F-agg, F-pg, and anchor harness entry points.

## Tier 3 — glue

- `crates/seam/src/lib.rs` — empty Stage 1 package marker; contains no tuner
  behavior.
- `crates/seam/tests/support/mod.rs` — shared pending-test marker and fixture
  module wiring.
- `python/seam_harness/__init__.py` — package marker.
- `python/vectorseam/tests/test_seam_anchor_harness.py` — database-free Python
  reference-hash smoke tests.
- `crates/seam/Cargo.toml` and workspace `Cargo.toml` — test dependency and
  package wiring.

## Confidence report

I am confident that all 28 criteria are traced: 27 criterion IDs are present
in machine-test names, while C7 is explicitly deferred to human review with
the owner's approval. The literal expected values and tolerances are sourced
from the frozen spec. I am also confident in the parquet column/metadata
transcription and in using `vectorseam-core` for `.vseam` segment
serialization.

The highest-value human review is the anchor adapter: it deliberately patches
only the anchor's query split while calling its existing private calculation
functions. Review should confirm that this is the desired meaning of “reuse”
before Gate 1. The F-pg fixture distribution is deterministic uniform data
normalized to unit length; the spec fixes the size, dimensionality, ordering,
and properties but not a distribution, so this choice is harness policy rather
than tuner behavior.

I first considered emitting `.vseam` bytes in Python. I changed course so the
fixture uses `vectorseam-core::segment::write_segment`, avoiding a second
segment-format implementation. I also use a dedicated tie table for B2 so the
main anchor fixture can preserve its required no-boundary-tie property.

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
