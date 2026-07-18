# Stage 1 summary

## Passed

- The owner-approved frozen-spec correction makes the round JSON example's ef
  grid valid for `k = 20`.
- All 28 criteria are traced. Twenty-seven have compiled,
  ignored-until-implementation Rust tests whose names contain their criterion
  IDs and whose assertions preserve the spec's literal values and tolerances.
  C7 is deferred with owner approval to the Gate 3 transaction-construction
  review in `docs/REVIEW_MAP.md`.
- Three database-free F-agg harness tests pass: exact parquet schemas and
  metadata with zstd, `.vseam` segment-header round trip through
  `vectorseam-core`, and a truth-only crash state.
- The F-pg generator completed against pgvector with 10,000 seed-0 normalized
  f32 documents (dimension 64), 500 ordered and pairwise-distinct queries, a
  cosine HNSW index, PostgreSQL k-boundary no-tie verification, and a separate
  deterministic B2 tie table.
- The Python driver completed and produced 2,500 recall rows by calling the
  existing anchor's ground-truth, sweep/recall, percentile, summary, and ef
  selection functions. It changes only the split membership hook to use the
  independent five-line FNV-1a rule.
- `cargo test --workspace`, `make lint-rust`, `make doc-rust`, and all 65
  Python unit tests pass.

## Red / intentionally not green

- The owner-requested ascending seed search selected seed `0`, the first
  candidate. It passes the strengthened PostgreSQL boundary-gap check for all
  500 queries; the minimum observed gap is `1.3683911141981753e-6`. The
  canonical fixture and reused Python anchor both reran successfully.
- A1–A5, B1–B12, C1–C6, C8, and D1–D3 are intentionally ignored because Stage 1
  contains no tuner implementation. Their exact status is in
  `docs/acceptance-map.md`.
- C7 is intentionally not machine-gated. Its deferral and manual Gate 3
  checklist are recorded in `docs/acceptance-map.md` and `docs/REVIEW_MAP.md`.
- The new Linux CI job has not run in this local workspace. Its isolated
  Compose service became healthy locally, but this host's Colima instance did
  not dynamically forward the newly published port 55432. The same complete
  fixture and anchor commands passed against the repository's existing
  pgvector container on port 5432. The isolated Compose wiring remains the CI
  path.

## Open questions

- None. The owner resolved the four pre-Stage-2 questions: insufficient
  transfer fields are null, empty splits are insufficient, the empty-round
  drop fraction is zero, and identifiers use quoted/delimited semantics. The
  decisions are now part of the frozen specification.

Gate 1 is approved and its pre-Stage-2 decisions are recorded. Stage 2 may
start.
