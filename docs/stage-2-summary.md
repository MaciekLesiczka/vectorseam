# Stage 2 summary

## Passed

- The typed YAML configuration model implements all startup validation in
  §2.3 and C5, including top-level data-source references, unique
  `(server, database)` pairs, conditional password-environment checks, secret
  rejection, duration and numeric bounds, minute-aligned storage windows,
  exact target/storage-window divisibility, owner-approved split-threshold
  handling, and quoted PostgreSQL identifiers.
- Phase B is a pure synchronous function of owned intermediates, listing,
  config, aligned round end, and caller-supplied `computed_at`. It performs no
  IO, async work, or clock reads.
- The estimator implements stored-recall handling, type-7 quantiles, exact
  FNV-1a split membership, smallest-clearing ef selection, insufficient and
  target-unmet outputs, regularized-beta confidence, rolling membership,
  listing and cross-part deduplication, drop/coverage counters, compatibility
  filtering, per-ef summaries, and deterministic round JSON.
- A typed Phase A table-smaller-than-k abort takes precedence over cached
  population size and forces `insufficient_samples` with the abort error and
  null selection fields.
- Synchronous glue reads the exact §2.4 parquet field schemas and metadata,
  cross-checks metadata `measured_count` against decoded truth rows, joins
  truth/sweep rows, and parses segment headers through `vectorseam-core`.
- B1, B3–B8, B10–B11, C4–C5, and C8 are fully passing. The Phase B paths of
  B9, B12, C3, and C6 also pass; their remaining Phase A/statement assertions
  stay explicit and ignored for Stage 3.
- The four required property tests pass: recommendation monotonicity under
  value relaxation, split stability, quantile bounds, and confidence
  monotonicity in successes for fixed holdout size.
- `cargo test --workspace`, `cargo clippy --workspace --all-targets
  --all-features -- -D warnings`, formatting, and workspace documentation all
  pass. The database-free CI target now includes Stage 2 acceptance and
  property tests, not only fixture-builder self-tests.

## Red / intentionally not green

- A1–A5 remain blocked until Stage 4 wire-up.
- B2 is a Phase A PostgreSQL ground-truth assertion and remains blocked until
  Stage 3.
- B9, B12, and C3 remain blocked overall only because their Stage 3 halves are
  still ignored; their Phase B halves are green.
- C1, C2, D1, and D3 remain blocked until Stage 3.
- D2 and `max_concurrent_queries` were removed with explicit owner approval on
  2026-07-17; the acceptance-map row records that sign-off.
- C6 remains blocked overall only because its Stage 3 table-size detection,
  statement-count, and other-cohort continuation path is still ignored; its
  Phase B forced-insufficient path is green.
- C7 remains the owner-approved manual deferral. Its transaction-construction
  checklist and pending Gate 3 sign-off are retained in `docs/REVIEW_MAP.md`.

## Open questions

- None. No assertion was weakened and no tolerance was widened.

## Gate 2 review corrections

- Replaced the generic aggregation error pass-through with a typed Phase A
  abort and added a cached-population C6 regression test.
- Enforced storage/target window alignment instead of truncating residual
  seconds.
- Removed dead no-op and unused alignment/adapter code; narrowed internal
  visibility and recorded the least-visibility rule in `agents.md`.
- Made C8 compare raw deterministic JSON bytes after normalizing only
  `computed_at`.
- Replaced tautological B4/B12 stability calls with aggregate-level survivor
  movement coverage.
- Excluded zero from the ef-selection property generator, aligned the
  confidence percentile guard with `(0, 1)`, and added the parquet
  `measured_count` integrity check.

Gate 2 was approved. The subsequent Stage 3 preflight data-source/config
revision is covered by active C5 and unit tests; Phase A implementation has
not started.
