# Stage 2 summary

## Passed

- The typed YAML configuration model implements all startup validation in
  §2.3 and C5, including secret rejection, config references, duration and
  numeric bounds, owner-approved split-threshold handling, and quoted
  PostgreSQL identifiers.
- Phase B is a pure synchronous function of owned intermediates, listing,
  config, aligned round end, and caller-supplied `computed_at`. It performs no
  IO, async work, or clock reads.
- The estimator implements stored-recall handling, type-7 quantiles, exact
  FNV-1a split membership, smallest-clearing ef selection, insufficient and
  target-unmet outputs, regularized-beta confidence, rolling membership,
  listing and cross-part deduplication, drop/coverage counters, compatibility
  filtering, per-ef summaries, and deterministic round JSON.
- Synchronous glue reads the exact §2.4 parquet field schemas and metadata,
  joins truth/sweep rows, and parses segment headers through
  `vectorseam-core`.
- B1, B3–B8, B10–B11, C4–C5, and C8 are fully passing. The Phase B paths of
  B9, B12, and C3 also pass; their remaining Phase A/statement assertions stay
  explicit and ignored for Stage 3.
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
- C1, C2, C6 and D1–D3 remain blocked until Stage 3.
- C7 remains the owner-approved manual deferral. Its transaction-construction
  checklist and pending Gate 3 sign-off are retained in `docs/REVIEW_MAP.md`.

## Open questions

- None. No assertion was weakened and no tolerance was widened.

Gate 2 is ready for the required human review of Tier 1 modules in
`docs/REVIEW_MAP.md`. Stage 3 has not started.
