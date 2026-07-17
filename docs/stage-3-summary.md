# Stage 3 summary

## Passed

- Per-part `.vseam` parsing, exact-byte deduplication, measurement, frozen
  parquet encoding, and truth-before-sweep durability.
- Object-store window listing/diffing, crash resume, config-fingerprint
  remeasurement, cached database-down publication, round/latest publication,
  noncanonical-name skipping, retryable connection outages, and graceful
  cancellation without a partial round.
- One serialized PostgreSQL connection and duty-cycle pacer per data source;
  one `REPEATABLE READ` transaction per sample with bound vectors, quoted
  identifiers, exact-query tie-break, ascending ef sweep, statement timeout,
  client operation deadline, explicit rollback, once-per-round reconnect,
  and supervised shutdown. Cooldown cancellation is allowed only before the
  transaction starts.
- F-agg acceptance/property suite, F-pg B2/C2/C6/D3 suite, full workspace
  tests, clippy, warning-free rustdoc, Rust 1.85 MSRV check, and the Phase B
  Criterion baseline. C2 covers both startup outage and in-flight client
  timeout recovery. D3 uses a locked, dedicated F-pg table to make the frozen
  1 ms timeout deterministic and counts sample transactions to prove there
  are no retries.

## Red or deferred

- A1–A5 remain blocked until Stage 4 anchor reproduction and wire-up.
- C7 is deferred from machine gating with owner approval; the explicit manual
  transaction checklist in `docs/REVIEW_MAP.md` was completed and approved
  on 2026-07-17.
- D2 remains recorded as the owner-approved removed criterion.

## Open questions

No unresolved Stage 3 specification question remains. The C7 manual review is
complete, and Gate 3 is approved.
