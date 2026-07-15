# VectorSeam Tuner (`seam`) — Specification

Status: draft for review
Scope: the tuner component only — inputs, estimator semantics, configuration,
storage contract, resource budgets, and acceptance criteria. Implementation
methodology lives in `docs/methodology.md` and is out of scope here.

## 1. Context

The tuner is the third VectorSeam component. The SDK samples query vectors,
the collector persists them as immutable `.vseam` segment parts under
`cohorts/<cohort>/window=<ts>/part-<ulid>.vseam` (see
`collection-and-tuning.md`). The tuner continuously runs the same pipeline as
the published `ann-recall-latency` benchmark — exact ground truth, `ef_search`
sweep, tail-percentile calibration, holdout transfer check — and publishes,
per cohort, the recommended `hnsw.ef_search` and a confidence number back to
the same storage. The Python benchmark pipeline is the trusted anchor: where
this spec defines math, it matches that pipeline's semantics exactly.

**Decision — component name**: crate `seam` (workspace member `crates/seam`),
binary `seam`. Rationale: this component is VectorSeam's centerpiece and will
grow beyond ef tuning (central sampling directives, more parameters), so it
owns the short brand word; the `vectorseam-` prefix stays on infrastructure
crates. If ever published to crates.io (`seam` is taken there), it publishes
as `vectorseam-seam`; nothing else in this spec depends on the name.

All timestamps in this spec are UTC. "Storage window" means the collector's
tumbling window (default 600 s); "calibration window" means a target's rolling
window `W` (e.g. 24 h).

## 2. Functional requirements

### 2.1 Pipeline

The tuner is a single long-running process (exactly one instance per storage
root; coordination between instances is a non-goal). Every
`calibration.interval` it runs one **round** over all configured cohorts,
sequentially. A round that is still running when the next tick fires causes
that tick to be skipped (single-flight; no catch-up backlog — the rolling
window self-heals).

Each round, per concrete cohort, has two phases:

**Phase A — measure** (touches the database, produces durable intermediates):

1. Compute `round_end = align(now − close_grace, storage_window)` and the
   round range `[round_end − W, round_end)`.
2. List segment parts under `cohorts/<cohort>/window=<ts>/` for every aligned
   storage window fully inside the round range.
3. Diff against already-measured parts under `measurements/<cohort>/…`. For
   each unmeasured part: fetch and parse the `.vseam` part, then for each kept
   sample run one database transaction that computes exact ground truth and
   the full ef sweep (§2.5), and finally write the part's `truth` and `sweep`
   parquet files (in that order).

**Phase B — aggregate** (pure function of the intermediates, no database):

4. Read all in-scope, metadata-compatible intermediates, split samples into
   train/holdout, compute per-ef compliance quantiles, select the recommended
   ef, compute transfer and confidence (§2.2), and publish the round result
   JSON: `calibrations/<cohort>/round-<ts>.json` plus an overwrite of
   `calibrations/<cohort>/latest.json`.

Database unavailability or per-sample errors degrade Phase A (failed samples
are counted, not retried within the round) but never block Phase B: the tuner
always publishes from whatever intermediates exist. Rationale: a stale-but-
honest output beats a silent gap, and the demo dashboard always has something
to poll.

### 2.2 Estimator semantics

This section is normative. Two implementations that follow it must produce
identical aggregation results from identical intermediates.

#### Population and sample identity

- The population for a round is every kept record in every in-scope segment
  part (after the deterministic measurement cap, §3.1). One record = one
  sample = one query observation. Duplicate query vectors are distinct
  samples. Rationale: the collector's sampling already approximates the
  production query distribution; deduplication would re-weight it.
- A sample's stable identity is `(part_ulid, record_index)` where
  `record_index` is the 0-based ordinal within the part. This identity keys
  intermediates, the train/holdout split, and the measurement cap, so resumes
  and re-listings can never double-count.
- Samples whose measurement failed (SQL error, statement timeout, unsupported
  dtype, dimension mismatch, table smaller than k) are excluded from the
  population and reported as a count. Only dtype `F32` frames are supported.

#### Ground truth

For sample vector `q` against the configured index (table/column/key):

```sql
BEGIN ISOLATION LEVEL REPEATABLE READ;
SET LOCAL statement_timeout = <budget.statement_timeout>;
SET LOCAL enable_indexscan = off;
SELECT <key>, <column> <=> $q AS distance
  FROM <table> ORDER BY <column> <=> $q ASC, <key> ASC LIMIT k;
```

- **Decision — exactness**: exact k-NN is obtained by disabling index scans
  inside the transaction, forcing a sequential scan with top-k sort.
  Rationale: no data export, works on the live table, matches "brute force"
  in the anchor methodology.
- **Decision — tie handling**: ground truth order is `(distance ASC, key
  ASC)`. Ties at the k boundary are broken by ascending primary key.
  Rationale: makes ground truth a deterministic function of the table
  snapshot; the anchor's `torch.topk` tie order is arbitrary, and ties are
  measure-zero for real embeddings, so this refinement cannot move aggregate
  results beyond the stated tolerances.
- **Decision — duplicate keys**: impossible by construction — `<key>` must be
  the table's primary key (or a unique, non-null integer column). MVP
  supports integer keys only (`int2/int4/int8`, stored as int64).
- If the table holds fewer than `k` rows the sample fails ("table smaller
  than k"). The recall denominator is never adjusted.

#### ANN sweep

In the **same transaction** (same snapshot), after re-enabling index scans:

```sql
SET LOCAL enable_indexscan = on;
-- for each ef in calibration.ef_search, ascending:
SET LOCAL hnsw.ef_search = <ef>;
SELECT <key> FROM <table> ORDER BY <column> <=> $q ASC LIMIT k;
```

- The ANN `ORDER BY` is the bare operator expression with **no** key
  tie-break — appending `key` would defeat the HNSW index scan.
- **Decision — snapshot semantics**: ground truth and all ef results for one
  sample share one REPEATABLE READ snapshot, so each per-sample recall is
  internally consistent. Across samples and rounds the table drifts and that
  is accepted. Rationale: per-sample consistency is what recall needs;
  pinning a snapshot across a whole round would hold a long transaction
  against a production database. Consequence: Phase A is not reproducible
  after the fact, Phase B is — reproducibility lives in the durable
  intermediates.
- **Decision — repeats**: each ANN query runs once (the anchor's `repeats: 3`
  exists for latency medians; recall is deterministic given the snapshot).
  Rationale: repeats triple database load against the traffic-control goal;
  latency here is informational only.
- Client-observed latency per ef statement is recorded (informational, never
  part of the target).

MVP supports `metric: cosine` (`<=>`) only; the config field exists and any
other value is a startup error.

#### recall@k

```
recall = |set(returned_keys) ∩ set(gt_keys)| / k
```

Set semantics over distinct keys; denominator is always the target's `k`. If
the ANN query returns fewer than `k` rows (e.g. `ef < k` cannot happen — see
config validation — but concurrent deletes can), recall is computed against
the short result and is thereby penalized. Matches the anchor exactly.

#### Train/holdout split

- **Decision — deterministic hash split**, not RNG shuffle: a sample is in
  the train set iff

  ```
  FNV1a64("s:" + split_seed + ":" + part_ulid + ":" + record_index) mod 10000
      < round(train_fraction * 10000)
  ```

  with `split_seed` (default 7) and `train_fraction` (default 0.7) from
  config, `part_ulid` as its 26-char Crockford string, integers in decimal
  ASCII. FNV-1a 64: `h = 0xcbf29ce484222325`; per byte `h ^= b; h *=
  0x100000001b3 (mod 2^64)`.
  Rationale: language-portable (5 lines in any language, no RNG
  compatibility problem with the Python anchor harness), order-independent,
  and stable — a sample keeps its split membership across rounds and
  resumes, so round-to-round output changes come from data, not re-shuffling.

#### Compliance quantile (percentile calculation)

A target `{k, value, percentile, window}` means: at least `percentile` of
queries must have `recall@k ≥ value` over the window. The tested statistic is
the lower `q = 1 − percentile` quantile of the per-query recall values
(e.g. `percentile: 0.95` → the p05 of recall must be ≥ `value`; the blog's
"p10 ≥ 0.9" is `percentile: 0.90`).

- **Decision — quantile method**: linear interpolation, Hyndman–Fan type 7 —
  exactly numpy's default: sort ascending, `h = (n−1)·q`,
  `Q = x[⌊h⌋] + (h−⌊h⌋)·(x[⌊h⌋+1] − x[⌊h⌋])` (for integer `h`, `Q = x[h]`).
  Defined for all `n ≥ 1`. Rationale: byte-compatible with `np.percentile`
  as used by the trusted anchor (`analyze.py`).
- The population for selection is the **train split**; the holdout is used
  only for transfer/confidence. Per-ef summaries in the output use the full
  population (informational).

#### ef selection rule

Over the configured grid in ascending order:

```
clearing = { ef | train_quantile(ef) ≥ value }
selected = min(clearing)            → status "ok"
selected = max(grid) if clearing=∅  → status "target_unmet"
```

- "Smallest clearing ef" is unique because the grid is strictly increasing;
  no tie-break is needed.
- **Decision — no ef meets the target**: emit `max(grid)` flagged
  `target_unmet`, with transfer and confidence still computed at that ef
  (confidence will be low). Rationale: fail-visible beats fail-silent — the
  max grid value is the most protective actionable setting, and the demo
  dashboard needs an output every round. The tuner never refuses to publish
  a round record.

#### Minimum sample count

- **Decision**: config `min_samples` (default 1000, validated ≥ 100),
  compared against the round's measured population size. Below it the round
  publishes `status: "insufficient_samples"` with `recommended_ef: null`,
  `confidence: null`, and full sample/coverage metadata. No degraded-
  confidence emission. Rationale: an ef recommendation from a tail quantile
  with too few tail points is noise; publishing an explicit refusal keeps
  the "app stays at its conservative default until the tuner speaks" demo
  narrative honest. Demo configs simply lower the threshold.

#### Rolling window semantics

- Round range is `[round_end − W, round_end)`, half-open, where
  `round_end = align(now − close_grace, storage_window_seconds)` and
  `close_grace` (default 30 s) absorbs collector flush latency at window
  close.
- **Membership is per storage window, never per record**: a part is in scope
  iff its header satisfies `window_start ≥ round_end − W` and
  `window_start + window_seconds ≤ round_end`. Only fully closed storage
  windows are consumed; record receive timestamps are not re-checked
  (the collector guarantees them by construction).
- **Late-arriving parts** (spills, delayed flushes) missed by one round are
  picked up by the next: every round re-lists all in-scope windows and
  measures any part it has no intermediates for.
- **No-double-count invariant**: aggregation reads intermediates keyed by
  `part_ulid`; a part contributes exactly once per round regardless of how
  often it was listed, re-listed, or re-measured after a crash. Resume never
  double-counts by construction — there is no incremental accumulator to
  corrupt.

#### Transferability and confidence

Transfer follows the blog: the ef selected on the train split must hold on
the holdout. The round reports `test_quantile_recall` (compliance quantile of
holdout recalls at the selected ef) and `transferred = test_quantile_recall ≥
value`.

- **Decision — confidence is one closed-form number**: with `n` holdout
  samples of which `m` have `recall ≥ value` at the selected ef,

  ```
  confidence = P(θ ≥ percentile),  θ ~ Beta(m + 1, n − m + 1)
             = 1 − I_percentile(m + 1, n − m + 1)
  ```

  where `I` is the regularized incomplete beta function (Bayesian posterior
  of the compliance fraction under a uniform prior). Range (0, 1);
  `confidence ≈ 0.5` means the holdout sits exactly on the target; it rises
  toward 1 as holdout evidence accumulates — exactly the demo's
  "confidence goes up over time" curve. Rationale: deterministic (no
  bootstrap RNG in the spec), computed on the holdout only so the selection
  bias the blog warns about is excluded, and directly interpretable as
  "probability that at least `percentile` of unseen queries meet the
  target". It expresses transferability only — dropped-frame and
  missing-window fractions are reported separately and never folded in.

#### Determinism summary

Phase B is a pure, order-independent, bit-reproducible function of
(intermediates, config). Phase A is deterministic per sample given the table
snapshot its transaction ran under, and not reproducible afterward; this is
recorded, accepted drift.

### 2.3 Configuration

One YAML file, path given by `--config` (env `SEAM_CONFIG`). Loaded once at
startup; runtime changes are a non-goal.

```yaml
calibration:
  interval: 10min                 # round tick
  ef_search: [10, 20, 40, 60, 80, 100, 150, 200, 300, 400]
  train_fraction: 0.7
  split_seed: 7
  min_samples: 1000
  close_grace: 30s

storage:
  root: /var/lib/vectorseam       # same object-store root the collector writes
  window_seconds: 600             # collector storage window; slot accounting

budget:                           # database traffic control, see §3.1
  db_share: 0.10
  max_concurrent_queries: 1
  statement_timeout: 5s
  max_samples_per_part: 1000

indexes:
  stockexchange:
    server: localhost:5432
    database: postgres
    user: postgres
    password_env: SEAM_PG_STOCKEXCHANGE   # optional; see secrets
    table: docs_stackexchange
    key: doc_id
    column: embedding
    metric: cosine                # MVP: cosine only

  reddit:
    server: localhost:5432
    database: postgres
    user: postgres
    password_env: SEAM_PG_REDDIT
    table: docs_reddit
    key: doc_id
    column: embedding

targets:
  queries_search_recall:
    k: 20
    value: 0.9
    percentile: 0.95
    window: 24h

cohorts:
  prod/superuser:                 # pattern: exact cohort name or segment prefix
    index: stockexchange
    target: queries_search_recall
  prod/reddit:
    index: reddit
    target: queries_search_recall
```

**Cohort patterns.** A pattern matches a concrete cohort when it equals the
cohort name or is a whole-segment prefix of it (`prod/reddit` matches
`prod/reddit/tldr`, not `prod/reddit-x`). Longest matching pattern wins;
same-length distinct patterns cannot both match one name. Concrete cohorts
are discovered by listing `cohorts/`; each matched concrete cohort gets its
own calibration and its own outputs. Unmatched cohorts are logged and
skipped.

**Secrets.** Decision: connection fields are plain config; the password, if
any, comes only from the environment variable named by `password_env`
(12-factor, maps directly onto Kubernetes `secretKeyRef` and local `export`).
The config loader **rejects** any inline credential — a `password` key or a
`user:pass@` userinfo in any connection value — with an error pointing at
`password_env`. Passwords never appear in config, logs, intermediates, or
outputs. Rationale: one mechanism, native to every deployment target, and a
hard validation stop is worth more trust than optional advice.

**Startup validation** (all violations are fatal):

- every cohort references an existing index and target; every pattern is a
  valid cohort-name prefix per the grammar
- `0 < percentile < 1`, `0 < value ≤ 1`, `k ≥ 1`, `window ≥
  storage.window_seconds`
- ef grid non-empty, strictly increasing, `min(grid) ≥ k` (pgvector caps
  results at `ef_search`, so `ef < k` can never satisfy the target),
  `max(grid) ≤ 1000` (pgvector bound)
- `0 < train_fraction < 1`, `min_samples ≥ 100`, `0 < db_share ≤ 1`,
  `max_concurrent_queries ≥ 1`, `max_samples_per_part ≥ 1`
- `table`, `column`, `key` are valid PostgreSQL identifiers; they are always
  emitted quoted/escaped
- no inline credentials (see secrets)

### 2.4 Storage contract

All tuner artifacts live in the same object store as the segments, under
prefixes that cannot collide with cohort paths (a cohort named
`…/calibration` is valid grammar, so tuner data must not nest under
`cohorts/`):

```
cohorts/<cohort>/window=<ts>/part-<ulid>.vseam            # collector (input)
measurements/<cohort>/window=<ts>/part-<ulid>.truth.parquet
measurements/<cohort>/window=<ts>/part-<ulid>.sweep.parquet
calibrations/<cohort>/round-<ts>.json                     # immutable history
calibrations/<cohort>/latest.json                         # mutable pointer copy
```

`<ts>` uses the collector's `YYYYMMDDTHHMMZ` format; the round timestamp is
`round_end`. `round-<ts>.json` is written first, then the identical bytes
overwrite `latest.json` (single atomic PUT each — the poll target for the
demo dashboard and, later, the sidecar).

#### Durable intermediates (parquet, zstd)

These schemas are part of the spec: downstream verification consumes them.
Both files carry parquet key-value metadata:
`format_version=1`, `cohort`, `part_ulid`, `window_start`, `window_seconds`,
`received_frame_count`, `record_count` (copied from the segment header),
`index` (config name), `table`, `column`, `key`, `metric`, `k`, `ef_grid`
(comma-joined), `failed_count`, `measured_count`, `computed_at_us`.
Aggregation skips (and reports) files whose `k`, `metric`, `index`, `table`,
`column`, `key`, `ef_grid`, or `format_version` don't match the current
config — the measure phase then re-measures those parts. This is how config
edits across restarts stay safe without runtime reconfiguration.

`part-<ulid>.truth.parquet` — one row per successfully measured sample:

| column            | type          | meaning                                          |
|-------------------|---------------|--------------------------------------------------|
| `record_index`    | int32         | ordinal in the part (sample identity)            |
| `receive_time_us` | int64         | from the segment record                          |
| `gt_keys`         | list<int64>   | exact top-k keys, ordered (distance ASC, key ASC)|
| `gt_distances`    | list<float64> | matching operator distances                      |

`part-<ulid>.sweep.parquet` — one row per (sample, ef):

| column          | type        | meaning                              |
|-----------------|-------------|--------------------------------------|
| `record_index`  | int32       | joins to truth                       |
| `ef`            | int32       | swept `hnsw.ef_search`               |
| `returned_keys` | list<int64> | ANN result keys, result order        |
| `recall`        | float64     | per §2.2 (stored, and re-derivable)  |
| `latency_ms`    | float64     | client-observed, informational       |
| `result_count`  | int32       | rows returned                        |

Write order per part: truth, then sweep. A part is **measured** iff both
files exist; truth-without-sweep (crash between the two PUTs) is remeasured
from scratch, overwriting both. Worst-case redo after a crash is one part.

#### Round output (JSON)

```jsonc
{
  "format_version": 1,
  "cohort": "prod/reddit/tldr",
  "computed_at": "2026-07-15T12:00:41Z",
  "window": { "start": "2026-07-14T12:00:00Z", "end": "2026-07-15T12:00:00Z",
              "duration_seconds": 86400 },
  "target": { "name": "queries_search_recall",
              "k": 20, "value": 0.9, "percentile": 0.95 },
  "index": "reddit",
  "ef_grid": [10, 20, 40, 60, 80, 100, 150, 200, 300, 400],
  "status": "ok",                    // "ok" | "target_unmet" | "insufficient_samples"
  "recommended_ef": 200,             // null when insufficient_samples
  "confidence": 0.971,               // null when insufficient_samples
  "transferred": true,               // test quantile >= value
  "train_quantile_recall": 0.90,
  "test_quantile_recall": 0.90,
  "samples": { "available": 5400,    // sum(record_count) of in-scope parts
               "measured": 5310,     // rows in compatible sweep intermediates / grid size
               "failed": 12,
               "train": 3717, "test": 1593 },
  "dropped_frame_fraction": 0.012,   // 1 - sum(record_count)/sum(received_frame_count)
  "empty_window_fraction": 0.0417,   // in-scope storage-window slots with no parts / total slots
  "parts_used": 144,
  "incompatible_parts": 0,           // intermediates skipped for config mismatch
  "per_ef": [                        // full-population summary, for the dashboard
    { "ef": 10, "quantile_recall": 0.55, "mean_recall": 0.71, "latency_p50_ms": 0.4 },
    { "ef": 20, "quantile_recall": 0.70, "mean_recall": 0.82, "latency_p50_ms": 0.6 }
    // ...
  ]
}
```

Notes: `dropped_frame_fraction` covers collector-side drops only (SDK
queue-full drops are invisible downstream). `empty_window_fraction` cannot
distinguish "no traffic" from "collector down" in the MVP; it is reported as
defined. Confidence excludes both by requirement.

### 2.5 Measure phase — normative details

Per unmeasured part:

1. GET the `.vseam` object; parse with `vectorseam-core` (header + records).
2. Apply the measurement cap: keep record `i` iff
   `FNV1a64("m:" + split_seed + ":" + part_ulid + ":" + i) mod 1000000 <
   floor(min(1, max_samples_per_part / record_count) · 1000000)` —
   deterministic, unbiased, cache-stable thinning of oversized parts.
3. For each kept record: parse the frame (dtype must be F32; the f32 vector
   is passed as a pgvector value), run the single-sample transaction of
   §2.2 under the traffic budget (§3.1). A failed sample increments
   `failed_count`; the transaction is rolled back and the round continues.
4. Buffer rows in memory (tuner memory is deliberately unbudgeted — parts
   are bounded by the collector's 32 MiB spill cap); write
   `*.truth.parquet` then `*.sweep.parquet`.

Connections: one pool per distinct `(server, database)` with
`max_size = budget.max_concurrent_queries`. Every transaction sets
`statement_timeout` via `SET LOCAL`.

### 2.6 Aggregate phase — normative details

1. List in-scope parts (from `cohorts/`) and intermediates (from
   `measurements/`); ignore incompatible intermediates (counted).
2. Load sweep rows; the stored `recall` column is authoritative.
3. Split per §2.2; if `measured < min_samples` publish
   `insufficient_samples` and stop.
4. Select ef per §2.2; compute holdout quantile, `transferred`, confidence.
5. Compose the round JSON (all counters from part headers and file
   metadata); PUT `round-<ts>.json`, then `latest.json`.

Publishing the same `round_end` twice (interval shorter than the storage
window, or restart) overwrites the same key with a fresher computation —
idempotent by design.

## 3. Non-functional requirements

### 3.1 Database traffic control

Goal: a skeptical operator can read three numbers from the config and bound
the tuner's worst-case database impact, PlanetScale-traffic-control style
(budgets per workload slice), without any server-side installation:

> At most `max_concurrent_queries` backend(s), each busy at most `db_share`
> of wall-clock time, and no single statement longer than
> `statement_timeout`.

Mechanisms, in priority order (P0 ships in the MVP; the human owner arbitrates
anything beyond):

- **P0 — concurrency budget**: a global semaphore of
  `max_concurrent_queries` (default 1) over all tuner database work.
- **P0 — duty-cycle pacing** (the "server share" idea): a global pacer;
  after a statement observed to take `t`, the tuner sleeps
  `t · (1 − db_share)/db_share` before the next statement. Expensive exact
  scans automatically stretch the pacing, so the bound holds regardless of
  table size. Default `db_share: 0.10`.
- **P0 — per-statement timeout**: `SET LOCAL statement_timeout` (default 5 s)
  on every transaction; a timed-out sample is a failed sample, never a retry
  storm.
- **P0 — measurement cap**: `max_samples_per_part` (§2.5) bounds per-round
  work for hot cohorts. Sizing guide (documented, not enforced): steady state
  fits the interval when
  `samples_per_interval · (t_scan + Σ t_ann) ≤ db_share · interval`.
- **P1 — plan mode**: `seam plan --config …` prints, per cohort, the pending
  part count, estimated statements, and worst-case busy seconds for the next
  round, then exits without touching the database (PlanetScale's "warn
  mode", adapted to a batch tool).
- **P2 — adaptive load signals**: pause/back off on `pg_stat_activity`
  saturation or replication lag. Explicitly out of the MVP; recorded so the
  priority call is visible.

### 3.2 Fault tolerance

- Every unit of progress is an immutable object PUT (parquet pair, round
  JSON); `latest.json` is the only overwritten key. No local state, no WAL,
  no database writes: after a crash the tuner restarts, re-lists storage,
  and resumes with at most one part of redone work (§2.4).
- Per-sample and per-part failures are counted and reported, never fatal.
  Storage listing/GET failures abort the cohort's round with a log; the next
  tick retries naturally.
- Graceful shutdown (SIGTERM/SIGINT): finish or abandon the in-flight
  sample's transaction, skip remaining work, exit without publishing a
  partial round. Nothing is lost — the next run redoes only the unfinished
  part.

### 3.3 Implementation constraints

- Rust, in this workspace; simple abstractions, new concepts introduced
  deliberately (the two-phase measure/aggregate split above is the intended
  seam between database glue and the semantically rich estimator, which must
  stay a pure function).
- Proposed crates (guidance, not contract): `tokio` + `tokio-postgres` +
  `deadpool-postgres` (pool = concurrency budget) + `pgvector` (Vector type);
  `arrow`/`parquet` (arrow-rs, zstd) for intermediates; `object_store`
  (already in workspace) for all storage IO; `serde` + a maintained YAML
  parser + `humantime-serde` for durations; `statrs` (or `puruspe`) for the
  regularized incomplete beta; `clap`, `tracing`, `thiserror`/`anyhow`,
  `ulid` as in the collector. FNV-1a is 5 lines, no dependency.
- Tuner process memory is intentionally unbudgeted (isolated component);
  database traffic is the only guarded resource.

## 4. Non-goals

- Runtime configuration changes (target edits require re-measurement; the
  config-fingerprint check in §2.4 handles restarts instead).
- Tight tuner memory budgets; strict realtime (demo compresses time via
  config knobs, production tolerates ≥ 10 min staleness).
- APIs, dashboards, metrics endpoints — file exchange only.
- SDK/collector consuming the recommendation; central sampling directives.
- Full SDK→collector→tuner end-to-end integration tests (later milestone).
- Data retention/compaction of segments, intermediates, or rounds.
- Non-integer primary keys; non-cosine metrics; non-pgvector backends.
- Multi-instance coordination or leader election (exactly one tuner per
  storage root).
- Per-server traffic budgets (the global budget is stricter; revisit when
  one tuner spans many servers).

## 5. Acceptance criteria

Executable assertions. ε values are absolute. "F-pg" is a seeded fixture:
~10k synthetic normalized f32 vectors (dim ≥ 64) with integer keys loaded
into pgvector (HNSW, cosine), plus ~500 seeded query vectors emitted **in the
same order** as (a) parquet for the Python anchor and (b) `.vseam` segments
for the tuner, with fixture-time verification that no ground-truth k-boundary
distance tie exists. "F-agg" means hand-crafted intermediates (parquet pairs
and segment headers) fed to Phase B only — no database.

### A. Anchor reproduction (the correctness anchor)

Same live pgvector instance and index, same k = 10, same grid
`[10, 20, 40, 80, 160]`, and identical train/holdout membership (both sides
computed by the §2.2 hash split — the anchor harness reuses the published
pipeline's recall/percentile/selection code with that split; numpy RNG
shuffles are not part of the contract):

- **A1** For ≥ 99% of (query, ef) pairs, tuner `recall` equals the anchor's
  recall exactly; disagreements are only where ground truth differs (torch
  float math vs pgvector scan near-ties).
- **A2** Per-ef full-population mean recall: |tuner − anchor| ≤ 0.005.
- **A3** Per-ef train compliance quantile (`percentile: 0.90`,
  `value: 0.9`): |tuner − anchor `np.percentile(_, 10)`| ≤ 0.01.
- **A4** `recommended_ef` identical to the anchor's
  "min ef with train p10 ≥ 0.9".
- **A5** `test_quantile_recall`: |tuner − anchor| ≤ 0.01, and `transferred`
  identical.

### B. Estimator semantics (one per §2.2 decision; F-agg unless stated)

- **B1 recall**: k = 10, `gt_keys = [1..10]`; `returned_keys =
  [1,2,3,11,12,13,14,15,16,17]` → recall = 0.3 exactly; `returned_keys` of 7
  rows containing 5 hits → 0.5 (short results penalized); two samples with
  identical vectors are two population members.
- **B2 tie-break** (F-pg): rows 7 and 9 given identical vectors that
  straddle the k boundary → `gt_keys` contains 7, not 9, on repeated runs.
- **B3 quantile**: recalls `[0.5, 0.7, 0.9, 1.0]`, `percentile: 0.95` →
  compliance quantile = 0.53 exactly (h = 0.15, type-7 linear); n = 1 →
  the single value.
- **B4 split**: membership computed by an independent 5-line FNV-1a
  reference matches the tuner for a fixed sample list; observed train
  fraction within 0.7 ± 0.03 for n = 10⁴; membership unchanged after
  simulated resume (same identities → same split).
- **B5 selection**: train quantiles `{10:0.62, 20:0.85, 40:0.91, 80:0.93,
  160:0.95}`, value 0.9 → `recommended_ef = 40`, `status = "ok"`.
- **B6 target unmet**: value 0.99 with B5's quantiles → `recommended_ef =
  160`, `status = "target_unmet"`, confidence and `test_quantile_recall`
  still present.
- **B7 min samples**: `min_samples = 1000`; 999 measured →
  `status = "insufficient_samples"`, `recommended_ef = null`,
  `confidence = null`, sample counts still reported; 1000 measured → a
  recommendation is emitted.
- **B8 window membership**: storage window 600 s, W = 3600 s,
  `round_end = 12:00` → exactly the six windows 11:00–11:50 are in scope; a
  part at 10:50 and one at 12:00 are excluded; with no part in the 11:20
  slot, `empty_window_fraction = 1/6 ± 1e-12`.
- **B9 no double-count**: two consecutive overlapping rounds over the same
  parts → each round's `samples.available` equals the sum of distinct
  in-scope `record_count`s; a part listed in both rounds triggers zero new
  database statements in the second (assert via statement counter).
- **B10 confidence**: n = 100, m = 100, `percentile: 0.95` →
  `confidence = 1 − 0.95¹⁰¹ ≈ 0.99438` within 1e-5; m = 0 → confidence
  < 1e-6; confidence values agree with `scipy.stats.beta.sf(percentile,
  m+1, n−m+1)` within 1e-6 on a grid of (n, m).
- **B11 drop fraction**: part headers (received = 100, records = 80) and
  (received = 50, records = 50) → `dropped_frame_fraction = 2/15 ± 1e-12`.

### C. Edge cases and durability

- **C1 resume mid-part**: storage containing a `*.truth.parquet` without its
  `*.sweep.parquet` → the next round re-measures that part (both files
  rewritten, database statements issued for it alone) and the published
  round equals a never-crashed run on the same data.
- **C2 database down**: with intermediates present and the database
  unreachable → Phase B still publishes; `samples.measured` reflects cached
  work only; process exits 0 at shutdown.
- **C3 config fingerprint**: intermediates written with k = 10 are ignored
  and re-measured after k changes to 20; `incompatible_parts` > 0 in the
  next round.
- **C4 empty round**: zero in-scope parts → `insufficient_samples`,
  `samples.available = 0`, `empty_window_fraction = 1.0`.
- **C5 config validation**: each of — inline `password` key, `user:pass@`
  in `server`, ef grid containing a value < k or > 1000 or non-increasing,
  cohort with unknown index/target, `percentile: 1.0`, `window <
  storage.window_seconds` — fails startup with a distinct error; the
  `password_env` error message names the expected mechanism.
- **C6 failed samples**: a table with fewer than k rows → every sample fails
  with "table smaller than k", `samples.failed = samples.available`, round
  publishes `insufficient_samples`, process keeps running.

### D. Resource ceilings (metric: statement wall-time duty cycle and
in-flight statement count, measured at the tuner's database client)

- **D1 duty cycle**: `db_share = 0.20` against an instrumented database
  taking ~50 ms/statement, ≥ 50 statements → total elapsed ≥ 0.95 · (total
  busy time / 0.20); equivalently the busy fraction over the run ≤ 0.21.
- **D2 concurrency**: instrumented max in-flight statements ≤
  `max_concurrent_queries` for the entire run (default config → exactly 1).
- **D3 statement timeout**: `statement_timeout = 1ms` on F-pg → every sample
  fails within the round (no retries, counted), the round completes, and no
  tuner statement remains running server-side afterward.
- **D4 measurement cap**: a part with `record_count = 10 ·
  max_samples_per_part` → measured samples ≤ 1.02 · `max_samples_per_part`,
  and the kept subset is identical across two runs (deterministic).
