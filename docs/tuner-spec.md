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
the same storage.

**Decision — component name**: crate `seam` (workspace member `crates/seam`),
binary `seam`. Rationale: this component is VectorSeam's centerpiece and will
grow beyond ef tuning (central sampling directives, more parameters), so it
owns the short brand word; the `vectorseam-` prefix stays on infrastructure
crates. If ever published to crates.io (`seam` is taken there), it publishes
as `vectorseam-seam`; nothing else in this spec depends on the name.

Terms used throughout:

- All timestamps are UTC. **Storage window**: the collector's tumbling
  window (default 600 s). **Calibration window**: a target's rolling window
  `W` (e.g. 24 h).
- **Anchor**: the published `ann-recall-latency` Python pipeline
  (`python/ann-recall-latency/` — `ground_truth.py`, `sweep.py`,
  `analyze.py`), the reference implementation behind the blog's findings.
  It is trusted: where this spec defines math it matches the anchor's
  semantics exactly, and acceptance criteria A compare the tuner's numbers
  against it on shared fixtures.
- **Intermediates**: the per-part parquet files (`*.truth.parquet` and
  `*.sweep.parquet`, schemas in §2.4) that the tuner persists next to the
  segments — for every measured sample, the exact ground-truth top-k and
  the per-ef ANN results. They are the durable raw material from which
  every published number is computed, the unit of crash recovery, and the
  interface downstream verification consumes.

## 2. Functional requirements

### 2.1 Pipeline

The tuner is a single long-running process (exactly one instance per storage
root; coordination between instances is a non-goal). Every
`calibration.interval` it runs one **round** over all configured cohorts,
sequentially. At most one round runs at a time (single-flight): if a round is
still running when the next tick fires, that tick is skipped and nothing is
queued for later.

Skipping ticks is safe because a round is *not* "process the data assigned to
tick T". Every round does the same two things regardless of when it runs:
(1) measure whatever in-scope parts have no durable intermediates yet, and
(2) re-aggregate the entire rolling window from all intermediates. There is
no per-tick work item that can be missed — anything a skipped tick would have
measured is still sitting unmeasured in storage, and the next round's diff
finds it. Ticks therefore control only the *freshness* of the published
output, never its completeness: after downtime, crashes, or slow rounds, the
next successful round produces the same result as if nothing had been
skipped. That is what "the rolling window self-heals" means throughout this
spec.

Each round, per configured cohort, has two phases:

**Phase A — measure** (touches the database, produces durable intermediates):

1. Compute `round_end = align(now)` and the round range
   `[round_end − W, round_end)`. `align(t)` floors a timestamp to the
   previous storage-window boundary, exactly the collector's
   `aligned_window_start`: `align(t) = t − (t mod storage.window_seconds)`.
   Example with 600 s windows: `align(12:07:23) = 12:00:00`, so a round
   ticking at 12:07:23 with `W = 24h` covers
   `[previous day 12:00:00, today 12:00:00)`.
2. List segment parts under `cohorts/<cohort>/window=<ts>/` for every aligned
   storage window fully inside the round range, plus the collector's window
   observation markers (§2.4) for the same range, used for coverage
   accounting.
3. Diff the listed parts against already-measured parts under
   `measurements/<cohort>/…`. For each unmeasured part: fetch and parse the
   `.vseam` part, then for each kept sample run the per-sample database
   transaction — exact ground truth plus one ANN query per value of the
   `calibration.ef_search` grid (required config, §2.3); the full SQL is
   spelled out in §2.2 — and finally write the part's `truth` and `sweep`
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

Every ambiguity in estimator semantics is a place where two correct-looking
implementations can silently disagree. This section removes that freedom: any
two implementations that follow it must produce identical aggregation results
from identical intermediates.

#### Population, deduplication, and sample identity

- A **sample** is one kept record in an in-scope segment part. Its **vector
  hash** is `FNV1a64` over the frame's raw little-endian f32 payload bytes;
  two samples are duplicates iff their vector bytes are equal (a 64-bit
  hash collision merging two distinct vectors is accepted as negligible).
- **Decision — deduplicate**: within a part, each distinct vector is
  measured once (Phase A, §2.5); across the round window, the population is
  one member per distinct vector hash (Phase B), keeping the row with the
  lexicographically smallest `(part_ulid, record_index)`. Rationale: an
  identical vector yields an identical database result within a round, so a
  duplicate adds cost but no information — and it actively harms the
  statistics: duplicates spanning the train/holdout split leak (transfer
  must be validated on genuinely unseen queries), and duplicated holdout
  rows inflate `n` in the confidence formula. Recorded tradeoff: the
  population is *distinct query vectors*, not traffic-weighted queries — a
  hot query counts once; `dup_count` is persisted so post-MVP re-weighting
  would not require re-measurement.
- A duplicate recurring across *different* parts is still measured once per
  part — a part's intermediates must stay self-contained for the resume
  model — and Phase B discards the extra measurement. Accepted MVP waste,
  bounded by `max_samples_per_part`.
- A sample's storage identity is `(part_ulid, record_index)`,
  `record_index` 0-based within the part; it keys the intermediates and the
  measurement cap, so resumes and re-listings can never double-count. The
  train/holdout split is keyed by the vector hash instead (see below): a
  duplicate can never straddle the split, and membership survives the
  rolling window even when the surviving occurrence changes as old parts
  age out.
- Samples whose measurement failed (SQL error, statement timeout, unsupported
  dtype, dimension mismatch, table smaller than k) are excluded from the
  population and reported as a count. Only dtype `F32` frames are supported.

#### Ground truth and ef sweep — one transaction per sample

Every measured sample runs exactly one database transaction containing the
exact ground-truth query followed by one ANN query per value of the
`calibration.ef_search` grid, ascending. The grid is required configuration
(§2.3); the published benchmark grid `[10, 20, 40, 60, 80, 100, 150, 200,
300, 400]` is the recommended starting point. Spelled out in full for an
index `{table: docs_reddit, key: doc_id, column: embedding, metric: cosine}`
with `k = 10` and that grid:

```sql
BEGIN ISOLATION LEVEL REPEATABLE READ;
SET LOCAL statement_timeout = '5s';        -- budget.statement_timeout

-- 1. Exact ground truth: force a sequential scan (brute-force k-NN).
SET LOCAL enable_indexscan = off;
SELECT "doc_id", "embedding" <=> '[0.011,-0.027,…]'::vector AS distance
FROM "docs_reddit"
ORDER BY "embedding" <=> '[0.011,-0.027,…]'::vector ASC, "doc_id" ASC
LIMIT 10;

-- 2. ANN sweep: same snapshot, index scans back on,
--    one query per grid value, ascending.
SET LOCAL enable_indexscan = on;

SET LOCAL hnsw.ef_search = 10;
SELECT "doc_id" FROM "docs_reddit"
ORDER BY "embedding" <=> '[0.011,-0.027,…]'::vector ASC
LIMIT 10;

SET LOCAL hnsw.ef_search = 20;
SELECT "doc_id" FROM "docs_reddit"
ORDER BY "embedding" <=> '[0.011,-0.027,…]'::vector ASC
LIMIT 10;

-- … identically for ef = 40, 60, 80, 100, 150, 200, 300 …

SET LOCAL hnsw.ef_search = 400;
SELECT "doc_id" FROM "docs_reddit"
ORDER BY "embedding" <=> '[0.011,-0.027,…]'::vector ASC
LIMIT 10;

COMMIT;
```

Mechanics: identifiers (`table`, `key`, `column`) come from config and are
always emitted quoted/escaped; the query vector is bound as a pgvector
parameter (shown inline above only for readability); `SET LOCAL
hnsw.ef_search` cannot take bind parameters, so the validated integer is
interpolated as a literal. The ground-truth `ORDER BY` carries the key
tie-break; the ANN `ORDER BY` must **not** — appending the key would defeat
the HNSW index scan.

- **Decision — exactness**: exact k-NN is obtained by disabling index scans
  inside the transaction, forcing a sequential scan with top-k sort.
  Rationale: no data export, works on the live table, matches "brute force"
  in the anchor methodology.
- **Decision — tie handling**: ground truth order is `(distance ASC, key
  ASC)`. Ties at the k boundary are broken by ascending primary key.
  Rationale: makes ground truth a deterministic function of the table
  snapshot — something the anchor never pinned down. Its `ground_truth.py`
  computes exact top-k with `torch.topk(scores, k)`; when several documents
  share the exact boundary score, which of them enters the top-k is an
  implementation detail of torch, undocumented and potentially different
  across CPU/GPU backends and versions — deterministic on one machine by
  accident, not a rule an independent implementation can follow. Exact
  distance ties require bit-identical vectors, so they are vanishingly rare
  on real embeddings, and fixing a rule for them cannot move aggregate
  results beyond the stated tolerances.
- **Decision — duplicate keys**: impossible by construction — `<key>` must be
  the table's primary key (or a unique, non-null integer column). MVP
  supports integer keys only (`int2/int4/int8`, stored as int64).
- If the table holds fewer than `k` rows the sample fails ("table smaller
  than k"). The recall denominator is never adjusted.

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

- **Decision — deterministic hash split** keyed by query content, not RNG
  shuffle: a population member is in the train set iff

  ```
  FNV1a64("s:" + split_seed + ":" + vector_hash) mod 10000
      < round(train_fraction * 10000)
  ```

  with `split_seed` (default 7) and `train_fraction` (default 0.7) from
  config, `split_seed` and `vector_hash` rendered as decimal ASCII. FNV-1a
  64: `h = 0xcbf29ce484222325`; per byte `h ^= b; h *= 0x100000001b3
  (mod 2^64)`.
  Rationale: language-portable (five lines in any language, no RNG
  compatibility problem with the Python anchor harness), order-independent,
  and content-keyed — membership is stable across rounds, resumes, and the
  rolling window, and identical vectors land on the same side by
  construction, so round-to-round output changes come from data, not
  re-shuffling.

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
  compared against the round's deduplicated population size
  (`samples.unique`). Below it the round
  publishes `status: "insufficient_samples"` with `recommended_ef: null`,
  `confidence: null`, and full sample/coverage metadata. No degraded-
  confidence emission. Rationale: an ef recommendation from a tail quantile
  with too few tail points is noise; publishing an explicit refusal keeps
  the "app stays at its conservative default until the tuner speaks" demo
  narrative honest. Demo configs simply lower the threshold.

#### Rolling window semantics

- Round range is `[round_end − W, round_end)`, half-open, with
  `round_end = align(now)` — floor to the previous storage-window boundary
  (§2.1). Examples with 600 s windows and `W = 1h`: a round at 12:07:23 has
  `round_end = 12:00` and covers exactly the six windows starting 11:00,
  11:10, … 11:50; a round at precisely 12:00:00 covers the same six — the
  window `[12:00, 12:10)` is still open and never in scope.
- **Membership is per storage window, never per record**: a part is in scope
  iff its header satisfies `window_start ≥ round_end − W` and
  `window_start + window_seconds ≤ round_end`. Only fully closed storage
  windows are consumed; record receive timestamps are not re-checked
  (the collector guarantees them by construction).
- **Late-arriving parts** (spills, delayed flushes) missed by one round are
  picked up by the next: every round re-lists all in-scope windows and
  measures any part it has no intermediates for. A round that ticks moments
  after a window closes may list that window mid-flush and see only some of
  its parts; the remainder is simply measured next round. Observation
  markers (§2.4) never gate measurement — they only classify coverage.
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
  ef_search: [20, 40, 60, 80, 100, 150, 200, 300, 400]  # REQUIRED: the sweep buckets
  train_fraction: 0.7
  split_seed: 7
  min_samples: 1000

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
  prod/superuser:                 # exact cohort name (MVP: no patterns)
    index: stockexchange
    target: queries_search_recall
  prod/reddit:
    index: reddit
    target: queries_search_recall
```

**Cohort names.** MVP: a cohort key is the exact, full cohort name — no
patterns, no prefix matching. This keeps the tuner free of cohort discovery:
it lists only the configured `cohorts/<name>/window=…` prefixes and ignores
everything else in storage, so there are no matching-precedence rules, no
discovery pass, and no "configured pattern vs concrete cohort" distinction
anywhere in the pipeline. Prefix patterns are a non-goal (§4).

**Secrets.** Decision: connection fields are plain config; the password, if
any, comes only from the environment variable named by `password_env`
(12-factor, maps directly onto Kubernetes `secretKeyRef` and local `export`).
The config loader **rejects** any inline credential — a `password` key or a
`user:pass@` userinfo in any connection value — with an error pointing at
`password_env`. Passwords never appear in config, logs, intermediates, or
outputs. Rationale: one mechanism, native to every deployment target, and a
hard validation stop is worth more trust than optional advice.

**Startup validation** (all violations are fatal):

- every cohort key is a valid cohort name per the grammar (exact names, no
  patterns) and references an existing index and target
- `0 < percentile < 1`, `0 < value ≤ 1`, `k ≥ 1`, `window ≥
  storage.window_seconds`
- ef grid present and non-empty (**required, no default** — a default grid
  would fail the `min(grid) ≥ k` rule below for larger `k`, and a surprise
  interplay between two defaults is worse than one explicit field), strictly
  increasing, `min(grid) ≥ k` for every configured target (pgvector caps
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
windows/window=<ts>/observed-<ulid>.json                  # collector (input): observation markers
measurements/<cohort>/window=<ts>/part-<ulid>.truth.parquet
measurements/<cohort>/window=<ts>/part-<ulid>.sweep.parquet
calibrations/<cohort>/round-<ts>.json                     # immutable history
calibrations/<cohort>/latest.json                         # mutable pointer copy
```

`<ts>` uses the collector's `YYYYMMDDTHHMMZ` format; the round timestamp is
`round_end`. `round-<ts>.json` is written first, then the identical bytes
overwrite `latest.json` (single atomic PUT each — the poll target for the
demo dashboard and, later, the sidecar).

#### Window observation markers (input; required collector addition)

The tuner must distinguish "the collector was up and there was simply no
traffic" from "the collector was down or started mid-window". Segment parts
cannot carry that distinction: a zero-traffic window produces no part at all,
and `first_receive` in a part header cannot separate "listening since 12:00,
first frame at 12:05" from "started listening at 12:03".

**Decision**: the collector additionally writes one small **observation
marker** per storage window it fully or partially observed, after all of
that window's parts are flushed:

```jsonc
// windows/window=<ts>/observed-<ulid>.json
{
  "format_version": 1,
  "window_start": 1783513800,   // aligned window start, unix seconds UTC
  "window_seconds": 600,
  "observed_from_us": ...,      // max(window start, listener start), unix micros
  "observed_to_us": ...         // min(window end, graceful-shutdown time), unix micros
}
```

Collector-side contract (a small, explicit collector change this spec
depends on):

- Written **after** the window's parts, at window close on the normal path,
  or at graceful shutdown for the currently open window. One small PUT —
  atomic on `LocalFileSystem` (`object_store` writes temp + rename) and on
  S3-style stores.
- Written **even when zero frames were received** — that is the point.
- Never overwritten: the ULID suffix follows the part-naming pattern, so a
  collector restart within one window produces multiple markers whose
  intervals the tuner unions.
- A crash writes no marker for the interval observed since the last window
  close; spill parts flushed before the crash remain. Coverage is therefore
  underestimated, never overestimated — conservative in the honest
  direction.

Worked example (600 s windows, collector starts 12:03, first frame 12:05):
the part has `window_start = 12:00` and `first_receive = 12:05`; the marker
says `observed_from = 12:03`, `observed_to = 12:10`. The tuner can state:
samples from 12:03–12:10 are trustworthy, and 12:00–12:03 may be missing
samples.

Tuner-side use — coverage only, never coordination:

- Per in-scope window slot: `observed_seconds = |union(marker intervals) ∩
  slot|`. Round metric: `unobserved_time_fraction = 1 − Σ observed_seconds
  / W`.
- A slot with a marker but no parts is an **observed zero-traffic** slot; a
  slot (or part of one) without marker coverage is a **collector gap**.
- Markers never gate measurement: parts found in an unmarked window are
  still measured and used — a missing marker (old collector build,
  pre-deployment history, crash) only raises `unobserved_time_fraction`.
- Markers deliberately do **not** define `round_end`. Anchoring the round to
  the wall clock keeps a collector outage visible as growing unobserved
  time; anchoring to the last marker would silently freeze the window
  instead. This decision supersedes an earlier `close_grace` heuristic.

Alternative considered and rejected: appending a `window_collection_start`
field to the segment header. It cannot represent windows with zero parts
(the common case for outages), and it duplicates per-process information
into every cohort's parts; the format's `header_len` extensibility remains
available if a per-part need appears later.

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

`part-<ulid>.truth.parquet` — one row per successfully measured distinct
vector:

| column            | type          | meaning                                          |
|-------------------|---------------|--------------------------------------------------|
| `record_index`    | int32         | first-occurrence ordinal in the part (identity)  |
| `vector_hash`     | uint64        | FNV-1a 64 of the raw f32 vector bytes (§2.2)     |
| `dup_count`       | int32         | occurrences of this vector within the part (≥ 1) |
| `receive_time_us` | int64         | from the first-occurrence segment record         |
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
               "measured": 5310,     // distinct vectors with intermediate rows
               "failed": 12,
               "unique": 5150,       // population after window-wide dedup (§2.2)
               "train": 3605, "test": 1545 },
  "dropped_frame_fraction": 0.012,   // collector-side: 1 - sum(record_count)/sum(received_frame_count)
  "coverage": {
    "unobserved_time_fraction": 0.05, // window time not covered by observation markers
    "windows_in_scope": 144,
    "windows_observed": 143,          // slots with at least one marker
    "windows_with_parts": 140
  },
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
queue-full drops are invisible downstream). The `coverage` block comes from
observation markers (§2.4): a marked slot without parts is observed
zero-traffic, not a gap; unmarked time — collector down, started mid-window,
or crashed — counts toward `unobserved_time_fraction`. `samples.failed`
carries Phase A's per-sample measurement failures (SQL errors, timeouts, bad
frames). Confidence expresses transferability only and never folds in drops,
gaps, or failures — they are reported alongside it so the consumer can judge
both independently.

### 2.5 Measure phase details

Per unmeasured part:

1. GET the `.vseam` object; parse with `vectorseam-core` (header + records).
2. Deduplicate within the part by exact vector bytes (§2.2): one
   measurement per distinct vector, identified by its first-occurrence
   `record_index`, with the occurrence count recorded as `dup_count`.
3. Apply the measurement cap over the distinct vectors: keep the vector
   with first occurrence `i` iff
   `FNV1a64("m:" + split_seed + ":" + part_ulid + ":" + i) mod 1000000 <
   floor(min(1, max_samples_per_part / distinct_count) · 1000000)` —
   deterministic, unbiased, cache-stable thinning of oversized parts.
4. For each kept vector: parse the frame (dtype must be F32; the f32 vector
   is passed as a pgvector value), run the single-sample transaction of
   §2.2 under the traffic budget (§3.1). A failed sample increments
   `failed_count`; the transaction is rolled back and the round continues.
5. Buffer rows in memory (tuner memory is deliberately unbudgeted — parts
   are bounded by the collector's 32 MiB spill cap); write
   `*.truth.parquet` then `*.sweep.parquet`.

Connections: one pool per distinct `(server, database)` with
`max_size = budget.max_concurrent_queries`. Every transaction sets
`statement_timeout` via `SET LOCAL`.

### 2.6 Aggregate phase details

1. List in-scope parts (from `cohorts/`) and intermediates (from
   `measurements/`); ignore incompatible intermediates (counted).
2. Load truth and sweep rows; the stored `recall` column is authoritative.
3. Deduplicate across the window by `vector_hash`, keeping the row with the
   smallest `(part_ulid, record_index)` (§2.2); the survivors are the
   population (`samples.unique`).
4. Split per §2.2; if `unique < min_samples` publish `insufficient_samples`
   and stop.
5. Select ef per §2.2; compute holdout quantile, `transferred`, confidence.
6. Compute `dropped_frame_fraction` from part headers and the `coverage`
   block from observation markers (§2.4).
7. Compose the round JSON (all counters from part headers, marker intervals,
   and file metadata); PUT `round-<ts>.json`, then `latest.json`.

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
- Cohort prefix/pattern matching — cohorts are configured by exact name.
- Traffic-weighted recall populations (duplicates re-weighting the
  percentile); `dup_count` in the intermediates keeps the door open.
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
distance tie exists and that query vectors are pairwise distinct (so
deduplication is a no-op in the anchor comparison). "F-agg" means
hand-crafted intermediates (parquet pairs and segment headers) fed to Phase B
only — no database.

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
  rows containing 5 hits → 0.5 (short results penalized).
- **B2 tie-break** (F-pg): rows 7 and 9 given identical vectors that
  straddle the k boundary → `gt_keys` contains 7, not 9, on repeated runs.
- **B3 quantile**: recalls `[0.5, 0.7, 0.9, 1.0]`, `percentile: 0.95` →
  compliance quantile = 0.53 exactly (h = 0.15, type-7 linear); n = 1 →
  the single value.
- **B4 split**: membership computed by an independent five-line FNV-1a
  reference matches the tuner for a fixed list of vector hashes; observed
  train fraction within 0.7 ± 0.03 for n = 10⁴ distinct vectors; membership
  unchanged after simulated resume and after the surviving occurrence moves
  to a different part (same vector → same split).
- **B5 selection**: train quantiles `{10:0.62, 20:0.85, 40:0.91, 80:0.93,
  160:0.95}`, value 0.9 → `recommended_ef = 40`, `status = "ok"`.
- **B6 target unmet**: value 0.99 with B5's quantiles → `recommended_ef =
  160`, `status = "target_unmet"`, confidence and `test_quantile_recall`
  still present.
- **B7 min samples**: `min_samples = 1000`; a unique population of 999 →
  `status = "insufficient_samples"`, `recommended_ef = null`,
  `confidence = null`, sample counts still reported; a unique population of
  1000 → a recommendation is emitted.
- **B8 window membership and coverage**: storage window 600 s, W = 3600 s, a
  round ticking at 12:07 → `round_end = 12:00` and exactly the six windows
  starting 11:00–11:50 are in scope; parts at 10:50 and 12:00 are excluded.
  With full markers on five slots and the 11:20 slot marked only from 11:23
  (420 s observed) → `unobserved_time_fraction = 180/3600 = 0.05 ± 1e-12`; a
  marked slot with zero parts contributes zero samples and zero unobserved
  time.
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
- **B12 markers never gate measurement**: a window containing spill parts
  but no marker → its samples are measured and included in the population,
  while the slot's unmarked time still counts toward
  `unobserved_time_fraction`; two markers in one slot (collector restart)
  have their intervals unioned, not double-counted.
- **B13 deduplication**: a part containing the same vector at record
  indexes 3, 5, and 9 → exactly one truth row (`record_index = 3`,
  `dup_count = 3`) and one grid of sweep rows for it; the same vector
  appearing again in a second in-scope part → `samples.unique` counts it
  once, the survivor coming from the lexicographically smaller
  `(part_ulid, record_index)`; duplicates never land on opposite sides of
  the train/holdout split.

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
- **C4 empty round**: zero in-scope parts and zero markers →
  `insufficient_samples`, `samples.available = 0`,
  `unobserved_time_fraction = 1.0`.
- **C5 config validation**: each of — inline `password` key, `user:pass@`
  in `server`, ef grid containing a value < k or > 1000 or non-increasing,
  cohort with unknown index/target, `percentile: 1.0`, `window <
  storage.window_seconds` — fails startup with a distinct error; the
  `password_env` error message names the expected mechanism.
- **C6 failed samples**: a table with fewer than k rows → every sample fails
  with "table smaller than k", `samples.failed = samples.available`, round
  publishes `insufficient_samples`, process keeps running.
- **C7 snapshot semantics** (F-pg; requires pausing the tuner's connection
  between statements, e.g. via a test proxy): after a sample's ground-truth
  statement returns and before its sweep statements run, a second connection
  inserts a "poison" row strictly closer to the query vector than every
  existing row → the poison key appears in none of that sample's
  `returned_keys` (ground truth and sweep shared one snapshot), while a
  later sample of the same round has the poison key in its `gt_keys`
  (cross-sample drift is real and accepted, per §2.2).
- **C8 Phase B reproducibility**: running aggregation twice over identical
  intermediates and config yields byte-identical round JSON except
  `computed_at` — the accepted drift lives entirely in Phase A; everything
  downstream of the intermediates is deterministic.

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
