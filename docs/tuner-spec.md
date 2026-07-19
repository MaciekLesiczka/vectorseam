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
- **In scope**: a segment part is in scope for a round iff its storage
  window lies fully inside that round's range `[round_end − W, round_end)`;
  the formal membership rule is in §2.2 (rolling window semantics).
  "In-scope windows / intermediates" follow the same rule.
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
   storage window fully inside the round range.
3. Diff the listed parts against already-measured parts under
   `measurements/<cohort>/…`. For each unmeasured part: fetch and parse the
   `.vseam` part, then for each kept sample run the per-sample database
   transaction — exact ground truth plus one ANN query per value of the
   `calibration.ef_search` grid (required config, §2.3); the full SQL is
   spelled out in §2.2 — and finally write the part's `truth` and `sweep`
   parquet files (in that order).

**Phase B — aggregate** (pure function of the intermediates and previous
published state, no database):

4. Read all in-scope, metadata-compatible intermediates, split samples into
   train/holdout, compute per-ef compliance quantiles, select the recommended
   ef, compute transfer and confidence, derive the `effective` recommendation
   from the previous `latest.json` (§2.2), and publish the round result
   JSON: `calibrations/<cohort>/round-<ts>.json` plus an overwrite of
   `calibrations/<cohort>/latest.json`.

Server-confirmed per-sample errors degrade Phase A (failed samples are
counted, not retried within the round) but do not block Phase B. Database
unavailability also does not block a fully cached cohort, but if a pending
part needs the unavailable connection the cohort publishes nothing and leaves
that part untouched for the next round (§3.2). Rationale: a stale-but-honest
cached output is useful; a fresh-looking output that silently omits a pending
part is not.

### 2.2 Estimator semantics

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
- **Decision — measurement-time dedup is scoped to one part, never wider.**
  Both wider scopes fail. Deduplicating across storage windows breaks the
  rolling window: when the earlier window ages out of scope, the duplicate
  that is still in scope would be left with no measurement. Deduplicating
  across the parts of one storage window would make a part's files depend
  on sibling parts, which can be listed in any order or appear late
  (spills, crash recovery) — and a part is the unit of measurement and
  crash recovery precisely because its two files are computable from that
  part alone (§2.4). So a duplicate recurring across parts is measured once
  per part, and Phase B discards the extras. Cheap in practice: a window
  normally has exactly one part (spill parts appear only under memory
  pressure), so part scope and storage-window scope usually coincide.
- A sample's storage identity is `(part_ulid, record_index)`,
  `record_index` 0-based within the part; it keys the intermediates, so
  resumes and re-listings can never double-count. The
  train/holdout split is keyed by the vector hash instead (see below): a
  duplicate can never straddle the split, and membership survives the
  rolling window even when the surviving occurrence changes as old parts
  age out.
- Sample-local frame failures (unsupported dtype or dimension mismatch) and
  database failures confirmed while the connection remains usable
  (server-confirmed SQL error or statement timeout) are excluded from the
  population and reported as a count. Connection-level failures instead
  leave the whole current part unmeasured (§3.2). Only dtype `F32` frames are
  supported.

#### Ground truth and ef sweep — one transaction per sample

Every measured sample runs exactly one database transaction containing the
exact ground-truth query followed by one ANN query per value of the
`calibration.ef_search` grid, ascending. The grid is required configuration
(§2.3); the published benchmark grid `[10, 20, 40, 60, 80, 100, 150, 200,
300, 400]` is the recommended starting point. Spelled out in full for an
index `{table: docs_reddit, key: doc_id, column: embedding}` with `k = 10`
and that grid:

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
- If the ground-truth query returns fewer than `k` rows, the table itself
  has fewer than `k` visible rows — a deployment problem, not per-sample
  noise, and it would fail identically for every remaining sample in the
  cohort. Detected on the first affected sample: the tuner aborts Phase A
  for this cohort for the rest of the round (no further scans), publishes
  the round with `status: "insufficient_samples"` and an `error` string
  naming the condition, and continues with the other cohorts. The recall
  denominator is never adjusted for small tables. Phase A passes this
  condition to Phase B as a typed cohort abort. It takes precedence over
  cached population size: even if prior compatible intermediates still meet
  `min_samples`, selection is skipped and the insufficient output shape is
  forced.

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

MVP supports cosine distance (`<=>`) only. There is deliberately no config
field for it — a field with exactly one legal value is dead config; other
pgvector operators are an additive change (§4.2).

#### recall@k

```
recall = |set(returned_keys) ∩ set(gt_keys)| / k
```

Set semantics over distinct keys; the denominator is always the target's
`k`, matching the anchor. A short ANN result (fewer than `k` rows) is scored
as-is, so every missing position counts as a miss. Short results cannot be
caused by concurrent writes — the REPEATABLE READ snapshot makes ground
truth and the sweep see the same rows — and `ef < k` is ruled out by config
validation. They can still occur: an HNSW scan spends its `ef_search`
candidate budget on index entries, and entries pointing at heap rows
invisible to the snapshot (deleted or updated, not yet vacuumed) consume
budget without producing rows.

#### Train/holdout split

- **Decision — deterministic hash split** keyed by query content, not RNG
  shuffle: a population member is in the train set iff

  ```
  FNV1a64("s:" + split_seed + ":" + vector_hash) mod 10000
      < round(train_fraction * 10000)
  ```

  with `split_seed` (default 7) and `train_fraction` (default 0.7) from
  config, `split_seed` and `vector_hash` rendered as decimal ASCII.
  Rationale: order-independent and content-keyed — membership is stable
  across rounds, resumes, and the rolling window; identical vectors land on
  the same side by construction; round-to-round output changes come from
  data, not re-shuffling.
- **FNV-1a 64** (Fowler–Noll–Vo, 64-bit, variant 1a) is the hash used here
  and for `vector_hash`. Chosen because the split must produce identical
  values in the Rust tuner and the Python acceptance harness, and FNV-1a is
  small enough to implement inline in both — no library dependency, no
  cross-language RNG compatibility questions. It is not cryptographic;
  nothing here needs it to be. Definition (all arithmetic on unsigned
  64-bit integers, multiplication wrapping on overflow) and reference
  values any implementation must reproduce:

  ```
  fn fnv1a64(bytes) -> u64:
      h = 0xcbf29ce484222325                # fixed start value ("offset basis")
      for each byte b in bytes:
          h = h XOR b
          h = (h * 0x100000001b3) mod 2^64  # multiply by the "FNV prime", keep low 64 bits
      return h

  fnv1a64("")       == 0xcbf29ce484222325
  fnv1a64("a")      == 0xaf63dc4c8601ec8c
  fnv1a64("foobar") == 0x85944171f73967e8
  ```

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
  `confidence: null`, `transferred: null`, `train_quantile_recall: null`,
  `test_quantile_recall: null`, and full sample/coverage metadata. The
  full-population `per_ef` summaries are still computed when the population is
  non-empty; for an empty population `per_ef` is `[]`. No degraded-confidence
  emission. Rationale: an ef recommendation from a tail quantile with too few
  tail points is noise; publishing an explicit refusal keeps the "app stays at
  its conservative default until the tuner speaks" demo narrative honest.
  Demo configs simply lower the threshold. The refusal is scoped to *this
  round's* fields: a previously calibrated ef is still served through the
  `effective` block (see "Effective recommendation" below), so the
  conservative default applies only until the tuner first speaks, not again
  on every transient gap.
- The same `insufficient_samples` output is published if the realized train or
  holdout split is empty. This is checked after splitting even when
  `samples.unique >= min_samples`, because type-7 quantiles are undefined for
  empty input. Configuration also rejects a `train_fraction` whose rounded
  10,000-bucket threshold is 0 or 10,000, preventing a guaranteed-empty side.

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
  its parts; the remainder is simply measured next round.
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

#### Effective recommendation (last known good)

`recommended_ef` describes what *this round* computed, and stays honestly
`null` when the round refuses selection. The client-facing signal is a
separate `effective` block — the ef a consumer should apply right now —
published in every round record (schema in §2.4):

- **Decision — carry the last known good recommendation.** A round whose
  `recommended_ef` is non-null (`ok` or `target_unmet`) sets `effective`
  from itself (`carried: false`). An `insufficient_samples` round copies the
  previous round's `effective` unchanged (`carried: true`). Rationale: an
  application that has been applying a calibrated ef must not be knocked
  back to its conservative default by one sampling blip or table incident —
  and the fix cannot live in clients, because a freshly started client whose
  first poll lands on an insufficient round has nothing to remember.
  `target_unmet` deliberately overrides an older `ok`: max(grid) is the
  newest honest signal and the most protective actionable setting.
- **The carry source is storage, never memory.** Before publishing, the
  round GETs the cohort's current `latest.json` (one GET per cohort per
  round) and takes its `effective` block as the carry source, so the chain
  survives restarts. Failure handling distinguishes three cases. Not-found
  is the normal bootstrap state: no carry source, no warning. A record that
  is malformed — or written before this field existed — is a persistent
  condition: no carry source, a warning; treating it as fatal would wedge
  the cohort, since `latest.json` is only rewritten by a successful publish.
  Any other GET failure is transient storage trouble and aborts the cohort
  round without publishing, exactly like every other storage failure (§3.2):
  publishing with an absent carry source would durably replace the last
  known good recommendation with `null` — the regression this block exists
  to prevent — while aborting preserves the stored chain for the next tick.
- **Config-fingerprint invalidation.** The carry source is used only when
  the previous round record's `cohort`, `index`, `ef_grid`, and `target`
  fields (`k`, `value`, `percentile`) all equal the current configuration;
  otherwise `effective` resets to `null`. Same philosophy as §2.4
  intermediate compatibility: never serve a recommendation calibrated for a
  different target or index. Carrying preserves this check inductively —
  every published `effective` is consistent with its own round's top-level
  fields — so validating against the previous round's fields alone is
  sufficient.
- **No expiry.** The tuner never silently decays `effective` to `null`;
  staleness is visible through `source_round`, and any maximum-age policy
  belongs to the consumer. Before the first successful round `effective` is
  `null` — the app stays at its conservative default until the tuner speaks,
  but never re-enters that state on a transient gap.

#### Determinism summary

Phase B is a pure, order-independent, bit-reproducible function of
(intermediates, config, previous `effective` carry source). Phase A is
deterministic per sample given the table snapshot its transaction ran under,
and not reproducible afterward; this is recorded, accepted drift.

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
  statement_timeout: 5s
  client_timeout: 10s

data_sources:
  primary:
    server: localhost:5432
    database: postgres
    user: postgres
    password_env: SEAM_PG_PRIMARY         # optional; see secrets

  analytics:
    server: analytics.internal:5432
    database: vectors
    user: vectorseam
    password_env: SEAM_PG_ANALYTICS

indexes:
  stockexchange:
    data_source: primary
    table: docs_stackexchange
    key: doc_id
    column: embedding

  reddit:
    data_source: analytics
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
anywhere in the pipeline. Prefix patterns are a recorded corner cut (§4.2).

**Data sources.** Database connection identity is a top-level concern.
Each named data source owns `server`, `database`, `user`, and optional
`password_env`; each index references exactly one data source and owns only
`table`, `key`, and `column`. Distinct data sources must have distinct
`(server, database)` pairs. Defining the same pair twice is a startup error,
regardless of user or password configuration. This makes the one-connection,
one-pacer ownership in §2.5 unambiguous.

**Secrets.** Decision: data-source connection fields are plain config; the
password, if any, comes only from the environment variable named by `password_env`
(the standard container pattern: Kubernetes injects secrets as environment
variables via `secretKeyRef`; local development uses `export`).
The config loader **rejects** any inline credential — a `password` key or a
`user:pass@` userinfo in any data-source connection value — with an error
pointing at `password_env`. When `password_env` is configured, the named
environment variable must exist at startup; when it is omitted, no password
environment variable is required. Passwords never appear in config, logs,
intermediates, or outputs. Rationale: one mechanism, native to every
deployment target, and a hard validation stop is worth more trust than
optional advice.

**Startup validation** (all violations are fatal):

- every cohort key is a valid cohort name per the grammar (exact names, no
  patterns) and references an existing index and target; every index
  references an existing data source
- every data source has a unique `(server, database)` pair; duplicate pairs
  are rejected even when their users or `password_env` values differ
- `storage.window_seconds` is a positive multiple of 60; `0 < percentile <
  1`, `0 < value ≤ 1`, `k ≥ 1`; every target `window` is at least one storage
  window and an exact multiple of `storage.window_seconds`
- ef grid present and non-empty (**required, no default** — a default grid
  would fail the `min(grid) ≥ k` rule below for larger `k`, and a surprise
  interplay between two defaults is worse than one explicit field), strictly
  increasing, `min(grid) ≥ k` for every configured target (pgvector caps
  results at `ef_search`, so `ef < k` can never satisfy the target),
  `max(grid) ≤ 1000` (pgvector bound)
- `0 < train_fraction < 1`, `min_samples ≥ 100`, `0 < db_share ≤ 1`,
  `statement_timeout > 0`, `client_timeout > 0`; additionally,
  `round(train_fraction * 10000)` must be in `1..=9999`
- `table`, `column`, `key` are valid PostgreSQL quoted/delimited identifiers:
  non-empty UTF-8, at most 63 bytes, and containing no NUL. They are always
  emitted quoted with embedded double quotes escaped.
- no inline credentials; every configured `password_env` exists at startup
  (see secrets)

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
`index` (config name), `table`, `column`, `key`, `k`, `ef_grid`
(comma-joined), `failed_count`, `measured_count`, `computed_at_us`.
Aggregation skips (and reports) files whose `k`, `index`, `table`,
`column`, `key`, `ef_grid`, or `format_version` don't match the current
config — the measure phase then re-measures those parts. This is how config
edits across restarts stay safe without runtime reconfiguration.
The reader rejects a pair when metadata `measured_count` differs from the
actual truth-row count; counters cannot silently disagree with the decoded
population.

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
  "ef_grid": [20, 40, 60, 80, 100, 150, 200, 300, 400],
  "status": "ok",                    // "ok" | "target_unmet" | "insufficient_samples"
  "error": null,                     // non-null Phase A abort forces insufficient_samples
  "recommended_ef": 200,             // null when insufficient_samples
  "confidence": 0.971,               // null when insufficient_samples
  "transferred": true,               // test quantile >= value; null when insufficient_samples
  "train_quantile_recall": 0.90,     // null when insufficient_samples
  "test_quantile_recall": 0.90,      // null when insufficient_samples
  "effective": {                     // last known good — what a client applies now (§2.2);
                                     // null only before any round has ever recommended
    "recommended_ef": 200,           // never null inside a non-null block
    "confidence": 0.971,             // confidence as of source_round
    "source_round": "2026-07-15T12:00:00Z",  // round_end that computed it
    "carried": false                 // true when copied from a previous round
  },
  "samples": { "available": 5400,    // sum(record_count) of in-scope parts
               "measured": 5310,     // distinct vectors with intermediate rows
               "failed": 12,
               "unique": 5150,       // population after window-wide dedup (§2.2)
               "train": 3605, "test": 1545 },
  "dropped_frame_fraction": 0.012,   // collector-side: 1 - sum(record_count)/sum(received_frame_count)
  "coverage": {
    "empty_window_fraction": 0.0417,  // in-scope storage windows with no parts / windows in scope
    "windows_in_scope": 144,
    "windows_with_parts": 138
  },
  "parts_used": 144,
  "incompatible_parts": 0,           // intermediates skipped for config mismatch
  "per_ef": [                        // full-population summary; [] for an empty population
    { "ef": 10, "quantile_recall": 0.55, "mean_recall": 0.71, "latency_p50_ms": 0.4 },
    { "ef": 20, "quantile_recall": 0.70, "mean_recall": 0.82, "latency_p50_ms": 0.6 }
    // ...
  ]
}
```

Notes: `dropped_frame_fraction` covers collector-side drops only (SDK
queue-full drops are invisible downstream). The `coverage` block is derived
purely from the part listing; an empty window can mean either zero traffic
or a down/late collector, and the MVP cannot tell the two apart — a
recorded corner cut (§4.2). `samples.failed` carries Phase A's per-sample
measurement failures (SQL errors, timeouts, bad frames). Confidence
expresses transferability only and never folds in drops, gaps, or failures —
they are reported alongside it so the consumer can judge both independently.
When the in-scope headers have `sum(received_frame_count) = 0`, including an
empty round, `dropped_frame_fraction` is defined as `0.0`; absence is reported
separately by coverage rather than represented as a drop.
`effective` is an additive extension: `format_version` stays 1, and a round
record without the field (written before it existed) is read as
`effective: null` when used as a carry source (§2.2). Because every round
record embeds the `effective` block, the immutable `round-<ts>.json` history
also records what was in effect at each point in time.

### 2.5 Measure phase details

Per unmeasured part:

1. GET the `.vseam` object; parse with `vectorseam-core` (header + records).
   A listed `*.vseam` object whose basename is not canonical
   `part-<ULID>.vseam` is skipped with a warning. A malformed canonical part
   remains fail-visible and aborts the cohort round (§3.2).
2. Deduplicate within the part by exact vector bytes (§2.2): one
   measurement per distinct vector, identified by its first-occurrence
   `record_index`, with the occurrence count recorded as `dup_count`.
3. For each distinct vector: parse the frame (dtype must be F32; the f32
   vector is passed as a pgvector value), run the single-sample transaction
   of §2.2 under the traffic budget (§3.1). A server-confirmed failure while
   the connection remains usable increments `failed_count`; the transaction
   is rolled back and the round continues — except "table smaller than k",
   which aborts Phase A for the whole cohort (§2.2). Connection-level failure
   handling is defined in §3.2 and never produces a durable failed part.
4. Buffer rows in memory (tuner memory is deliberately unbudgeted — parts
   are bounded by the collector's 32 MiB spill cap); write
   `*.truth.parquet` then `*.sweep.parquet`.

Connections: one long-lived connection and one duty-cycle pacer per data
source. Data-source pair uniqueness means this is exactly one connection and
pacer per `(server, database)`. All statements for that PostgreSQL pair are
serialized. The pacer sleeps only *between* sample transactions, never inside
one (§3.1): a sample's statements run back-to-back so its `REPEATABLE READ`
snapshot is held for the minimum possible time. Every transaction sets
`statement_timeout` via `SET LOCAL`. Unavailable or closed connections are
reconnected once at the start of each round; healthy connections remain
long-lived, and no reconnect is attempted mid-round.

### 2.6 Aggregate phase details

1. List in-scope parts (from `cohorts/`) and intermediates (from
   `measurements/`); ignore incompatible intermediates (counted).
2. Load truth and sweep rows; the stored `recall` column is authoritative.
3. Deduplicate across the window by `vector_hash`, keeping the row with the
   smallest `(part_ulid, record_index)` (§2.2); the survivors are the
   population (`samples.unique`).
4. Split per §2.2; if Phase A supplied a table-smaller-than-k abort,
   `unique < min_samples`, or either realized split is empty, publish
   `insufficient_samples` per §2.2 and stop selection. The Phase A abort's
   error string is copied to the round. Full-population `per_ef` summaries are
   still included for a non-empty cached population.
5. Select ef per §2.2; compute holdout quantile, `transferred`, confidence.
6. Compute `dropped_frame_fraction` from part headers and the `coverage`
   block from the window/part listing.
7. GET the cohort's current `latest.json` and derive this round's
   `effective` block per §2.2 (not-found, malformed, pre-`effective`, or
   fingerprint-incompatible → no carry source; any other GET failure aborts
   the cohort round per §3.2).
8. Compose the round JSON (all counters from part headers and file
   metadata); PUT `round-<ts>.json`, then `latest.json`.

Publishing the same `round_end` twice (interval shorter than the storage
window, or restart) overwrites the same key with a fresher computation —
idempotent by design. This includes `effective`: a recommending round
re-derives the block from itself, and an insufficient round re-carries the
same block it already published, so republication never changes or loses it.

## 3. Non-functional requirements

### 3.1 Database traffic control

Goal: a skeptical operator can read three numbers from the config and bound
the tuner's worst-case database impact, PlanetScale-traffic-control style
(budgets per workload slice), plus the client's maximum wait for a database
operation, without any server-side installation:

> One backend per `(server, database)`, busy at most `db_share` of wall-clock
> time; server-side statement execution is capped by `statement_timeout`,
> while connection attempts and client protocol operations are capped by
> `client_timeout`.

Mechanisms, in priority order (P0 ships in the MVP; the human owner arbitrates
anything beyond):

- **P0 — serialized connection and duty-cycle pacing** (the "server share"
  idea): one connection and one pacer per `(server, database)`. The paced
  unit is one whole sample transaction (`BEGIN` through `COMMIT`/`ROLLBACK`):
  after a transaction observed to take `t` of wall time, the pacer sleeps
  `t · (1 − db_share)/db_share` before starting the next one. The pacer never
  sleeps inside an open transaction — statements within one sample run
  back-to-back. **Decision — pace between transactions, not statements**:
  sleeping mid-transaction would stretch how long the `REPEATABLE READ`
  snapshot is held (delaying vacuum's xmin horizon for the whole database),
  buying pacing granularity at the cost of MVCC health; the busy-fraction
  bound is identical either way, and each statement stays individually capped
  by `statement_timeout`. The accepted trade-off is burstier load: one
  transaction's statements arrive back-to-back. Expensive exact scans
  automatically stretch the pacing, so the bound holds regardless of table
  size. Default `db_share: 0.10`.
- **P0 — per-statement timeout**: `SET LOCAL statement_timeout` (default 5 s)
  on every transaction; a timed-out sample is a failed sample, never a retry
  storm.
- **P0 — client operation timeout**: `client_timeout` (default 10 s) bounds
  every connection attempt and each PostgreSQL protocol operation, including
  `BEGIN`, settings, queries, and `COMMIT`/`ROLLBACK`. Expiry abandons the
  connection instead of reusing uncertain protocol state; the current part
  remains unmeasured and is retried after reconnect on the next round.
- **P1 — plan mode**: `seam plan --config …` prints, per cohort, the pending
  part count, estimated statements, and worst-case busy seconds for the next
  round, then exits without touching the database (PlanetScale's "warn
  mode", adapted to a batch tool).
- **P2 — adaptive load signals**: pause/back off on `pg_stat_activity`
  saturation or replication lag. Explicitly out of the MVP; recorded so the
  priority call is visible.

There is deliberately no per-round sample cap in the MVP (§4.2): every
in-scope sample is measured. If a cohort's sustained sample flow exceeds
what the duty-cycle budget can measure per interval — roughly when
`samples_per_interval · (t_scan + Σ t_ann) > db_share · interval` — rounds
fall behind and the published output goes stale, while the database stays
protected by the pacer. The SDK's adaptive sampling (default ~1 sample/s per
cohort per instance) bounds the inflow in practice.

### 3.2 Fault tolerance

- Every unit of progress is an immutable object PUT (parquet pair, round
  JSON); `latest.json` is the only overwritten key. No local state, no WAL,
  no database writes: after a crash the tuner restarts, re-lists storage,
  and resumes with at most one part of redone work (§2.4).
- Server-confirmed per-sample failures on a still-usable connection are
  counted in `failed_count` and reported in the round. PostgreSQL statement
  timeout is in this class.
- Startup connection failure, a closed `tokio-postgres` connection, client
  operation timeout, or failed transaction cleanup is connection-level. If
  one occurs while measuring a part, the tuner abandons that connection,
  discards all buffered rows for the in-progress part, writes neither
  intermediate, and aborts that cohort without publishing. Other cohorts
  sharing the source issue no database work for the rest of the round,
  although a fully cached cohort may still aggregate and publish as in C2.
  At the next round start the tuner makes one reconnect attempt for each
  unavailable/closed data source; successful reconnect makes the untouched
  source part naturally retryable.
- Any storage listing or GET failure, or a source-segment parse failure, aborts
  that cohort's round without publishing and logs the object key; the next tick
  retries naturally. This includes the `effective` carry-source GET of
  `latest.json`; only its not-found and malformed-content cases degrade to an
  absent carry source instead (§2.2).
  A malformed intermediate pair is treated as incomplete and remeasured from
  scratch, overwriting truth then sweep. The round schema deliberately has no
  failed-part counter because publishing from a source segment whose header
  cannot be decoded would make availability, drop, and coverage accounting
  unknowable.
- Graceful shutdown (SIGTERM/SIGINT): cancellation interrupts a duty-cycle
  cooldown before the next transaction starts. Once a transaction starts it
  is awaited through `COMMIT`/`ROLLBACK`; it is never dropped mid-flight.
  Remaining work is skipped and no partial round is published. Nothing is
  lost — the next run redoes only the unfinished part.

### 3.3 Implementation constraints

- Rust, in this workspace; simple abstractions, new concepts introduced
  deliberately (the two-phase measure/aggregate split above is the intended
  seam between database glue and the semantically rich estimator, which must
  stay a pure function).
- Proposed crates (guidance, not contract): `tokio` + `tokio-postgres` +
  `pgvector` (Vector type);
  `arrow`/`parquet` (arrow-rs, zstd) for intermediates; `object_store`
  (already in workspace) for all storage IO; `serde` + a maintained YAML
  parser + `humantime-serde` for durations; `statrs` (or `puruspe`) for the
  regularized incomplete beta; `clap`, `tracing`, `thiserror`/`anyhow`,
  `ulid` as in the collector. FNV-1a is 5 lines, no dependency.
- Tuner process memory is intentionally unbudgeted (isolated component);
  database traffic is the only guarded resource.

## 4. Non-goals and MVP corner cuts

### 4.1 Non-goals

- Runtime configuration changes (target edits require re-measurement; the
  config-fingerprint check in §2.4 handles restarts instead).
- Tight tuner memory budgets; strict realtime (demo compresses time via
  config knobs, production tolerates ≥ 10 min staleness).
- APIs, dashboards, metrics endpoints — file exchange only.
- SDK/collector consuming the recommendation; central sampling directives.
- Full SDK→collector→tuner end-to-end integration tests (later milestone).
- Data retention/compaction of segments, intermediates, or rounds.
- Non-pgvector backends.
- Multi-instance coordination or leader election (exactly one tuner per
  storage root).

### 4.2 Deliberate MVP corner cuts (revisit as improvements)

Corners cut knowingly to ship the MVP, each with its consequence, so they
can come back as prioritized improvements:

- **No per-round measurement cap** — every in-scope sample is measured.
  Consequence: sustained sample flow beyond the duty-cycle budget makes
  rounds fall behind and outputs go stale; the database stays protected
  (§3.1). Revisit with a deterministic per-part cap if staleness bites.
- **Source headers require full object reads.** Every round currently GETs
  and parses every in-scope `.vseam` object to validate its header, including
  already measured parts; pending parts are fetched and parsed a second time.
  With the default 24-hour window this can mean roughly 144 objects per
  cohort every 10 minutes, each up to the collector's 32 MiB spill cap.
  Revisit first with a header-only range GET and then, if needed, an in-memory
  header cache keyed by canonical part ULID.
- **No observation-coverage signal.** An in-scope storage window with no
  parts can mean zero traffic or a down/late collector; the tuner cannot
  tell them apart and reports both as `empty_window_fraction`. Future
  option (small collector change): `CohortState` records its creation
  time — every flush replaces the state, so each part header would
  naturally carry the observation range of its window. Example: collector
  starts 12:03, first frame 12:05 → `window_start = 12:00`, collection
  start 12:03: samples from 12:03–12:10 are trustworthy, 12:00–12:03 may be
  missing. Zero-traffic windows would still leave no trace; covering those
  needs a per-window marker object.
- **Ground truth runs only in-database.** Alternative for scan-averse
  deployments: bulk-export the vector column once per round and brute-force
  in the tuner — trades per-sample scan load for one big read, network,
  tuner memory, and staleness of the exported copy.
- **Cosine distance only.** Other pgvector operators (`<->`, `<#>`) are an
  additive config-plus-operator-string change; no field exists until then.
- **Integer primary keys only** (`int2/int4/int8`). Text/UUID keys need a
  polymorphic key column in the intermediates.
- **Exact cohort names only.** Prefix patterns need matching-precedence
  rules and a cohort discovery pass (§2.3).
- **Population is distinct query vectors, not traffic-weighted queries.**
  `dup_count` in the intermediates allows re-weighting later without
  re-measurement (§2.2).
- **One shared traffic policy.** Every data source uses the same
  top-level `db_share`, `statement_timeout`, and `client_timeout`;
  per-data-source overrides are deferred until operators demonstrate a need.
- **No database connection pool or concurrent statements per PostgreSQL
  instance.** Each `(server, database)` pair has one serialized connection and
  pacer. Consequence: independent samples cannot use parallel database
  capacity even when the server could tolerate it; revisit with explicitly
  defined per-connection pacing before adding more connections.
- **No adaptive load signals** (`pg_stat_activity`, replication lag) —
  §3.1 P2.

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
`[10, 20, 40, 80, 160]`, `percentile: 0.90`, `value: 0.8`, and identical
train/holdout membership (both sides computed by the §2.2 hash split — the
anchor harness reuses the published pipeline's recall/percentile/selection
code with that split; numpy RNG shuffles are not part of the contract):

- **A1** For ≥ 99% of (query, ef) pairs, tuner `recall` equals the anchor's
  recall exactly; disagreements are only where ground truth differs (torch
  float math vs pgvector scan near-ties).
- **A2** Per-ef full-population mean recall: |tuner − anchor| ≤ 0.005.
- **A3** Per-ef train compliance quantile (`percentile: 0.90`,
  `value: 0.8`): |tuner − anchor `np.percentile(_, 10)`| ≤ 0.01.
- **A4** `recommended_ef` identical to the anchor's
  "min ef with train p10 ≥ 0.8", and equal to `80`.
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
- **B4 split**: the tuner's FNV-1a reproduces the three reference values in
  §2.2; membership computed by an independent five-line FNV-1a
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
  starting 11:00–11:50 are in scope; parts at 10:50 and 12:00 are excluded;
  with no parts in the 11:20 slot,
  `empty_window_fraction = 1/6 ± 1e-12`.
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
- **B12 deduplication**: a part containing the same vector at record
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
  work only; process exits 0 at shutdown. With a pending part, a
  connection-level failure writes no intermediate and publishes no cohort
  round; after the data source becomes reachable, the next round reconnects
  once, measures that same part, and publishes it with no durable outage
  failures.
- **C3 config fingerprint**: intermediates written with k = 10 are ignored
  and re-measured after k changes to 20; `incompatible_parts` > 0 in the
  next round.
- **C4 empty round**: zero in-scope parts → `insufficient_samples`,
  `samples.available = 0`, `empty_window_fraction = 1.0`.
- **C5 config validation**: each of — inline `password` key, `user:pass@`
  in `server`, configured but missing `password_env`, duplicate
  `(server, database)` data-source pair, index with unknown data source, ef
  grid containing a value < k or > 1000 or non-increasing, cohort with
  unknown index/target, `percentile: 1.0`, `window <
  storage.window_seconds`, `client_timeout: 0s` — fails startup with a
  distinct error; the
  credential errors name the expected `password_env` mechanism.
- **C6 table smaller than k**: a cohort whose table has fewer than k rows →
  Phase A aborts for that cohort after the first ground-truth result (at
  most one exact scan issued, no sweep statements), the round publishes
  `insufficient_samples` with a non-null `error` even when cached compatible
  intermediates meet `min_samples`, and the other cohorts' rounds proceed
  normally.
- **C7 snapshot semantics** (F-pg; requires pausing the tuner's connection
  between statements, e.g. via a test proxy): after a sample's ground-truth
  statement returns and before its sweep statements run, a second connection
  inserts a "poison" row strictly closer to the query vector than every
  existing row → the poison key appears in none of that sample's
  `returned_keys` (ground truth and sweep shared one snapshot), while a
  later sample of the same round has the poison key in its `gt_keys`
  (cross-sample drift is real and accepted, per §2.2).
- **C8 Phase B reproducibility**: running aggregation twice over identical
  intermediates, config, and previous-`effective` carry source yields
  byte-identical round JSON except `computed_at` — the accepted drift lives
  entirely in Phase A; everything downstream of the intermediates is
  deterministic.

### D. Resource ceilings (metric: database busy wall-time duty cycle over
paced units — whole sample transactions — measured at the tuner's database
client)

- **D1 duty cycle**: `db_share = 0.20` against instrumented database work
  taking ~50 ms per paced unit, ≥ 50 units → total elapsed ≥ 0.95 · (total
  busy time / 0.20); equivalently the busy fraction over the run ≤ 0.21.
  No pacer sleep occurs inside a unit.
- **D3 statement timeout**: `statement_timeout = 1ms` on F-pg → every sample
  fails within the round (no retries, counted), the round completes, and no
  tuner statement remains running server-side afterward.

### E. Effective recommendation (no database: hand-crafted intermediates or
mock-measured local store)

- **E1 carry on insufficient**: an `ok` round selecting ef `X`, followed by
  an `insufficient_samples` round over the same storage → the second round
  publishes `recommended_ef: null` but `effective = {recommended_ef: X,
  carried: true, source_round: <first round_end>}`, with `confidence` equal
  to the first round's, in both `round-<ts>.json` and `latest.json`;
  republishing the insufficient round's `round_end` again yields the
  identical `effective` block.
- **E2 newest honest signal wins**: an `ok` round at ef `X` followed by a
  `target_unmet` round → `effective.recommended_ef = max(grid)` with
  `carried: false`; a subsequent `insufficient_samples` round carries the
  `target_unmet` value, not the older `ok` one.
- **E3 carry survives restart**: publish an `ok` round, discard all
  in-memory state (a fresh pipeline/tuner instance over the same store),
  run an `insufficient_samples` round → `effective` still carries the first
  round's recommendation; storage is the only carry source.
- **E4 fingerprint reset**: an `ok` round published under `k = 10`, then the
  target changes to `k = 20` and the next round is `insufficient_samples` →
  `effective` is `null`, never a recommendation calibrated for a different
  target; an `ef_grid` change behaves identically.
- **E5 bootstrap and malformed carry source**: with no `latest.json`, an
  insufficient first round publishes `effective: null` and logs no carry
  warning; with a corrupt `latest.json`, or one lacking the `effective`
  field, the round still publishes (warning logged) and `effective` is
  `null`; with a `latest.json` GET failing for any reason other than
  not-found, the round aborts without publishing and the stored `effective`
  chain is intact for the next round.
