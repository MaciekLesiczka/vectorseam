# VectorSeam demo — Milestone 1 spec

Status: draft
Scope: minimal end-to-end run proving SDK → collector → storage → tuner →
`latest.json` works on one cohort, one index, one data source. No dashboard,
no docker for app components, no HuggingFace sourcing.

## Goal

A single documented sequence of commands that produces
`demo/data/store/calibrations/superuser/latest.json` with `status: "ok"` and a
`recommended_ef` consistent with the published benchmark for the SuperUser
corpus (p10 ≥ 0.9 → expected 60, ±1 grid step tolerated).

## Repository layout

```
demo/
  README.md            # deliverable: end-to-end run instructions
  api/                 # FastAPI search service
  driver/              # query replay script
  scripts/
    load_data.py       # loads postgres + emits queries.txt
  config/
    seam.yaml          # tuner demo config
  data/                # gitignored
    queries.txt        # produced by load_data.py
    store/             # object store root (collector writes, tuner reads/writes)
```

## Data inputs

Milestone 1 assumes the benchmark data already exists locally at:

- raw docs: `python/ann-recall-latency/data/processed/stackexchange/docs.parquet`
- raw queries: `python/ann-recall-latency/data/processed/stackexchange/queries.parquet`
- doc embeddings: `python/ann-recall-latency/data/embeddings/stackexchange/BAAI_bge-small-en-v1.5__5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/docs.parquet`

No download logic anywhere. If a path is missing, scripts fail with an error
naming the path. HuggingFace hosting is a later milestone.

## Components

### 1. Postgres

pgvector image via docker, the only containerized piece. Persistent volume so
the index build happens once. Same index parameters as the benchmark:
HNSW, `m = 16`, `ef_construction = 64`, cosine (`vector_cosine_ops`), 384 dims.

### 2. `demo/scripts/load_data.py`

Self-contained script. Does two things:

1. Reads raw `queries.parquet` and writes query texts to
   `demo/data/queries.txt`, one query per line, in file order. No embeddings —
   the API embeds live; that is the pipeline under test.
2. Loads the documents table: joins raw `docs.parquet` (text) with the
   embeddings `docs.parquet` by document id and writes one table:

   ```sql
   CREATE TABLE docs_superuser (
       doc_id    bigint PRIMARY KEY,
       body      text NOT NULL,
       embedding vector(384) NOT NULL
   );
   ```

   This is the one difference from the benchmark loader: the table carries the
   text column, because the API returns text results. Load via COPY, build the
   HNSW index after loading, use parallel index build
   (`max_parallel_maintenance_workers`).

Idempotent: drop and recreate the table on rerun. Print row count and index
build time at the end.

### 3. `demo/api/` — search service

FastAPI, one endpoint:

```
POST /search
{"query": "<text>", "k": 10}
→ {"results": [{"doc_id": ..., "body": "<truncated to 300 chars>", "distance": ...}],
   "latency_ms": ..., "ef_search": ...}
```

Behavior per request:

1. Embed the query text with `BAAI/bge-small-en-v1.5`. **Copy-paste the
   embedding code from the benchmark** (`python/ann-recall-latency/`) — the
   query instruction prefix and normalization must be byte-identical to what
   produced the doc embeddings, or tuner numbers stop being comparable to the
   blog. Do not import from the benchmark package. The app is self-contained
   and depends only on the vectorseam public SDK, FastAPI, psycopg,
   sentence-transformers/torch, numpy.
2. `capture_vector("superuser", vec)` with the vector as contiguous `<f4`
   (as in the SDK README example). Cohort name is the single segment
   `superuser`.
3. Run the pgvector query in one transaction:
   `SET LOCAL hnsw.ef_search = <EF>` then
   `SELECT doc_id, body, embedding <=> $1 AS distance FROM docs_superuser
   ORDER BY embedding <=> $1 LIMIT k`. No key tie-break in ORDER BY (would
   defeat the index scan).
4. `ef_search` comes from env `DEMO_EF_SEARCH`, default `100`, read at
   startup. No polling of `latest.json` — applying recommendations is
   milestone 2. Report the value in every response.

`VectorSocketSender` configured via env: `COLLECTOR_HOST` (default
`127.0.0.1`), `COLLECTOR_PORT` (default `7737`). Started on app startup,
stopped on shutdown. Best-effort: the API must serve normally if the collector
is down.

Sampling: capture every query. Do not use the adaptive sampler in M1.

### 4. `demo/driver/` — query replay

Script, argparse:

```
python -m driver --queries demo/data/queries.txt --url http://127.0.0.1:8000 \
                 --qps 5 --seed 7
```

- Reads all lines, shuffles once with the seed, loops forever over the
  shuffled order.
- `--qps` (float, default 5) controls request rate; simple sleep pacing is
  fine.
- Logs one line per N requests: request count, error count, rolling mean
  latency. Non-200 responses are counted and skipped, never crash the loop.

Known ceiling, not a bug: the query pool has 2,000 entries and the tuner
dedups by vector hash, so `samples.unique` caps at 2,000 regardless of how
long the driver runs.

### 5. Collector

Run locally: `cargo run -p <collector crate> -- <args/config>` with TCP
listener on `7737` and storage root `demo/data/store`. Storage window 60 s.
No collector code changes expected.

### 6. Tuner

`cargo run -p seam -- --config demo/config/seam.yaml`.

`demo/config/seam.yaml`:

```yaml
calibration:
  interval: 1min
  ef_search: [10, 20, 40, 60, 80, 100, 150, 200, 300, 400]
  min_samples: 300

storage:
  root: ./demo/data/store
  window_seconds: 60

budget:
  db_share: 1.0                  # demo only; production default is 0.10
  statement_timeout: 30s         # must exceed measured exact-scan time, see below
  client_timeout: 60s

data_sources:
  demo:
    server: localhost:5432
    database: postgres
    user: postgres
    password_env: SEAM_PG_DEMO

indexes:
  superuser:
    data_source: demo
    table: docs_superuser
    key: doc_id
    column: embedding

targets:
  demo_recall:
    k: 10
    value: 0.9
    percentile: 0.90             # blog's p10 >= 0.9
    window: 600s                 # 10 storage windows; must be exact multiple of window_seconds

cohorts:
  superuser:
    index: superuser
    target: demo_recall
```

## Pre-flight (do before wiring anything)

Run one brute-force query manually against the loaded corpus:

```sql
BEGIN;
SET LOCAL enable_indexscan = off;
EXPLAIN ANALYZE
SELECT doc_id FROM docs_superuser
ORDER BY embedding <=> '<any embedding literal>' ASC, doc_id ASC
LIMIT 10;
ROLLBACK;
```

Record the time. Two consequences:

1. If it exceeds `statement_timeout`, every tuner sample fails silently into
   `failed_count` and the round stays `insufficient_samples` forever while
   everything looks alive. Set the timeout comfortably above the measured
   value.
2. Scan time × ~300 samples ≈ Phase A duration = wait for the first real
   round. At 500 ms/scan that is ~3 min; at 5 s/scan it is ~30 min and the
   corpus is too big for comfortable iteration on this machine.

## Verification sequence

Each artifact proves one arrow. Run in order:

1. Start postgres, `load_data.py`, collector, API, driver. After the next
   minute boundary + flush:
   `demo/data/store/cohorts/superuser/window=*/part-*.vseam` exists →
   SDK → collector → storage works. Collector counters: received ≈ records,
   drops ≈ 0.
2. Start tuner. After a round over a closed window:
   `measurements/superuser/window=*/part-*.truth.parquet` and
   `.sweep.parquet` exist → Phase A works. Sanity check with duckdb:

   ```sql
   SELECT ef, avg(recall), quantile_cont(recall, 0.10)
   FROM 'demo/data/store/measurements/superuser/**/*.sweep.parquet'
   GROUP BY ef ORDER BY ef;
   ```

   The curve must resemble the blog (full corpus: mean recall ≈ 0.94 at
   ef = 40). If it is far off, the embedding path diverged from the benchmark
   — stop and fix that before anything else.
3. `calibrations/superuser/round-*.json` appears with
   `status: "insufficient_samples"` and honest sample counts → Phase B works.
4. Once `samples.unique ≥ 300` sits in closed windows: `status: "ok"`,
   `latest.json` updated → full loop works.
5. Final assertion: `recommended_ef` ≈ the benchmark's pick for the same
   corpus and target (60 for full 300k SuperUser). Stable-after-a-few-rounds
   is the pass criterion; flipping between adjacent grid steps in early
   rounds is expected at this sample size, not a failure.

Expected wall clock at 5 qps, sample-everything, db_share 1.0, sub-second
scans: first `.vseam` ≤ 2 min, first parquet pair ≤ 3–4 min, first `ok` round
~6–10 min. If nothing appears by 2× those numbers, check in order: collector
counters (are frames arriving), `samples.failed` in the round JSON (statement
timeouts), window alignment (the tuner reads only closed windows — a query is
invisible for up to ~2 min by design).

## Deliverables

1. `demo/api/`, `demo/driver/`, `demo/scripts/load_data.py`,
   `demo/config/seam.yaml` as specified.
2. `demo/README.md` describing the full run end to end: prerequisites (local
   data paths, docker, rust toolchain, python env), the exact command
   sequence (postgres → load_data → collector → API → driver → tuner), what
   files to expect where and roughly when, and the verification steps above
   including the duckdb sanity query and the pre-flight scan timing.
3. A completed run on the local machine ending in
   `calibrations/superuser/latest.json` with `status: "ok"`.

## Out of scope for M1

- Dashboard of any kind (`watch -n 5 "jq '{status, recommended_ef,
  confidence, samples}' demo/data/store/calibrations/superuser/latest.json"`
  is the M1 dashboard).
- API consuming `latest.json` / applying recommendations.
- Adaptive sampling.
- Second cohort, second corpus, hierarchical cohort names.
- Docker compose for app components; k8s.
- HuggingFace data hosting or any download logic.
- Any collector or tuner code changes.
