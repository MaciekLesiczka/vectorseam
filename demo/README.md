# VectorSeam M1 demo

This demo runs one SuperUser cohort end to end:

```
live query -> FastAPI -> Python SDK -> collector -> tuner -> latest.json
                       \-> pgvector
```

## Prerequisites

- Docker, a Rust toolchain compatible with the workspace MSRV, and
  [uv](https://docs.astral.sh/uv/).
- DuckDB's CLI for the optional sweep sanity query.
- A local checkout of the pinned `BAAI/bge-small-en-v1.5` model, or network
  access when the API first loads it.
- The three existing benchmark artifacts below. The demo never downloads data
  and reports the missing path if any input is absent.

```
python/ann-recall-latency/data/processed/stackexchange/docs.parquet
python/ann-recall-latency/data/processed/stackexchange/queries.parquet
python/ann-recall-latency/data/embeddings/stackexchange/BAAI_bge-small-en-v1.5__5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/docs.parquet
```

Run every command below from the repository root. First install the Python
environment, start pgvector, and load the data:

```sh
uv sync
make demo-postgres-up
uv run python demo/scripts/load_data.py
```

The loader recreates `docs_superuser`, writes `demo/data/queries.txt`, and
prints the 300,000-row count and HNSW build time. The Docker volume is
persistent, so keep it running after the first load.

## Pre-flight exact scan

Measure one brute-force query before starting the tuner. Capture a vector
literal, paste it in place of `<embedding>`, and retain the `Execution Time`
reported by `EXPLAIN ANALYZE`:

```sh
docker compose -f demo/docker-compose.yml exec -T postgres \
  psql -U postgres -d postgres -Atc \
  'SELECT embedding FROM docs_superuser LIMIT 1'

docker compose -f demo/docker-compose.yml exec -T postgres \
  psql -U postgres -d postgres
```

```sql
BEGIN;
SET LOCAL enable_indexscan = off;
EXPLAIN ANALYZE
SELECT doc_id FROM docs_superuser
ORDER BY embedding <=> '<embedding>' ASC, doc_id ASC
LIMIT 10;
ROLLBACK;
```

`demo/config/seam.yaml` uses a 30-second statement timeout. Increase it above
the measured scan time if necessary; one scan multiplied by roughly 300
samples is also a useful estimate for the first full measurement phase.

## Run the pipeline

Start each long-running command in a separate terminal, in this order:

```sh
cargo run -p vectorseam-collector -- \
  --listen 127.0.0.1:7737 \
  --storage-root ./demo/data/store \
  --window-seconds 60
```

```sh
DATABASE_URL=postgresql://postgres:password@127.0.0.1:5432/postgres \
COLLECTOR_HOST=127.0.0.1 COLLECTOR_PORT=7737 DEMO_EF_SEARCH=100 \
  uv run uvicorn demo.api.app:app --host 127.0.0.1 --port 8000
```

```sh
PYTHONPATH=demo uv run python -m driver \
  --queries demo/data/queries.txt \
  --url http://127.0.0.1:8000 \
  --qps 5 \
  --seed 7
```

```sh
SEAM_PG_DEMO=password \
  cargo run -p seam -- --config demo/config/seam.yaml
```

The API embeds and captures every query. Capture is best-effort, so searches
continue normally while the collector is unavailable. The driver shuffles the
2,000-query pool once and repeats it forever; tuner deduplication therefore
caps `samples.unique` at 2,000.

## Verify

After the next minute boundary and collector flush, a segment proves the SDK
to-storage path:

```sh
find demo/data/store/cohorts/superuser -name 'part-*.vseam'
```

Collector logs should show received records with approximately zero drops.
After the tuner processes a closed window, these files prove Phase A:

```sh
find demo/data/store/measurements/superuser \
  \( -name '*.truth.parquet' -o -name '*.sweep.parquet' \)
```

Use DuckDB to compare the sweep with the published benchmark. On the full
corpus, mean recall around 0.94 at `ef = 40` is the important sanity check:

```sh
duckdb -c "
SELECT ef, avg(recall), quantile_cont(recall, 0.10)
FROM 'demo/data/store/measurements/superuser/**/*.sweep.parquet'
GROUP BY ef ORDER BY ef;
"
```

Round JSON first appears with honest `insufficient_samples` counts. Once at
least 300 unique samples are in closed windows, the expected final artifact is:

```sh
jq '{status, recommended_ef, confidence, samples}' \
  demo/data/store/calibrations/superuser/latest.json
```

It should report `status: "ok"` and settle near `recommended_ef: 60` (one
adjacent grid step is tolerated in early rounds). Typical timings at 5 qps are
up to two minutes for the first `.vseam`, three to four minutes for the first
Parquet pair, and six to ten minutes for the first successful calibration.
If artifacts take more than twice that long, check collector counters, then
`samples.failed` for statement timeouts, then closed-window alignment.

For a continuously refreshed view, use:

```sh
watch -n 5 "jq '{status, recommended_ef, confidence, samples}' \
  demo/data/store/calibrations/superuser/latest.json"
```
