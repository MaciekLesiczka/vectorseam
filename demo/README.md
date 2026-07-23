# VectorSeam M1 demo

This demo runs one SuperUser cohort end to end. PostgreSQL, the collector, the
API, and the tuner run in Docker Compose; the query driver runs on the host:

```
live query -> FastAPI -> Python SDK -> collector -> tuner -> latest.json
                       \-> pgvector
```

## Prerequisites

- Docker and [uv](https://docs.astral.sh/uv/).
- DuckDB's CLI for the optional sweep sanity query.
- Network access when a new API container first loads the pinned
  `BAAI/bge-small-en-v1.5` model.
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
make demo-load-data
```

The loader recreates `docs_superuser`, writes `demo/data/queries.txt`, and
prints the 300,000-row count and HNSW build time. PostgreSQL data lives in the
gitignored host directory `demo/data/postgres`, so `make demo-down` and later
Compose runs preserve the loaded table and index.


## Run the pipeline

Start the service stack in one terminal:

```sh
make demo
```

This is an attached Compose run. Collector and tuner logs stay in the
foreground; PostgreSQL and API logs are suppressed. From another terminal,
wait until `docker compose -f demo/docker-compose.yml ps` reports the API as
healthy, then run the driver on the host:

```sh
PYTHONPATH=demo uv run python -m driver \
  --queries demo/data/queries.txt \
  --url http://127.0.0.1:8000 \
  --qps 5 \
  --seed 7
```

Optional paremeters

`API_LOGS=1` includes API startup and request logs in an attached run.
`DETACHED=1` starts the stack in the background, where Compose does not stream
any service logs. Follow operational logs afterward with
`docker compose -f demo/docker-compose.yml logs -f collector tuner api`.

The PostgreSQL data, collector segments, and tuner measurements/calibrations
are bind-mounted under `demo/data`. They remain visible on the host and
survive container replacement and `make demo-down`. The API is stateless.

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
jq '{status, recommended_ef, confidence, effective, samples}' \
  demo/data/store/calibrations/superuser/latest.json
```

It should report `status: "ok"`, settle near `recommended_ef: 60`, and expose
the same value as `effective.recommended_ef` with `carried: false` (one
adjacent grid step is tolerated in early rounds). `recommended_ef` describes
only the current round. If a later transient failure produces
`insufficient_samples`, it becomes null while `effective` retains the last
reliable recommendation with `carried: true`. M1 displays that value but does
not apply it; recommendation consumption remains milestone 2.

Typical timings at 5 qps are up to two minutes for the first `.vseam`, three
to four minutes for the first Parquet pair, and six to ten minutes for the
first successful calibration. If artifacts take more than twice that long,
check collector counters, then `samples.failed` for statement timeouts, then
closed-window alignment.

For a continuously refreshed view, use:

```sh
watch -n 5 "jq '{status, recommended_ef, confidence, effective, samples}' \
  demo/data/store/calibrations/superuser/latest.json"
```

Stop the stack without deleting its data:

```sh
make demo-down
```

To tear down the stack and delete every demo artifact, including the loaded
PostgreSQL cluster and any legacy Compose-managed volume, run:

```sh
make demo-clean
```
