# VectorSeam demo

This demo runs SuperUser and Reddit TLDR cohorts end to end. PostgreSQL, the
collector, the API, the tuner, and the dashboard run in Docker Compose; one
multi-cohort query driver runs on the host:

```
live query -> FastAPI -> Python SDK -> collector -> tuner -> latest.json -> dashboard
                       \-> pgvector
```

Once the stack is up, the dashboard is served at http://localhost:8080. It
reads the tuner's calibration output live for both cohorts and renders
recommended `ef_search`, holdout confidence, the recall/latency tradeoff, and
round history. See [../dashboard/README.md](../dashboard/README.md) for the
component and its configuration.

## Prerequisites

- Docker and [uv](https://docs.astral.sh/uv/).
- DuckDB's CLI for the optional sweep sanity query.
- Network access when a new API container first loads the pinned
  `BAAI/bge-small-en-v1.5` model.
- The six existing benchmark artifacts below. The demo never downloads data
  and reports the missing path if any input is absent.

```
python/ann-recall-latency/data/processed/stackexchange/docs.parquet
python/ann-recall-latency/data/processed/stackexchange/queries.parquet
python/ann-recall-latency/data/embeddings/stackexchange/BAAI_bge-small-en-v1.5__5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/docs.parquet
python/ann-recall-latency/data/processed/reddit/docs.parquet
python/ann-recall-latency/data/processed/reddit/queries.parquet
python/ann-recall-latency/data/embeddings/reddit/BAAI_bge-small-en-v1.5__5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/docs.parquet
```

Today these artifacts are produced by the benchmark pipeline
([../python/ann-recall-latency/README.md](../python/ann-recall-latency/README.md)):

```sh
make ann-recall-latency-download
make ann-recall-latency-load
make ann-recall-latency-embed
```

The embedding stage is the slow one — budget several hours on a laptop. It
runs once; every later demo run reuses the cached files.

Run every command below from the repository root. First install the Python
environment, start pgvector, and load the data:

```sh
uv sync
make demo-load-data
```

The loader recreates `docs_superuser` and `docs_reddit`, writes
`demo/data/queries_superuser.txt` and `demo/data/queries_reddit.txt`, and prints
each 300,000-row count and HNSW build time. PostgreSQL data lives in the
gitignored host directory `demo/data/postgres`, so `make demo-down` and later
Compose runs preserve both tables and indexes.


## Run the pipeline

Start the service stack in one terminal:

```sh
make demo
```

The first run builds the three local images (the API image bundles PyTorch
and sentence-transformers), which takes a few minutes; later builds are
cached.

This is an attached Compose run. Collector and tuner logs stay in the
foreground; PostgreSQL and API logs are suppressed. From another terminal,
wait until `docker compose -f demo/docker-compose.yml ps` reports the API as
healthy, then run the driver on the host:

```sh
make demo-driver
```

The driver loads both query pools and randomly selects a cohort for every
request. Its default 5 qps is shared across both cohorts, rather than applied
to each cohort separately. Override the total rate and seed with
`DEMO_DRIVER_QPS` and `DEMO_DRIVER_SEED`.

The dashboard switches from sample data to the live view after both cohorts
have published their first `latest.json`.

Optional parameters:

- `API_LOGS=1` includes API startup and request logs in an attached run.
- `DETACHED=1` starts the stack in the background, where Compose does not
  stream any service logs. Follow operational logs afterward with
  `docker compose -f demo/docker-compose.yml logs -f collector tuner api`.

The PostgreSQL data, collector segments, and tuner measurements/calibrations
are bind-mounted under `demo/data`. They remain visible on the host and
survive container replacement and `make demo-down`. The API is stateless.

The API embeds and captures every query under the selected cohort, then
searches that cohort's table. Capture is best-effort, so searches continue
normally while the collector is unavailable. The driver shuffles each
2,000-query pool once, randomly interleaves them, and repeats each pool
forever; tuner deduplication therefore caps `samples.unique` at 2,000 per
cohort.

## Verify

After the next minute boundary and collector flush, a segment proves the SDK
to-storage path:

```sh
find demo/data/store/cohorts \
  \( -path '*/superuser/*' -o -path '*/reddit/*' \) \
  -name 'part-*.vseam'
```

Collector logs should show received records with approximately zero drops.
After the tuner processes a closed window, these files prove Phase A:

```sh
find demo/data/store/measurements \
  \( -name '*.truth.parquet' -o -name '*.sweep.parquet' \)
```

Use DuckDB to compare both sweeps with the published benchmark. On the full
corpora, mean recall at `ef = 40` should be around 0.94 for SuperUser and 0.87
for Reddit:

```sh
duckdb -c "
SELECT regexp_extract(filename, 'measurements/([^/]+)/', 1) AS cohort,
       ef, avg(recall), quantile_cont(recall, 0.10)
FROM read_parquet(
  'demo/data/store/measurements/*/**/*.sweep.parquet',
  filename = true
)
GROUP BY cohort, ef ORDER BY cohort, ef;
"
```

Round JSON first appears with honest `insufficient_samples` counts. The
expected final artifact is:

```sh
for cohort in superuser reddit; do
  jq '{
    cohort, status, recommended_ef, train_confidence, test_compliance,
    confidence, effective, samples, ground_truth_latency_mean_ms
  }' "demo/data/store/calibrations/$cohort/latest.json"
done
```

The round reports `insufficient_samples` until each realized split can attain
its own gate with all successes. At percentile 0.90 the 0.98 selection gate
needs 37 train samples and the 0.90 approval gate needs 21 holdout samples,
so 62 unique samples at `train_fraction: 0.6`.

The tuner then selects the smallest ef whose training confidence clears the
98% selection gate, and evaluates that ef exactly once on the untouched
holdout. A holdout confidence of at least the 90% approval gate replaces
`effective`; anything lower carries the prior effective block, or leaves it
null before the first approval. A `target_unmet` round reports the maximum
grid ef with its evidence and also carries.

The two gates differ on purpose. The holdout is the smaller split, so at the
same true compliance its confidence is lower than the train split's —
selecting at 0.90 would keep proposing ef values the holdout cannot confirm.
The 0.98 selection gate spends that margin deliberately: a slightly higher ef
that gets approved beats the smallest ef that never does. Expect the first
recommendation to be conservative and to settle near `recommended_ef: 60` for
SuperUser and `recommended_ef: 200` for Reddit as evidence accumulates. The
demo displays the effective value but does not apply it — the API serves
every search with a fixed `ef_search` (`DEMO_EF_SEARCH`, default 100).
Automatic consumption of recommendations is future work.

At 5 shared qps, allow up to two minutes for the first `.vseam`. The tuner
processes cohorts sequentially, so Parquet and successful-calibration timing
depends on both database scans and the randomly realized per-cohort request
rate. If artifacts stall, check collector counters, then `samples.failed` for
statement timeouts, then closed-window alignment.

For a continuously refreshed view, use:

```sh
watch -n 5 'for cohort in superuser reddit; do
  jq "{cohort, status, recommended_ef, confidence, effective, samples}" \
    "demo/data/store/calibrations/$cohort/latest.json"
done'
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
