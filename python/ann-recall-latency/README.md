# ef_search Calibration

Tests how different `hnsw.ef_search` values impact recall and latency for
real-world datasets. Validates whether the parameter can be tuned to provide a
predictable recall SLI.

## Methodology

Parameters live in
[`config.yaml`](config.yaml). The default run independently samples `300000`
documents and `2000` queries per dataset with seed `7`, filters short text,
computes exact top-k ground truth, then sweeps `ef_search`.

Datasets and query origins:

- StackExchange: Archive.org item `stackexchange_20251231`, file
  `superuser.com.7z`, dump date `2025-12-31`. Documents are sampled from Super
  User question bodies; queries are sampled separately from Super User question
  titles. StackExchange content is CC-BY-SA; see [`NOTICE`](NOTICE).
- Reddit: Hugging Face repo `webis/tldr-17`, file
  `data/corpus-webis-tldr-17.zip`. Documents use the `content` field; queries
  use the `summary` field.

Embeddings:

- Model: `BAAI/bge-small-en-v1.5`
- Dimension: `384`
- Normalized

ANN implementation:

- `pgvector/pgvector:pg16` in [`docker-compose.yml`](docker-compose.yml).
- Each dataset is loaded into `docs_<dataset>` with a pgvector `embedding`
  column.
- The index is pgvector HNSW with `vector_cosine_ops`, `m = 16`, and
  `ef_construction = 64`.
- Sweep queries run `ORDER BY embedding <=> $query_vector LIMIT k` while setting
  `hnsw.ef_search` to each configured value.
- Default `k = 10`, `ef_search = [10, 20, 40, 60, 80, 100, 150, 200, 300, 400]`,
  and each ANN query is repeated `3` times.
- Validates whether tuning on the train set transfers to the test set, given the
  configured number of observations.

Outputs are written under [`results/`](results/). The file
[`results/calibration_transfer.csv`](results/calibration_transfer.csv) contains
the tuning-validation results.

## Reproduce

The test runs in several stages:

```
Download datasets
    -> Load documents and query samples into parquet files
        -> Create embeddings
            -> Load document embeddings into Postgres
                -> Calculate true top-k
                    -> Search HNSW with different ef_search
                        -> Summarize the results
```

Every intermediate step has its own artifacts and a corresponding manifest
describing the stage output in `python/ann-recall-latency/data`.

Because stages can be time-consuming (embedding takes 5 hours to complete on my
Mac), intermediate results are cached: subsequent job runs skip a stage if the
data is already present.

Run stages individually to reuse local caches:

```sh
make ann-recall-latency-download
make ann-recall-latency-load
make ann-recall-latency-embed
make ann-recall-latency-pg-load
make ann-recall-latency-ground-truth
make ann-recall-latency-sweep
make ann-recall-latency-analyze
```

Run `make all-ann-recall-latency` for the full pipeline.

Changing a parameter in [`config.yaml`](config.yaml) that a stage depends on
will trigger a rerun of that stage.
