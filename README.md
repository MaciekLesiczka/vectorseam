# Tests and benchmarks

# ef_search calibration

Tests how different `hnsw.ef_search` values impact recall and latency for real-world datasets.
Validates whether the parameter can be tuned to provide a predictable recall SLI.


## Methodology

Parameters live in
[`python/ann-recall-latency/config.yaml`](python/ann-recall-latency/config.yaml).
The default run samples `300000` documents and `2000` queries per dataset with
seed `7`, filters short text, computes exact top-k ground truth, then sweeps
`ef_search`.

Datasets and query origins:

- StackExchange: Archive.org item `stackexchange_20251231`, file
  `superuser.com.7z`, dump date `2025-12-31`. Documents are Super User question
  bodies; queries are the corresponding question titles.
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
- Validates whether tuning on the train set transfers to the test set, given the provided number of observations.

Outputs are written under `python/ann-recall-latency/results/`.
The file `python/ann-recall-latency/results/calibration_transfer.csv` contains the tuning-validation results.

## Reproduce

The test is done in several stages

```
Data sets download
    -> Load doc and query samples into parquet
        -> Create embeddings
            -> Load docs embeedings into Postgres 
                -> Calculate true top k
                    -> Search HNSW with different ef_search
                        -> Summarize the results

```

Every intermediate step has its own artefacts and corresponging manifest describing the result that land in python/ann-recall-latency/data. 

Because stages can be time consiming (embeeding takes 5 hours to complete on my Mac), intermediate results are cached. I.e. subsequent job runs skip the stage if the data is already present.

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


Changing a parameter in `python/ann-recall-latency/config.yaml` which a stage depends on, will triger rerun of the stage.