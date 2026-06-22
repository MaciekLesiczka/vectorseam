# pgvector ANN Recall-Latency Benchmark

Measures pure pgvector HNSW recall and latency against exact brute-force
top-k, comparing StackExchange and Reddit corpora across `hnsw.ef_search`.

Stage 0 downloads pinned raw inputs and writes `data/manifest.json`:

```sh
make ann-recall-latency-download
```

Stage 1 parses raw inputs into normalized `docs.parquet` and `queries.parquet`:

```sh
make ann-recall-latency-load
```

Stage 2 embeds docs and queries with the same model and preprocessing:

```sh
make ann-recall-latency-embed
```

Stage 3 loads embeddings into Postgres and builds HNSW indexes:

```sh
make ann-recall-latency-pg-load
```

Stage 4 computes exact brute-force top-k ground truth:

```sh
make ann-recall-latency-ground-truth
```

Stage 5 sweeps pgvector `hnsw.ef_search` and records recall/latency:

```sh
make ann-recall-latency-sweep
```

Stage 6 summarizes and plots the cross-dataset comparison:

```sh
make ann-recall-latency-analyze
```

Raw downloads live under `python/ann-recall-latency/data/` and are gitignored.
StackExchange content is CC-BY-SA; see `NOTICE`.
