# Vectorseam

Vectorseam currently contains vector search benchmarks and measurement code.

The main public artifact is an ANN recall/latency benchmark for calibrating
`hnsw.ef_search` on real-world datasets:

- [ANN recall/latency benchmark](python/ann-recall-latency/README.md)

Benchmark result files under `python/ann-recall-latency/results/` are checked in
intentionally so external writeups can reference stable artifacts.

Run local checks with:

```sh
make test
```

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
