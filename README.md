<p align="center">
  <img src="dashboard/vectorseam_dashboard/static/favicon.svg" width="72" alt="VectorSeam">
</p>

# VectorSeam

[![CI](https://github.com/MaciekLesiczka/vectorseam/actions/workflows/ci.yml/badge.svg)](https://github.com/MaciekLesiczka/vectorseam/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Monitor and tune vector index parameters from live query traffic, instead
of staying blind.**

Approximate indexes trade recall for latency, and the knob that sets the
trade-off — `hnsw.ef_search` in pgvector — is usually set once and never
checked against real traffic again. The recall your queries actually get in
production is invisible: it depends on your corpus and your query
distribution, and it drifts as both change.

VectorSeam turns that into a measured number. It samples the query vectors
your application really sends, replays them against your own index — exact
ground truth plus an `ef_search` sweep — and publishes the smallest
`ef_search` that meets a recall target you declare, together with the
statistical confidence behind it, validated on held-out queries.

## How it works

```
app (Python SDK) ─▶ collector ─▶ object store ─▶ tuner ─▶ recommendation ─▶ dashboard
                                                   │
                                                   └─▶ exact ground truth + ef_search sweep
                                                       against your pgvector database
```

- **SDK** (Python) — one call on your query path: `capture_vector(cohort,
  vec)`. Sampling happens before any work; capture is non-blocking and
  best-effort, so search never stalls or fails because of VectorSeam.
  Adaptive sampling targets a fixed sample rate per cohort regardless of
  traffic level.
- **Collector** (Rust) — receives frames over TCP, buffers per cohort under
  hard memory bounds, and writes immutable segments to object storage on a
  window schedule. Overload drops frames and counts them, so bias is visible
  instead of silent.
- **Tuner** (`seam`, Rust) — for each sampled query, computes exact top-k
  ground truth and sweeps `ef_search` in a single database transaction, then
  recommends the smallest value whose recall target clears a confidence gate
  on training data and validates it once on an untouched holdout. Runs
  against your live database under a duty-cycle budget (default: busy at
  most 10% of wall time), with statement and client timeouts. Vectors stay
  in your database; nothing is exported.
- **Dashboard** — per cohort: the recommendation and its confidence, the
  recall/latency trade-off across the sweep, and round-by-round history.

You declare the target, for example *at least 90% of queries reach recall@10
≥ 0.9*; the tuner publishes per cohort:

```jsonc
// calibrations/products.search.query/latest.json (trimmed)
{
  "status": "ok",
  "recommended_ef": 60,
  "confidence": 0.97,        // probability the target holds on unseen queries
  "effective": { "recommended_ef": 60, "carried": false },
  "samples": { "unique": 1834, "train": 1100, "test": 734 }
}
```

Everything moves through object storage as plain files — no service mesh, no
agent in your database, nothing installed server-side.

## See it running

The repository ships a Docker Compose demo that runs the whole pipeline on
two real corpora (Super User questions and Reddit TL;DR, 300k documents
each): live queries, collection, tuning, and the dashboard at
`http://localhost:8080`. Once the input data is in place
([demo/README.md](demo/README.md) walks through the prerequisites), the
stack is one command:

```sh
make demo
```

## Instrumenting an application

```python
from vectorseam import VectorSocketSender, capture_vector

sender = VectorSocketSender(host="vectorseam-collector.default.svc", port=7737)
sender.start()

# on the query path, right after embedding:
capture_vector("products.search.query", query_vector)
```

Cohort names are hierarchical (`prod/tenant-a/products`), so one deployment
can calibrate many indexes, tenants, or environments independently.

## Status

MVP — working end to end, early days. Today it supports pgvector HNSW
`ef_search` with cosine distance, a Python SDK, and local-filesystem object
storage. Recommendations are published and displayed, not yet applied
automatically — closing that loop, remote object stores, and more parameters
and backends are next.

The estimator is anchored to a reproducible offline benchmark
([python/ann-recall-latency](python/ann-recall-latency/README.md)); the tuner
continuously runs the same pipeline the benchmark runs offline, and its
acceptance tests compare the two implementations on shared fixtures.

## Documentation

- [Demo walkthrough](demo/README.md) — run the full pipeline locally
- [Collector specification](docs/collector-spec.md) — segment format,
  storage layout, resource bounds
- [Tuner specification](docs/tuner-spec.md) — estimator semantics,
  configuration, storage contract
- [Adaptive sampling](docs/adaptive-sampling.md) — how sample flow stays
  bounded across traffic levels
- [Dashboard](dashboard/README.md) — configuration and data contract
- [ANN recall/latency benchmark](python/ann-recall-latency/README.md) — the
  offline methodology behind the tuner

## Development

```sh
make test
```

runs the Rust workspace and Python test suites. See the
[Makefile](Makefile) for benchmark, fixture, and demo targets, and
[CONTRIBUTING.md](CONTRIBUTING.md) if you want to open a pull request.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
