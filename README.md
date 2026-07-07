# Vectorseam

Vectorseam currently contains vector search benchmarks and measurement code.

The main public artifact is an ANN recall/latency benchmark for calibrating
`hnsw.ef_search` on real-world datasets:

- [ANN recall/latency benchmark](python/ann-recall-latency/README.md)

Benchmark result files under `python/ann-recall-latency/results/` are checked in
intentionally so external writeups can reference stable artifacts.

## Python SDK

Capture a NumPy vector with the process-wide producer:

```python
capture_vector("products.search.query", query_vector)
```

Minimal pgvector-style application instrumentation:

```python
import numpy
import psycopg

from vectorseam import VectorSocketSender, capture_vector

sender = VectorSocketSender(socket_path="/tmp/vectorseam.sock")
sender.start()

try:
    query_vector = numpy.ascontiguousarray(
        embed_query("waterproof hiking jacket"),
        dtype=numpy.dtype("<f4"),
    )
    capture_vector(
        "products.search.query",
        query_vector,
    )

    vector_literal = "[" + ",".join(str(float(v)) for v in query_vector) + "]"
    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute(
            """
            SELECT id, name, embedding <-> %s::vector AS distance
            FROM products
            ORDER BY embedding <-> %s::vector
            LIMIT 10
            """,
            (vector_literal, vector_literal),
        ).fetchall()
finally:
    sender.stop()
```

Run local checks with:

```sh
make test
```

The Python SDK can capture vectors into a byte-bounded producer queue and send
marshalled frames with `VectorSocketSender`. The sender uses a daemon background
thread and Unix-domain `SOCK_STREAM` socket, micro-batches with
`flush_interval_seconds` and `max_batch_bytes`, and is best-effort: failed
sends drop the current batch and close the connection. Sends are bounded by
`send_timeout_seconds`. `max_batch_bytes` is a target size, frames are never
split, and there is no TCP fallback, ACK, retry, or durable spooling yet.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
