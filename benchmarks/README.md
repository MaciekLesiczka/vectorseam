# Python Benchmarks

These benchmarks measure Python vector frame marshalling and capture hot
paths only. They do not cover batching, sockets, sender workers, collector
behavior, or end-to-end IPC.

The benchmark uses `pyperf` so local runs get calibration, warmups, subprocess
isolation, and JSON output suitable for comparison. GitHub-hosted runners are
noisy, so CI uploads results as artifacts but does not enforce thresholds.
By default it collects 7 values across 20 worker processes per benchmark; use
`--values` and `--processes` to tune sample counts for a specific run.

Run all default dimensions:

```bash
make bench-python
```

Run one module:

```bash
uv run --extra bench python benchmarks/bench_frame.py
uv run --extra bench python benchmarks/bench_vector_capture.py
```

Save results:

```bash
uv run --extra bench python benchmarks/bench_frame.py --output before.json
uv run --extra bench python benchmarks/bench_frame.py --output after.json
uv run --extra bench python -m pyperf compare_to before.json after.json
```

Run one dimension:

```bash
uv run --extra bench python benchmarks/bench_vector_capture.py --dimension 3072
```

Run several dimensions:

```bash
uv run --extra bench python benchmarks/bench_vector_capture.py --dimensions 384,768,1536,3072,4096
```

Inspect results:

```bash
make bench-python-report
```

Missing result files are skipped with the matching `make bench-python-*`
command to generate them.

Benchmark families:

```text
frame_list_f32_*:
  experimental convenience path using list[float] encoded as F32

frame_memoryview_dim_*:
  production-recommended path using packed bytes/memoryview and returning bytes

frame_memoryview_bytearray_*:
  benchmark-only comparison showing the cost of final bytearray-to-bytes copy

frame_memoryview_dim_with_crc_*:
  production path plus a CRC32 scan over the returned bytes

capture_sample_rate_0_01_*:
  capture path with 1% sampling

capture_sample_rate_1_0_*:
  capture path with sampling always enabled

capture_numpy_sample_rate_1_0_*:
  convenience capture path using NumPy dimension and dtype metadata
```
