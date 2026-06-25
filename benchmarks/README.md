# Python Message Benchmarks

These benchmarks measure Python vector message marshalling only. They do not
cover queueing, batching, sockets, sender workers, collector behavior, or
end-to-end IPC.

The benchmark uses `pyperf` so local runs get calibration, warmups, subprocess
isolation, and JSON output suitable for comparison. GitHub-hosted runners are
noisy, so CI uploads results as artifacts but does not enforce thresholds.
By default it collects 7 values across 20 worker processes per benchmark; use
`--values` and `--processes` to tune sample counts for a specific run.

Run all default dimensions:

```bash
python benchmarks/bench_message.py
```

With the project-managed environment:

```bash
make bench-python
make bench-python-report
```

Save results:

```bash
python benchmarks/bench_message.py --output before.json
python benchmarks/bench_message.py --output after.json
python -m pyperf stats before.json
python -m pyperf compare_to before.json after.json
```

Run one dimension:

```bash
python benchmarks/bench_message.py --dimension 3072
```

Run several dimensions:

```bash
python benchmarks/bench_message.py --dimensions 384,768,1536,3072,4096
```

The `Python Benchmarks` GitHub Actions workflow writes
`.benchmarks/message.json` and uploads it as the
`python-message-benchmark` artifact.

Inspect a downloaded artifact:

```bash
make bench-python-report PYTHON_BENCH_JSON=message.json
```

Benchmark families:

```text
message_list_f32_*:
  current convenience path using list[float]

message_memoryview_current_*:
  current production-like bytes/memoryview path

message_memoryview_no_crc_bytes_*:
  prototype showing cost without CRC but still returning bytes

message_memoryview_no_crc_bytearray_*:
  prototype showing cost without CRC and without final immutable bytes copy

message_frame_parts_no_vector_copy_*:
  prototype lower-bound for future no-copy frame-parts design
```
