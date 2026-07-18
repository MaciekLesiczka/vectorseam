# Phase B benchmark baseline

Stage 3 baseline recorded on 2026-07-17 with `rustc 1.96.0`, an Apple M3
(`arm64`), and the Criterion release profile:

| Benchmark | 95% estimate interval |
|---|---:|
| `phase_b/aggregate_10k_samples_5_ef` | 2.7665–2.8528 ms |
| `phase_b/type7_quantile_10k` | 47.665–51.978 µs |

Run `make bench-seam-phase-b` to reproduce the benchmark. Absolute timings
are machine-dependent; this file fixes the Stage 3 workload and initial
reference result so later changes can report their environment and relative
movement without changing the benchmark fixture.
