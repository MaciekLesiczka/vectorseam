# Adaptive sampling

## Overview

VectorSeam uses adaptive sampling to keep query-vector sample flow useful and
bounded across cohorts with very different traffic levels. The SDK samples each
cohort toward a target number of emitted samples per second, instead of asking
operators to choose a fixed probability that depends on deployment traffic.

For each cohort, the sampler estimates recent query rate and computes:

```
p = min(1.0, target_samples_per_second / observed_qps)
```

This gives VectorSeam the behavior it needs for calibration:

- Low-traffic cohorts are sampled at `p = 1.0`, so cold starts gather data as
  quickly as possible.
- High-traffic cohorts are thinned automatically, bounding network, queue, and
  collector load.
- The default target can work across many deployments because it describes the
  desired sample flow, not the application traffic rate.

At the default `1.0` sample/sec, a cohort can accumulate about 3,600 samples per
hour, which is above the sample-size floor measured for calibration transfer at
300k-document scale.

Adaptive sampling is the same broad pattern used in tracing systems such as
Jaeger, AWS X-Ray, and Sentry: event volume follows application traffic, while
the backend needs a controlled and roughly stable sample flow.

## Requirements

- Sampling is per cohort name because tuning is per cohort and traffic can vary
  widely between cohorts.
- Every query in a sampling window should have the same keep probability, so the
  kept set is not biased toward the start or end of a burst.
- The SDK hot path remains simple: sampling decision first, then marshalling,
  then non-blocking enqueue.
- The local SDK variant requires no coordination service and works with the
  collector as currently designed.
- Queue bounds remain the final safety backstop. If local traffic temporarily
  exceeds the estimated probability, queue-full drops are counted and visible.

## Sampling Semantics

The SDK keeps a small rate-estimation state per cohort. Cohort cardinality is
expected to be low because cohort names are hand-assigned labels, not unbounded
request dimensions.

The local sampler estimates arrivals in fixed time buckets. Probability is
recomputed when a bucket rolls over and remains constant within the next bucket.
Keeping probability constant within a bucket preserves unbiased sampling at that
fine-grained timescale.

Cold start behavior is intentionally eager:

```
no completed rate bucket -> p = 1.0
```

Low-traffic cohorts may remain at `p = 1.0`, which is the desired behavior.

If a cohort goes quiet, stale rate estimates decay toward zero as empty buckets
elapse. When traffic resumes, sampling moves back toward `p = 1.0` instead of
being suppressed by old high-traffic estimates.

## Probability, Not Rate Limiting

Adaptive sampling uses probability adjustment rather than a leaky-bucket rate
limiter. A rate limiter keeps the first N events in a period and drops the rest;
that creates a sample biased toward the beginning of bursts and quieter parts of
traffic.

VectorSeam calibrates `ef` from the captured query distribution, so burst shape
matters. If busy-period queries differ from quiet-period queries, a
rate-limited sample can tune for the wrong distribution. Probabilistic thinning
keeps each query eligible with the same probability in the relevant window.

The probability estimate can lag sudden traffic changes because it is based on
recent observed rate. During a burst, the SDK may briefly over-sample until the
estimate catches up. This direction of error is acceptable: the byte-bounded
producer queue and collector memory limits cap retained work, and drops are
counted instead of silently hidden.

## Local Variant

The local variant is the SDK implementation for the current milestone.

Each SDK process independently estimates query rate per cohort and computes its
own probability. This requires no backend coordination and no collector-to-SDK
feedback path.

With N application instances behind a load balancer, aggregate sample flow is
approximately `N * target_samples_per_second` for a hot cohort because each
instance targets the rate independently. This is acceptable for the MVP: the
target is small, the SDK queue is byte-bounded, and the collector enforces its
own channel and memory limits.

The local sampler implements the existing `SamplingPolicy` protocol. Future
variants can replace where the probability comes from without changing the
capture hot path.

## Central Variant

The central variant is deferred until after the collector and tuner milestones.

The tuner will know how many samples it needs per cohort per calibration cycle
because it determines that through holdout validation. It also sees aggregate
arrival rates across all SDK instances. That makes the tuner the natural place
to compute a cohort-level probability for the whole deployment.

The existing result-object path can carry this directive later: the sidecar
already polls for calibrated `ef`, and the same object can include sampling
instructions. That closes the loop so sample volume adapts to calibration needs
and avoids the `N * target` multiplication of the local variant.

The central variant requires a collector or sidecar feedback path back to the
SDK, so it is out of scope for the initial SDK and collector work.

## SDK Design

- `AdaptiveSampler` implements `SamplingPolicy`.
- `ProbabilitySampler` remains available for deterministic fixed-probability
  behavior, tests, and explicit user needs.
- Per cohort state uses fixed time buckets: a current bucket count, a rate
  estimate, and bucket timing.
- No timestamp queues or per-call allocations are required.
- Timing uses `time.monotonic()`.
- Thread safety follows the SDK capture module discipline: one lock, short
  critical sections, and no work inside the lock beyond state update and random
  decision bookkeeping.

## Defaults

- `target_samples_per_second = 1.0` per cohort.
- Rate bucket duration: 5 seconds.

These defaults are intended to work without per-deployment tuning until the
central variant is available.
