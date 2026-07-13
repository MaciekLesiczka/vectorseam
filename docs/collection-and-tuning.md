# Collection and tuning: design

Status: accepted for collector MVP
Scope: collector (Rust), segment format, storage layout, cohort naming,
tuner consumer contract

## Overview

The collector listens for sampled query-vector frames from the SDK (protocol
v1), buffers them per cohort in memory, and flushes them as immutable segment
files into an object store on a fixed window schedule. The production listener
is TCP so the collector can run as its own pod or service and receive traffic
from many application pods. A Unix-domain socket listener remains available
for same-host demos and simple local test setups. The tuner later reads these
segments, computes ground truth, sweeps `ef`, and publishes a recommendation.

Everything is best effort. The collector must never block the application's
network writes and must never grow without bound. When it cannot keep up, it
drops frames and counts the drops, so bias is visible instead of silent.

Storage doubles as transport: the collector writes segments, the tuner reads
them. There is no push API between them.

## Cohort names

Cohort names appear in object store paths, so the free-form UTF-8 allowed by
protocol v1 is not acceptable end to end. One grammar is enforced everywhere;
nothing is escaped or encoded, invalid names are rejected.

Grammar:

- A name is 1 to 8 segments joined by `/`.
- A segment is 1 to 63 ASCII bytes containing only letters, digits, `.`,
  `_`, `-`, or `=`.
- `=` is an ordinary character in cohort segments. `env=Prod`,
  `env=te.nant`, `env==prod`, `=prod`, and `env=` are all valid.
- The exact segments `.` and `..` are invalid because they are ambiguous in
  local filesystem paths.
- A segment must not start with `window=`. The storage layout uses
  `window=<timestamp>` as the segment that marks the end of the cohort path.
- Whole name at most 255 bytes. No empty segments, no leading or trailing
  `/`.

The hierarchy separator is `/` so that environments, tenants, and indexes map
directly onto object store prefixes — for example `prod/tenant-a/products`,
`prod.tenant`, or `env=Prod/tenant=te.nant/index=products` — and a tuner or an
operator can list one subtree.

The grammar is path-oriented, not Hive-oriented. Names are stored verbatim in
object keys and local filesystem paths; nothing is URL-escaped, case-folded,
or parsed as key/value metadata.

Enforcement:

- SDK validates at capture time and raises `ValueError`. Fail fast, at the
  developer's desk.
- Collector validates every frame independently (it must not trust the
  network). Invalid names are dropped and counted. No quarantine, no
  rewriting.

## Windows

- Tumbling windows, 10 minutes, aligned to the UTC wall clock: boundaries at
  :00, :10, :20, and so on. Alignment makes segment discovery and crash
  recovery trivial — for any point in time, the window it belongs to is pure
  arithmetic, no state.
- A window that starts mid-interval (collector startup, or recovery after a
  crash) keeps its aligned name. Partial coverage is visible in the segment
  header: first/last receive timestamps and frame counts. The name never
  lies about which interval the data belongs to; the header says how much of
  the interval was actually observed.
- Window duration is a config value with a 10-minute default, not an
  operator-facing knob to tune. The calibration window is decided by the
  tuner, which reads as many recent segments as it needs — storage windows
  only set slicing granularity.

## Storage layout

One immutable object per flush:

```
cohorts/<cohort path>/window=<YYYYMMDD>T<HHMM>Z/part-<ulid>.vseam
```

- `window=` marks the end of the cohort path, so hierarchical cohorts are
  unambiguous. Cohort segments may contain `=`, but no cohort segment may
  start with `window=`.
- The timestamp is the aligned window start, UTC.
- A window normally has one part; memory-pressure spills produce more. Parts
  carry a ULID so restarts within a window cannot overwrite earlier parts,
  and lexicographic order roughly follows time.
- Objects are written once and never modified. This matches object store
  semantics (single atomic PUT) and keeps the tuner's read side simple.

Storage backend goes through the `object_store` crate. MVP configures
`LocalFileSystem`; S3 and friends become a config change, not a code change.

## Segment format (`.vseam`)

Little-endian throughout. The stored frame is the byte-exact frame received
from the socket — protocol v1 frames are self-delimiting (they begin with
their own length), so the per-record envelope adds only a receive timestamp.

Header:

| field | type | meaning |
|---|---|---|
| magic | 4 bytes | ASCII `VSG1` |
| header_len | u32 | byte length of the remaining header fields |
| window_start | u64 | aligned window start, unix seconds UTC |
| window_seconds | u32 | window duration |
| first_receive | u64 | receive time of first kept frame, unix micros |
| last_receive | u64 | receive time of last kept frame, unix micros |
| received_frame_count | u64 | frames received for this cohort in this part, including frames later dropped |
| record_count | u64 | records stored in this part |
| cohort_len | u16 | byte length of cohort name |
| cohort | bytes | UTF-8 cohort name |

Records, repeated to end of file:

| field | type | meaning |
|---|---|---|
| receive_time | u64 | collector receive time, unix micros |
| frame | bytes | raw protocol v1 frame, self-delimiting |

Counts are per segment part. A normal window usually has one part; early
memory-pressure spills create additional parts. Dropped frames for a part are
`received_frame_count - record_count`. The tuner can sum all parts in a window
without treating spills specially. Coverage for a window is
`sum(record_count) / sum(received_frame_count)`; the tuner skips or flags
windows where that ratio is suspect.

`header_len` lets a future version append header fields without breaking old
readers.

## Collector runtime requirements

- Hot path (connection reader): read a length-delimited frame, check magic,
  version, and a maximum frame size, parse only far enough to extract and
  validate the cohort name, stamp the receive time, hand off to the writer.
  No float parsing, no copies beyond the read buffer, no storage IO.
- Handoff is a bounded channel. A frame reserves global memory before entering
  the channel; if the reservation fails or the channel is full, the reader
  drops the frame and increments a counter. The reader never waits on the
  writer.
- Concurrent client connections are bounded by configuration. When the limit
  is reached, the listener waits for an existing connection task to finish
  before accepting more.
- TCP is the default production listener. A Unix-domain socket can be selected
  explicitly for local same-host use; both listener types share the same frame
  parsing, validation, buffering, and flushing path.
- Writer owns per-cohort buffers for the current window. At window close,
  every non-empty buffer flushes as a part.
- Memory budget: a per-cohort cap and a global cap. The collector reserves a
  fixed slice of the global cap for one serialized flush buffer:
  `per_cohort_memory_bytes + MAX_SEGMENT_OVERHEAD_BYTES`. The remaining live
  budget covers frames in the reader-to-writer channel and writer-buffered
  records. A cohort exceeding its cap flushes early (spill part). Global
  pressure flushes the largest cohort. Flushes are serialized one at a time;
  while storage is slow, the bounded handoff channel absorbs a small backlog
  and then readers drop new frames when live memory or channel capacity is
  exhausted. More sophisticated concurrent or streaming flushes are
  deliberately out of scope for the MVP; simple, tight resource accounting is
  preferred over higher flush throughput.
- Flush failures (storage errors) are logged and counted; the collector
  keeps running. Losing samples is acceptable, crashing the sidecar is not.
- Graceful shutdown (SIGTERM/SIGINT): stop accepting, drain connection tasks,
  close the writer channel, flush all open buffers, and exit. Shutdown waits
  with finite deadlines and aborts remaining tasks as a forced fallback.
- Counters (received, records, dropped by reason, flush failures) are logged
  periodically; per-part received and record counts are embedded in segment
  headers. No metrics endpoint in MVP.

Default sizing assumes one collector may serve up to 1024 application pods,
with each pod sending roughly one sampled frame per second. The connection cap
is 1024 and the reader-to-writer channel capacity is 2048, giving roughly one
extra second of full-rate burst absorption. The default max frame size is 32
KiB, enough for a 2048-dimensional F64 vector plus protocol and cohort-name
overhead without making oversized streams cheap; a full default handoff
channel therefore accounts for about 64 MiB of the global budget. The default
per-cohort buffer is 32 MiB, so hot cohorts spill early instead of
monopolizing memory. The default global budget is 256 MiB, a reasonable
minimum for a production sidecar while still leaving room for queued frames,
multiple cohorts, and the fixed flush reserve.

## Tuner requirements (consumer contract)

Collector's output is Tuners input, so the contract is fixed here:

- The tuner lists `cohorts/<cohort>/` prefixes and reads windows newest
  first until it has the sample count it needs. The effective calibration
  window is therefore emergent — a result of traffic and required samples,
  not a configured value.
- The tuner determines the required sample count itself via holdout
  validation (random train/test split over the pooled samples). A failed
  holdout within pooled windows means insufficient samples; distribution
  change is observed only across successive calibration cycles.
- Per-window coverage (kept/received) tells the tuner when drops may have
  biased a window; it can skip such windows.
- Ground truth requires access to the corpus, so the tuner runs with
  database connectivity (exact scan plus `ef` sweep against the live index
  or a replica). The corpus is never exported through this pipeline.
- The tuner publishes its result as a small object (recommended `ef`, target
  percentile, sample count used, windows used) that the sidecar polls and
  serves to the application. The same object later carries the sampling
  directive for the central variant of adaptive sampling (see
  `adaptive-sampling.md`).

## Out of scope for the collector MVP

- Tuner logic and the result/polling path.
- Sampling-rate feedback to the SDK.
- Remote object stores in configuration (the code path is
  `object_store`-generic; only local filesystem is wired up and tested).
- Metrics endpoints, compression, compaction of old windows, retention.
