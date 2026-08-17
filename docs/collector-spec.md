# VectorSeam Collector — Specification

Status: implemented
Scope: the collector component (Rust) — runtime, cohort naming, segment
format, storage layout, and the storage contract consumers rely on. The
tuner that consumes these segments is specified in `tuner-spec.md`.

## Overview

The collector listens for sampled query-vector frames from the SDK (protocol
v1), buffers them per cohort in memory, and flushes them as immutable segment
files into an object store on a fixed window schedule. The production listener
is TCP so the collector can run as its own pod or service and receive traffic
from many application pods. A Unix-domain socket listener remains available
for same-host demos and simple local test setups. The tuner later reads these
segments, computes ground truth, sweeps `ef`, and publishes a recommendation
(`tuner-spec.md`).

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
  operator-facing knob to tune. The calibration window is the tuner's
  concern — a configured rolling window spanning many storage windows
  (`tuner-spec.md` §2.2) — so storage windows only set slicing granularity.

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
| first_receive | u64 | receive time of first kept frame, unix micros; `0` when `record_count = 0` |
| last_receive | u64 | receive time of last kept frame, unix micros; `0` when `record_count = 0` |
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

A segment part may contain zero records when the collector attributes dropped
frames to a cohort/window but kept no frames for that part. In that case
`record_count = 0`, `received_frame_count > 0`, and `first_receive` /
`last_receive` are `0`, making zero coverage explicit for the tuner.

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
  before accepting more. Connections also have an idle read timeout so dead
  peers cannot hold slots forever.
- TCP is the default production listener. A Unix-domain socket can be selected
  explicitly for local same-host use; both listener types share the same frame
  parsing, validation, buffering, and flushing path.
- Writer owns per-cohort buffers for the current receive-time window. Before
  buffering a record whose receive timestamp belongs to a later aligned
  window, it flushes the previous window; at window close, every non-empty
  buffer and every drop-only cohort count flushes as a part.
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
- Inline single-flight flush is acceptable for the LocalFileSystem MVP. At the
  design point, a full 32 MiB local write should complete well within the
  two-second default channel cushion. For remote stores such as S3, 32 MiB
  PUTs and sequential window-close PUTs can exceed that cushion and create
  time-correlated drops around spills or window boundaries. The remote-store
  milestone must revisit this tradeoff, for example with a larger channel or
  one overlapped in-flight PUT while explicitly reserving memory for the extra
  serialized segment.
- Memory accounting is intentionally based on buffered record payload bytes:
  `frame.len() + 8` for the receive timestamp. It does not count Rust
  container overhead such as `BufferedRecord`, `Bytes`, `MemoryGuard`,
  `Vec` capacity slack, `CohortName`, or Tokio channel slot metadata. For
  expected vector frames around 3-16 KiB, this overhead is small, roughly
  1-2%. A hostile stream of minimum-size valid frames can make resident memory
  substantially higher than the configured byte budget, roughly 2.5x in the
  worst case. This MVP accepts that tradeoff to keep hot-path accounting
  simple; the frame-size cap, channel cap, per-cohort cap, and connection cap
  still bound growth. Treat `global_memory_bytes` as a frame-byte budget, not
  a process RSS limit; container memory limits should include headroom for
  runtime, allocator, task, channel, and object-store overhead. For the default
  256 MiB frame-byte budget, a 512 MiB container limit is a more realistic
  starting point than 256 MiB.
- Flush failures (storage errors or PUT timeouts) are logged and counted; the
  collector keeps running. Losing samples is acceptable, crashing the
  collector is not.
- Graceful shutdown (SIGTERM/SIGINT): stop accepting, drain connection tasks,
  close the writer channel, flush all open buffers, and exit. Shutdown waits
  with finite deadlines and aborts remaining tasks as a forced fallback. The
  writer shutdown wait is 20 seconds. Including connection drain, summary,
  and the optional recommendation server, the default collector configuration's
  sequential worst-case wait remains below Kubernetes' default 30-second
  termination grace. A flush still in progress after that may be lost;
  object-store atomic PUT semantics prevent torn segments.
- Counters (received, records, dropped by reason, flush failures) are logged
  periodically; per-part received and record counts are embedded in segment
  headers. No metrics endpoint in MVP.

Default sizing assumes one collector may serve up to 1024 application pods,
with each pod sending roughly one sampled frame per second. The connection cap
is 2048, leaving headroom for reconnects while stale half-open connections are
reaped by the default 300-second idle read timeout. The reader-to-writer
channel capacity is 2048, giving roughly two seconds of full-rate burst
absorption. The default max frame size is 32 KiB, enough for a
2048-dimensional F64 vector plus protocol and cohort-name overhead without
making oversized streams cheap; a full default handoff channel therefore
accounts for about 64 MiB of the global budget. The default per-cohort buffer
is 32 MiB, so hot cohorts spill early instead of monopolizing memory. The
default global budget is 256 MiB, a reasonable minimum for a production
collector while still leaving room for queued frames, multiple cohorts, and
the fixed flush reserve.

## Consumer contract

Storage is the only interface between captured segments and their consumers;
there is no segment push API. The collector's side of the contract is:

- Segment parts are immutable once written, appear only under their aligned
  window prefix, and land in a single atomic PUT — a consumer never observes
  a torn or growing segment.
- A window is safe to consume only after it has closed
  (`window_start + window_seconds` has passed) plus flush latency. Late
  parts (memory-pressure spills, crash recovery) can still appear after a
  consumer's first listing, so consumers must re-list rather than assume a
  window's part set is final.
- Per-part headers carry `received_frame_count` and `record_count`, so a
  consumer can compute per-window coverage
  (`sum(record_count) / sum(received_frame_count)`) and decide when drops
  may have biased a window. Zero-record parts make attributed drops with no
  kept frames explicit rather than invisible.

How the tuner consumes these segments — rolling-window membership,
deduplication, sample sufficiency, and the published
`calibrations/<cohort>/round-<ts>.json` and `latest.json` outputs — is
specified in `tuner-spec.md`. The published result object is also the
intended carrier for the sampling directive of the central variant of
adaptive sampling (see `adaptive-sampling.md`).

## Effective recommendation HTTP API

The collector hosts the independent `vectorseam-recommendation-server`
library. It reads the tuner-owned `latest.json` through the collector's object
store and exposes one endpoint:

```text
GET /v1/ef-search/<cohort>
```

A recommendation returns `200 OK` with only
`effective.recommended_ef` as a plain-text integer. A missing artifact or null
`effective` returns `404 Not Found`; an invalid cohort returns `400 Bad
Request`; a malformed, unsupported, mismatched, oversized, or out-of-range
artifact returns `500 Internal Server Error`; overload, lookup timeout, and
transient storage failures return `503 Service Unavailable`.

The listener is enabled by default on `127.0.0.1:7738`; it can be disabled with
`--recommendation-enabled=false`. When enabled, configuration or bind failure
prevents the collector from starting; disabling it is the explicit way to run
ingest without the API. The demo publishes both the ingest port and port 7738.
The server accepts at most 100 connections and runs at most 100 request
handlers. Each HTTP/1 request head has a five-second deadline, each complete
object-store lookup has a three-second deadline, and shutdown drains connection
tasks for at most five seconds before aborting and joining them. Each HTTP/1
connection buffer is capped at 16 KiB.

Positive recommendations use a lazy 10,000-cohort cache. Missing and defective
artifacts use a separate 256-cohort cache so arbitrary missing cohort names
cannot evict valid recommendations. Both caches have a 60-second TTL.
Concurrent misses for the same cohort share one object-store GET; transient
storage failures are not cached. Cache, connection, concurrency, and deadline
limits are configurable; the small HTTP/1 protocol buffer cap is fixed.

The host-agnostic library owns `RecommendationServerOptions`, including its
CLI and environment mappings, defaults, conversion to `Config`, validation,
the accept loop, and bounded shutdown. The collector owns only the decision to
enable the hosted server and flattens the library's remaining options into its
CLI. The server can serve a caller-provided shutdown future or spawn against a
cancellation token, so a future standalone process can reuse the options and
the same `ObjectStore` interface. Generic bounded task shutdown and join
handling lives separately in `vectorseam-runtime` for reuse by either host.

## Out of scope for the collector MVP

- Tuner logic and outputs (`tuner-spec.md`).
- Sampling-rate feedback to the SDK.
- Remote object stores in configuration (the code path is
  `object_store`-generic; only local filesystem is wired up and tested).
- Metrics endpoints, compression, compaction of old windows, retention.
