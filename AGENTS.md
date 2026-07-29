
# Coding guidance

Python code should strictly follow https://google.github.io/styleguide/pyguide.html
Python uses uv.

For public SDK APIs, perform explicit runtime validation at the boundary even
when arguments are type annotated. Prefer clear `TypeError` or `ValueError`
messages over incidental downstream failures.

Prefer simple, hot-path-oriented SDK APIs. Avoid adding abstractions unless
they clearly serve production use.

When changing code, remove or wire up obsolete declarations, counters,
branches, configuration, tests, and documentation. Do not leave dead code or
permanently-zero observability fields behind.

For collector and storage implementation, treat resource budgets as hard
constraints. Carefully account for temporary copies, in-flight IO payloads,
queue capacity, and per-cohort/global memory caps before introducing
concurrency or buffering. Prefer simple single-flight designs when they make
the memory bound obvious; document any intentional tradeoff.

For vector capture / producer-style SDK code:
- Keep the hot path minimal: sampling -> marshalling -> non-blocking enqueue.
- Sampling must happen before marshalling.
- Do not infer vector dimension in hot-path APIs; require `dimension`
  explicitly.
- Let lower-level marshalling code validate dtype, dimension, name, and vector
  length when it already owns that validation. Do not duplicate validation in
  wrapper APIs unless the wrapper introduces a new boundary or invariant.
- Return simple enums for capture outcomes when that is enough.
- Avoid result dataclasses, stats, or counter abstractions unless callers need
  them.
- Queue bounds should be by total bytes, not item count.
- Queue-full behavior should drop, not block.
- Store immutable `bytes` frames from production marshalling.
- Keep only minimal queue access needed by future sender code, for example
  `try_dequeue()`. Do not add broad drain helpers unless needed.
- The capture producer should be process-wide by default, with a singleton
  accessor around its underlying queue.
- Document thread-safety explicitly when a producer is safe for concurrent
  calls. Mention that queue state is protected by an OS-level mutex.
- Do not write dummy variable usage like `del name`; just leave unused protocol
  parameters alone.

For protocols and sampling:
- `typing.Protocol` is acceptable for structural injection points such as
  sampling policies.
- If a concrete default implementation exists, explicitly subclassing the
  protocol can make intent clearer.
- Use `random.Random` for non-security sampling and allow seeded RNGs for
  deterministic tests.

For benchmarks:
- For all new Python SDK production functionality, always consider whether a
  benchmark should be added.
- If adding a benchmark, put it in a separate benchmark module matching the
  production module, for example `bench_<module>.py`.
- Do not mix unrelated production modules into an existing benchmark file.
- Keep benchmark dimensions consistent with existing benchmark dimensions
  unless there is a strong reason not to.
- Benchmark capture hot paths separately for important sampling modes, such as
  low-rate sampling and always-capture.
- If a benchmark enqueues frames, consume/dequeue successful captures inside
  the benchmark so it remains a steady-state hot-path benchmark, not an
  accidental queue-full benchmark.
- Make report targets graceful when expected benchmark JSON output is missing;
  print the generation command instead of failing.

# Other

When updating readme, keep it concise and short. Favor adding tasks to Makefile
rather than bloated instructions and lengthy commands.

Project uses uv for python, and you should use it instead of raw pip3 or
python3 calls.


# Rust

Use the least visibility that satisfies the current design: private by default,
`pub(crate)` only for cross-module crate internals, and `pub` only for an
intentional external API. Central `pub(crate)` APIs may keep Rustdoc when it
helps reviewers understand the module boundary.

## Rust Tokio Development Rules

When writing or reviewing Rust code that uses Tokio, follow these rules.

### Runtime and task model

* Treat an async function call as construction of a lazy `Future`. Calling an async function does not independently schedule or execute it.
* Use `.await` to poll a future as part of the current task.
* Treat `.await` as suspension of the current task, not blocking of the OS thread. Code following `.await` still cannot run until the awaited future completes.
* Use `tokio::spawn` only when work should run as an independently scheduled task.
* Do not assume a spawned task executes before the next source-code statement. Spawning makes it eligible for scheduling; execution order remains nondeterministic.
* Do not assume a Tokio task has a dedicated OS thread. Tokio multiplexes many tasks over runtime worker threads.
* Retain and inspect `JoinHandle` values for important tasks. Do not silently detach critical work.
* Handle both layers of task errors: `JoinError` from the task itself and the result returned by the task.
* Use `JoinSet` to own and supervise a dynamic collection of tasks. Remember that `join_next()` yields tasks in completion order, not insertion order.
* Avoid unstructured spawning. Every long-lived task should have an identifiable owner, shutdown mechanism, and completion path.

### Blocking and CPU-intensive work

* Never call blocking APIs such as `std::thread::sleep` in ordinary async tasks.
* Use `tokio::time::sleep` for asynchronous delays.
* Move unavoidable blocking operations to `tokio::task::spawn_blocking`.
* Do not use `spawn_blocking` as an unlimited CPU work queue. Bound concurrency with a semaphore or use a dedicated CPU thread pool such as Rayon.
* Remember that started `spawn_blocking` closures cannot generally be cancelled.
* Break long CPU loops into bounded work or move them off runtime workers.
* Investigate any async function that performs substantial work without reaching `.await`; it may starve other tasks.

### Shared state and locks

* Use `Arc<T>` when multiple independently owned tasks or threads need access to the same value.
* Understand that `Arc::clone` clones only the reference-counted handle; it does not clone the underlying value.
* Use `Arc<Mutex<T>>` only when shared mutable ownership is genuinely required.
* Prefer message passing and single-owner tasks for I/O resources, state machines, and complex asynchronous resources.
* Prefer `std::sync::Mutex` for short, purely synchronous critical sections when its guard never crosses `.await`.
* Use `tokio::sync::Mutex` when asynchronous lock acquisition or holding the guard across `.await` is actually required.
* Avoid holding any mutex or read/write lock guard across `.await`.
* Compute asynchronous inputs before locking, lock only to update or read state, and release the guard before the next `.await`.
* Keep critical sections short and free of network calls, file operations, sleeps, channel waits, and expensive computation.
* Never acquire multiple locks in inconsistent orders.
* Do not assume async mutexes prevent deadlocks; they only avoid blocking the runtime thread while waiting.

### Channels and backpressure

* Prefer bounded channels unless unbounded memory growth is explicitly acceptable and justified.
* Choose channel types according to semantics:

  * `mpsc` for multiple producers and one consumer.
  * `oneshot` for one response or one one-time signal.
  * `watch` for the latest persistent state.
  * `broadcast` for events delivered to all currently subscribed receivers.
* Use bounded `mpsc` capacity as explicit backpressure.
* Ensure all sender clones are eventually dropped when channel closure is used to terminate a receiver.
* Do not retain accidental sender clones that keep receiver loops alive indefinitely.
* Treat channel send failures as lifecycle information rather than automatically unwrapping them.
* Consider `reserve()` when cancellation of `mpsc::send` could lose an important place in the fairness queue.
* Use a task plus `mpsc` and `oneshot` replies when one task should exclusively own an asynchronous resource.

### Cancellation safety

* Assume that a future losing a `tokio::select!` race is immediately dropped.
* Before placing an operation in `select!`, especially inside a loop, check its documented cancellation-safety guarantees.
* A cancellation-safe operation must not silently consume data or lose externally significant progress when dropped before completion.
* Treat partial writes, partially completed transactions, removed-but-not-returned queue entries, and discarded parser state as cancellation hazards.
* Do not confuse memory safety with cancellation safety. Cancellation-unsafe Rust code may be memory-safe while still losing data or corrupting protocol state.
* Keep durable progress outside transient futures when an operation may be repeatedly cancelled and recreated.
* Pin and continue polling the same future when abandoning and reconstructing it would lose progress.
* Move non-interruptible logical operations into owned tasks and decide explicitly whether shutdown should await, abort, or detach them.
* Use database transactions, idempotency keys, checkpoints, or rollback mechanisms for multi-step operations that must remain consistent.
* Put synchronous cleanup in RAII guards. Arrange an explicit asynchronous shutdown phase for cleanup that requires `.await`.
* Do not abort tasks while they may be performing cancellation-unsafe work unless forced termination is intentionally accepted.

### `select!` usage

* Use `tokio::select!` for racing asynchronous events, shutdown signals, timeouts, and multiplexed inputs.
* Remember that non-winning branches are dropped.
* Ensure continuously ready branches cannot starve shutdown or maintenance branches.
* Do not rely on undocumented branch polling order.
* Keep branch bodies small; delegate substantial work to functions after selecting the event.
* Do not place a newly constructed cancellation-unsafe future in every loop iteration.
* Handle channel closure explicitly rather than repeatedly selecting a permanently closed receiver.
* Use branch preconditions carefully and ensure an `else` path exists when all branches may become disabled.

### Timeouts

* Apply timeouts at meaningful operation boundaries, not blindly around arbitrary multi-step workflows.
* Understand that `tokio::time::timeout` cancels its wrapped future by dropping it when the deadline expires.
* Verify that timed operations are cancellation-safe or design recovery for partial progress.
* Distinguish connection timeout, request timeout, idle timeout, and total workflow deadline.
* Propagate a shared deadline through nested operations where appropriate instead of restarting a full timeout at every layer.
* Do not assume a timeout can interrupt code that does not yield to the runtime.

### Networking and asynchronous I/O

* Treat TCP as a byte stream with no message boundaries.
* Implement explicit framing, such as length prefixes, delimiters, or a documented codec.
* Handle partial reads and partial writes.
* Use `write_all` only after considering cancellation behavior and protocol consequences of partial output.
* Bound frame sizes and input buffers to prevent uncontrolled memory use.
* Treat zero-byte reads as end-of-stream where appropriate.
* Separate connection acceptance, per-connection handling, and application-level processing.
* In resource-management loops such as accepting connections, do not propagate
  operational errors with `?` when doing so bypasses cleanup, draining, or
  task shutdown. Log the OS error, apply a bounded backoff, and continue
  unless an intentional shutdown path will run all cleanup.
* Bound concurrent connections and requests with semaphores, queues, or admission control.
* Do not spawn an unlimited task for every untrusted input without resource limits.
* Apply idle and request timeouts deliberately.
* Ensure malformed input cannot leave parser or protocol state inconsistent.

### Task ownership and failures

* Do not ignore task panics or returned errors.
* Use supervised task structures for long-lived workers.
* Decide whether failure of one task should:

  * be logged and isolated,
  * trigger a restart,
  * cancel sibling tasks, or
  * shut down the application.
* Do not use `unwrap` on expected network, channel, timeout, or shutdown errors.
* Reserve `unwrap` and `expect` for invariants whose violation is a programming error, and include a useful invariant description.
* Do not let detached background tasks hold resources indefinitely.
* Ensure every spawned task eventually completes, is cancelled, or is intentionally process-lifetime work.

### Graceful shutdown

* Implement shutdown in three phases: detect, notify, and wait.
* At the application boundary, convert OS signals such as Ctrl+C or SIGTERM into one internal cancellation mechanism.
* Prefer `tokio_util::sync::CancellationToken` for cancellation across a task tree.
* Use child cancellation tokens when subcomponents need hierarchical ownership.
* Use channel closure when shutdown naturally means “no more work.”
* Use `watch` when tasks need a persistent lifecycle state such as running, draining, or stopping.
* Use `oneshot` for one receiver and `broadcast` only when event semantics are required.
* On shutdown, stop accepting new work before cancelling workers that process existing work.
* Decide explicitly whether buffered and in-flight work should be drained, rejected, retried, or abandoned.
* Await task completion so cleanup and errors are observed.
* Impose a finite graceful-shutdown deadline.
* After the deadline, abort remaining async tasks and continue joining them to observe termination.
* Treat task abortion as a forced fallback, not the normal shutdown path.
* Remember that blocking tasks may outlive async cancellation and require separate process or thread-level strategies.
* Flush important logs, telemetry, and buffered output before process termination when possible.

### Concurrency limits

* Never create unbounded concurrency from untrusted or potentially large input.
* Use `Semaphore`, bounded worker queues, stream concurrency limits, or fixed worker pools.
* Acquire concurrency permits before spawning when spawning itself would otherwise become unbounded.
* Keep permits alive for exactly the duration of the protected operation.
* Account for nested concurrency so independent limits do not multiply into excessive load.
* Apply limits to connections, requests, database operations, filesystem work, and blocking tasks separately where necessary.

### Lifetimes and spawned data

* Remember that `tokio::spawn` generally requires the spawned future to be `Send + 'static`.
* Interpret `'static` as “contains no borrowed data that may expire before the task,” not “must live forever.”
* Use `async move` to transfer owned values into spawned tasks.
* Clone `Arc`, channel senders, cancellation tokens, and other handles before moving them into a task.
* Do not solve ownership issues by leaking values or converting arbitrary data into `'static` references.
* Prefer owned data at task boundaries.

### Testing and diagnostics

* Use `#[tokio::test]` for asynchronous tests.
* Pause and advance Tokio time in timer-heavy tests instead of waiting in real time.
* Test cancellation at every significant `.await` boundary in stateful workflows.
* Test channel closure, receiver loss, sender loss, timeouts, task panics, and forced shutdown.
* Test concurrency limits under more input than the configured capacity.
* Use structured `tracing` spans and events instead of relying solely on `println!`.
* Include task, request, connection, and operation identifiers in diagnostics.
* Instrument queue time, operation time, timeout counts, task failures, and shutdown duration.
* Investigate tasks that remain busy for long periods without yielding.

### Review checklist

Before accepting Tokio code, verify:

1. No runtime worker is blocked by synchronous waiting or heavy CPU work.
2. Every spawned task has ownership, error handling, and shutdown semantics.
3. Concurrency and channel capacity are bounded.
4. Locks are short-lived and do not cross `.await` without a documented reason.
5. Every `select!` and timeout operation has been evaluated for cancellation safety.
6. Network protocols handle partial I/O and explicit framing.
7. Shutdown stops new work, notifies tasks, drains or cancels in-flight work, and has a deadline.
8. Important task results and panics are observed.
9. Shared state uses the simplest safe ownership model.
10. Transient resource-management errors do not accidentally bypass cleanup, draining, or task joins.
11. Tests cover cancellation, timeout, overload, channel closure, and shutdown behavior.
