"""pyperf benchmarks for vector capture hot paths."""

from __future__ import annotations

import array
import dataclasses
import pathlib
import random
import sys


_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PYTHON_DIR = _ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import pyperf  # noqa: E402
import numpy  # noqa: E402

from vectorseam.message import DType  # noqa: E402
from vectorseam.vector_capture import (  # noqa: E402
    AdaptiveSampler,
    CaptureResult,
    ProbabilitySampler,
    VectorCaptureProducer,
    capture_vector,
)


_DEFAULT_DIMENSIONS = (384, 768, 1536, 3072, 4096)
_DEFAULT_PROCESSES = 20
_DEFAULT_VALUES = 7
_FIXED_FRAME_SIZE = 28
_NAME = "benchmark"


class _BenchmarkClock:
    """Mutable clock used to pre-warm adaptive sampler state."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now = round(self.now + seconds, 10)


@dataclasses.dataclass(frozen=True)
class BenchmarkInput:
    """Pre-created inputs for one benchmark dimension."""

    dimension: int
    name: str
    numpy_vector: numpy.ndarray
    vector_view: memoryview
    capture_adaptive_rate_001: VectorCaptureProducer
    capture_adaptive_rate_1: VectorCaptureProducer
    capture_rate_001: VectorCaptureProducer
    capture_rate_1: VectorCaptureProducer
    capture_numpy_rate_1: VectorCaptureProducer


def _make_adaptive_sampler(
    *,
    dimension: int,
    target_samples_per_second: float,
    calls_per_second: float,
) -> AdaptiveSampler:
    """Creates an adaptive sampler pre-warmed at a fixed arrival rate."""
    clock = _BenchmarkClock()
    sampler = AdaptiveSampler(
        target_samples_per_second=target_samples_per_second,
        rng=random.Random(dimension),
        clock=clock,
    )
    interval_seconds = 1.0 / calls_per_second
    for _ in range(int(calls_per_second * 50)):
        sampler.should_sample(_NAME)
        clock.advance(interval_seconds)
    return sampler


def _make_input(dimension: int) -> BenchmarkInput:
    """Creates deterministic benchmark data outside timed functions."""
    vector_array = array.array(
        "f",
        ((index % 1024) / 1024.0 for index in range(dimension)),
    )
    if sys.byteorder != "little":
        vector_array.byteswap()
    numpy_vector = numpy.frombuffer(vector_array, dtype=numpy.dtype("<f4"))
    frame_bytes = _FIXED_FRAME_SIZE + len(_NAME.encode("utf-8"))
    frame_bytes += dimension * DType.F32.byte_size
    return BenchmarkInput(
        dimension=dimension,
        name=_NAME,
        numpy_vector=numpy_vector,
        vector_view=memoryview(vector_array).cast("B"),
        capture_adaptive_rate_001=VectorCaptureProducer(
            sampler=_make_adaptive_sampler(
                dimension=dimension,
                target_samples_per_second=1.0,
                calls_per_second=100.0,
            ),
            max_queue_bytes=frame_bytes,
        ),
        capture_adaptive_rate_1=VectorCaptureProducer(
            sampler=AdaptiveSampler(
                target_samples_per_second=1.0,
                rng=random.Random(dimension),
            ),
            max_queue_bytes=frame_bytes,
        ),
        capture_rate_001=VectorCaptureProducer(
            sampler=ProbabilitySampler(0.01, rng=random.Random(dimension)),
            max_queue_bytes=frame_bytes,
        ),
        capture_rate_1=VectorCaptureProducer(
            sampler=ProbabilitySampler(1.0),
            max_queue_bytes=frame_bytes,
        ),
        capture_numpy_rate_1=VectorCaptureProducer(
            max_queue_bytes=frame_bytes
        ),
    )


def _consume_frame(frame: bytes) -> int:
    """Touches the encoded frame so benchmark return values are used."""
    return len(frame) ^ frame[0] ^ frame[-1]


def _bench_capture(
    benchmark_input: BenchmarkInput,
    producer: VectorCaptureProducer,
) -> int:
    """Captures one packed vector and consumes any queued frame."""
    result = producer.capture_vector(
        benchmark_input.name,
        benchmark_input.vector_view,
        dimension=benchmark_input.dimension,
    )
    if result is CaptureResult.NOT_SAMPLED:
        return 0
    if result is CaptureResult.QUEUE_FULL:
        raise RuntimeError("capture benchmark queue unexpectedly filled")

    frame = producer.try_dequeue()
    if frame is None:
        raise RuntimeError("capture benchmark enqueue produced no frame")
    return _consume_frame(frame)


def _bench_capture_sample_rate_001(benchmark_input: BenchmarkInput) -> int:
    """Hot path with probability sampling at 1%."""
    return _bench_capture(
        benchmark_input,
        benchmark_input.capture_rate_001,
    )


def _bench_capture_adaptive_rate_001(benchmark_input: BenchmarkInput) -> int:
    """Hot path with adaptive sampling warmed to 1%."""
    return _bench_capture(
        benchmark_input,
        benchmark_input.capture_adaptive_rate_001,
    )


def _bench_capture_adaptive_rate_1(benchmark_input: BenchmarkInput) -> int:
    """Hot path with adaptive sampling at probability 1."""
    return _bench_capture(
        benchmark_input,
        benchmark_input.capture_adaptive_rate_1,
    )


def _bench_capture_sample_rate_1(benchmark_input: BenchmarkInput) -> int:
    """Hot path with sampling always enabled."""
    return _bench_capture(
        benchmark_input,
        benchmark_input.capture_rate_1,
    )


def _bench_capture_numpy_sample_rate_1(
    benchmark_input: BenchmarkInput,
) -> int:
    """Convenience capture path with NumPy metadata inference."""
    result = capture_vector(
        benchmark_input.name,
        benchmark_input.numpy_vector,
        producer=benchmark_input.capture_numpy_rate_1,
    )
    if result is CaptureResult.QUEUE_FULL:
        raise RuntimeError("capture benchmark queue unexpectedly filled")

    frame = benchmark_input.capture_numpy_rate_1.try_dequeue()
    if frame is None:
        raise RuntimeError("capture benchmark enqueue produced no frame")
    return _consume_frame(frame)


def _add_cli_args(runner: pyperf.Runner) -> None:
    """Adds benchmark-specific arguments to pyperf's parser."""
    runner.argparser.add_argument(
        "--dimension",
        action="append",
        type=int,
        help="Benchmark one dimension; may be passed multiple times.",
    )
    runner.argparser.add_argument(
        "--dimensions",
        help="Comma-separated dimensions to benchmark.",
    )


def _add_worker_cmdline_args(cmd: list[str], args: object) -> None:
    """Propagates benchmark-specific arguments to pyperf workers."""
    for dimension in getattr(args, "dimension") or ():
        cmd.extend(("--dimension", str(dimension)))
    dimensions_arg = getattr(args, "dimensions") or ""
    if dimensions_arg:
        cmd.extend(("--dimensions", dimensions_arg))


def _selected_dimensions(args: object) -> tuple[int, ...]:
    """Returns selected dimensions, preserving first-seen order."""
    dimensions: list[int] = []
    for dimension in getattr(args, "dimension") or ():
        dimensions.append(dimension)
    dimensions_arg = getattr(args, "dimensions") or ""
    for raw_dimension in dimensions_arg.split(","):
        raw_dimension = raw_dimension.strip()
        if raw_dimension:
            dimensions.append(int(raw_dimension))
    if not dimensions:
        dimensions.extend(_DEFAULT_DIMENSIONS)

    selected: list[int] = []
    seen: set[int] = set()
    for dimension in dimensions:
        if dimension <= 0:
            raise ValueError("dimensions must be positive integers")
        if dimension not in seen:
            selected.append(dimension)
            seen.add(dimension)
    return tuple(selected)


def main() -> None:
    """Runs pyperf benchmarks for vector capture hot paths."""
    runner = pyperf.Runner(
        processes=_DEFAULT_PROCESSES,
        values=_DEFAULT_VALUES,
        add_cmdline_args=_add_worker_cmdline_args,
    )
    _add_cli_args(runner)
    args = runner.parse_args()
    if args.output:
        pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    inputs = [
        _make_input(dimension) for dimension in _selected_dimensions(args)
    ]

    for benchmark_input in inputs:
        dimension = benchmark_input.dimension
        runner.bench_func(
            f"capture_adaptive_rate_0_01_dim_{dimension}",
            _bench_capture_adaptive_rate_001,
            benchmark_input,
        )
        runner.bench_func(
            f"capture_adaptive_rate_1_0_dim_{dimension}",
            _bench_capture_adaptive_rate_1,
            benchmark_input,
        )
        runner.bench_func(
            f"capture_sample_rate_0_01_dim_{dimension}",
            _bench_capture_sample_rate_001,
            benchmark_input,
        )
        runner.bench_func(
            f"capture_sample_rate_1_0_dim_{dimension}",
            _bench_capture_sample_rate_1,
            benchmark_input,
        )
        runner.bench_func(
            f"capture_numpy_sample_rate_1_0_dim_{dimension}",
            _bench_capture_numpy_sample_rate_1,
            benchmark_input,
        )


if __name__ == "__main__":
    main()
