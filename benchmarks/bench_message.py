"""pyperf benchmarks for vector message marshalling."""

from __future__ import annotations

import array
import dataclasses
import pathlib
import struct
import sys
import zlib
from collections.abc import Sequence


_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PYTHON_DIR = _ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import pyperf  # noqa: E402

from vectorseam.message import (  # noqa: E402
    DType,
    encode_vector_message,
    encode_vector_message_from_list,
)


_DEFAULT_DIMENSIONS = (384, 768, 1536, 3072, 4096)
_DEFAULT_PROCESSES = 20
_DEFAULT_VALUES = 7
_FIXED_FRAME_SIZE = 28
_MAGIC = b"VQS1"
_NAME = "benchmark"
_U32 = struct.Struct("<I")
_U32_MAX = 0xFFFFFFFF
_VERSION = 1


@dataclasses.dataclass(frozen=True)
class BenchmarkInput:
    """Pre-created inputs for one benchmark dimension."""

    dimension: int
    name: str
    name_bytes: bytes
    vector_list: list[float]
    vector_view: memoryview


def _make_input(dimension: int) -> BenchmarkInput:
    """Creates deterministic benchmark data outside timed functions."""
    vector_list = [(index % 1024) / 1024.0 for index in range(dimension)]
    vector_array = array.array("f", vector_list)
    if sys.byteorder != "little":
        vector_array.byteswap()
    return BenchmarkInput(
        dimension=dimension,
        name=_NAME,
        name_bytes=_NAME.encode("utf-8"),
        vector_list=vector_list,
        vector_view=memoryview(vector_array).cast("B"),
    )


def _encode_bytearray(
    name_bytes: bytes,
    dtype: DType,
    dimension: int,
    vector: memoryview,
) -> bytearray:
    """Mirrors production marshalling but skips the final bytes() copy."""
    name_len = len(name_bytes)
    vector_len = vector.nbytes
    frame_len = _FIXED_FRAME_SIZE - 4 + name_len + vector_len

    frame = bytearray(_FIXED_FRAME_SIZE + name_len + vector_len)
    _U32.pack_into(frame, 0, frame_len)
    frame[4:8] = _MAGIC
    _U32.pack_into(frame, 8, _VERSION)
    _U32.pack_into(frame, 12, dtype.value)
    _U32.pack_into(frame, 16, name_len)
    _U32.pack_into(frame, 20, dimension)
    _U32.pack_into(frame, 24, vector_len)
    frame[_FIXED_FRAME_SIZE : _FIXED_FRAME_SIZE + name_len] = name_bytes
    frame[_FIXED_FRAME_SIZE + name_len :] = vector
    return frame


def _parse_header(frame: bytes | bytearray) -> dict[str, int | bytes]:
    """Parses the current fixed v1 header fields used by validation."""
    return {
        "frame_len": _U32.unpack_from(frame, 0)[0],
        "magic": bytes(frame[4:8]),
        "version": _U32.unpack_from(frame, 8)[0],
        "dtype": _U32.unpack_from(frame, 12)[0],
        "name_len": _U32.unpack_from(frame, 16)[0],
        "dimension": _U32.unpack_from(frame, 20)[0],
        "vector_len": _U32.unpack_from(frame, 24)[0],
    }


def _validate_frame(
    frame: bytes | bytearray,
    dtype: DType,
    dimension: int,
    name_bytes: bytes,
) -> None:
    """Checks that a current v1 frame is structurally valid."""
    header = _parse_header(frame)
    expected_vector_len = dimension * dtype.byte_size
    expected_frame_len = _FIXED_FRAME_SIZE - 4 + len(name_bytes)
    expected_frame_len += expected_vector_len
    if len(frame) - 4 != expected_frame_len:
        raise ValueError("encoded frame length is inconsistent")
    if header["frame_len"] != expected_frame_len:
        raise ValueError("frame_len field is inconsistent")
    if header["magic"] != _MAGIC:
        raise ValueError("magic field is invalid")
    if header["version"] != _VERSION:
        raise ValueError("version field is invalid")
    if header["dtype"] != dtype.value:
        raise ValueError("dtype field is invalid")
    if header["name_len"] != len(name_bytes):
        raise ValueError("name_len field is invalid")
    if header["dimension"] != dimension:
        raise ValueError("dimension field is invalid")
    if header["vector_len"] != expected_vector_len:
        raise ValueError("vector_len field is invalid")


def _validate_inputs(inputs: Sequence[BenchmarkInput]) -> None:
    """Runs lightweight correctness checks before pyperf registration."""
    for benchmark_input in inputs:
        dtype = DType.F32
        dimension = benchmark_input.dimension
        name_bytes = benchmark_input.name_bytes
        production_list = encode_vector_message_from_list(
            benchmark_input.name,
            benchmark_input.vector_list,
            dtype,
        )
        production_view = encode_vector_message(
            benchmark_input.name,
            dtype,
            dimension,
            benchmark_input.vector_view,
        )
        bytearray_frame = _encode_bytearray(
            name_bytes,
            dtype,
            dimension,
            benchmark_input.vector_view,
        )

        _validate_frame(production_list, dtype, dimension, name_bytes)
        _validate_frame(production_view, dtype, dimension, name_bytes)
        _validate_frame(bytearray_frame, dtype, dimension, name_bytes)


def _consume_frame(frame: bytes | bytearray) -> int:
    """Touches the encoded frame so benchmark return values are used."""
    return len(frame) ^ frame[0] ^ frame[-1]


def _consume_frame_with_crc(frame: bytes, crc32: int) -> int:
    """Touches the encoded frame and crc value so both are used."""
    return _consume_frame(frame) ^ crc32


def _bench_list_current(benchmark_input: BenchmarkInput) -> int:
    """Experimental convenience path: traverses list[float] and packs F32."""
    frame = encode_vector_message_from_list(
        benchmark_input.name,
        benchmark_input.vector_list,
        DType.F32,
    )
    return _consume_frame(frame)


def _bench_memoryview_current(benchmark_input: BenchmarkInput) -> int:
    """Primary path: copies into an immutable contiguous bytes frame."""
    frame = encode_vector_message(
        benchmark_input.name,
        DType.F32,
        benchmark_input.dimension,
        benchmark_input.vector_view,
    )
    return _consume_frame(frame)


def _bench_memoryview_bytearray(benchmark_input: BenchmarkInput) -> int:
    """Benchmark-only proof of final bytes() conversion overhead."""
    frame = _encode_bytearray(
        benchmark_input.name_bytes,
        DType.F32,
        benchmark_input.dimension,
        benchmark_input.vector_view,
    )
    return _consume_frame(frame)


def _bench_memoryview_with_crc(benchmark_input: BenchmarkInput) -> int:
    """Primary path plus a CRC32 scan over the returned bytes."""
    frame = encode_vector_message(
        benchmark_input.name,
        DType.F32,
        benchmark_input.dimension,
        benchmark_input.vector_view,
    )
    crc32 = zlib.crc32(frame) & _U32_MAX
    return _consume_frame_with_crc(frame, crc32)


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
    """Runs pyperf benchmarks for vector message marshalling."""
    runner = pyperf.Runner(
        processes=_DEFAULT_PROCESSES,
        values=_DEFAULT_VALUES,
        add_cmdline_args=_add_worker_cmdline_args,
    )
    _add_cli_args(runner)
    args = runner.parse_args()
    if args.output:
        pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    inputs = [_make_input(dimension) for dimension in _selected_dimensions(args)]
    _validate_inputs(inputs)

    for benchmark_input in inputs:
        dimension = benchmark_input.dimension
        runner.bench_func(
            f"message_list_f32_dim_{dimension}",
            _bench_list_current,
            benchmark_input,
        )
        runner.bench_func(
            f"message_memoryview_dim_{dimension}",
            _bench_memoryview_current,
            benchmark_input,
        )
        runner.bench_func(
            f"message_memoryview_bytearray_dim_{dimension}",
            _bench_memoryview_bytearray,
            benchmark_input,
        )
        runner.bench_func(
            f"message_memoryview_dim_with_crc_dim_{dimension}",
            _bench_memoryview_with_crc,
            benchmark_input,
        )


if __name__ == "__main__":
    main()
