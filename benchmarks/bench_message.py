"""pyperf benchmarks for vector message marshalling."""

from __future__ import annotations

import array
import dataclasses
import pathlib
import struct
import sys
from collections.abc import Sequence


_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PYTHON_DIR = _ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import pyperf  # noqa: E402

from vectorseam.message import (  # noqa: E402
    DType,
    encode_vector_message,
    encode_vector_message_le_bytes,
)


_DEFAULT_DIMENSIONS = (384, 768, 1536, 3072, 4096)
_DEFAULT_PROCESSES = 20
_DEFAULT_VALUES = 7
_FIXED_FRAME_SIZE = 32
_MAGIC = b"VQS1"
_NAME = "benchmark"
_U32 = struct.Struct("<I")
_VERSION = 1


@dataclasses.dataclass(frozen=True)
class BenchmarkInput:
    """Pre-created inputs for one benchmark dimension."""

    dimension: int
    name: str
    name_bytes: bytes
    vector_list: list[float]
    vector_view: memoryview


@dataclasses.dataclass(frozen=True)
class FrameParts:
    """Prototype frame representation that keeps vector storage separate."""

    header: bytes
    name: bytes
    vector: memoryview


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


def _pack_header(
    name_len: int,
    dtype: DType,
    dimension: int,
    vector_len: int,
) -> bytearray:
    """Returns a v1 frame header with crc32 set to zero."""
    frame_len = _FIXED_FRAME_SIZE - 4 + name_len + vector_len
    header = bytearray(_FIXED_FRAME_SIZE)
    _U32.pack_into(header, 0, frame_len)
    header[4:8] = _MAGIC
    _U32.pack_into(header, 8, _VERSION)
    _U32.pack_into(header, 12, 0)
    _U32.pack_into(header, 16, dtype.value)
    _U32.pack_into(header, 20, name_len)
    _U32.pack_into(header, 24, dimension)
    _U32.pack_into(header, 28, vector_len)
    return header


def _build_no_crc_frame(
    name_bytes: bytes,
    dtype: DType,
    dimension: int,
    vector: memoryview,
) -> bytearray:
    """Builds a contiguous frame with crc32 set to zero."""
    name_len = len(name_bytes)
    vector_len = vector.nbytes
    frame = bytearray(_FIXED_FRAME_SIZE + name_len + vector_len)
    frame[:_FIXED_FRAME_SIZE] = _pack_header(
        name_len,
        dtype,
        dimension,
        vector_len,
    )
    frame[_FIXED_FRAME_SIZE : _FIXED_FRAME_SIZE + name_len] = name_bytes
    frame[_FIXED_FRAME_SIZE + name_len :] = vector
    return frame


def _encode_no_crc_bytes(
    name_bytes: bytes,
    dtype: DType,
    dimension: int,
    vector: memoryview,
) -> bytes:
    """Builds a contiguous frame, skips crc32, and returns immutable bytes."""
    return bytes(_build_no_crc_frame(name_bytes, dtype, dimension, vector))


def _encode_no_crc_bytearray(
    name_bytes: bytes,
    dtype: DType,
    dimension: int,
    vector: memoryview,
) -> bytearray:
    """Builds a contiguous frame, skips crc32, and returns the bytearray."""
    return _build_no_crc_frame(name_bytes, dtype, dimension, vector)


def _encode_frame_parts_no_vector_copy(
    name_bytes: bytes,
    dtype: DType,
    dimension: int,
    vector: memoryview,
) -> FrameParts:
    """Creates header/name only and retains the original vector memoryview."""
    header = bytes(_pack_header(len(name_bytes), dtype, dimension, vector.nbytes))
    return FrameParts(header=header, name=name_bytes, vector=vector)


def _parse_header(frame: bytes | bytearray) -> dict[str, int | bytes]:
    """Parses the fixed v1 header fields used by validation."""
    return {
        "frame_len": _U32.unpack_from(frame, 0)[0],
        "magic": bytes(frame[4:8]),
        "version": _U32.unpack_from(frame, 8)[0],
        "dtype": _U32.unpack_from(frame, 16)[0],
        "name_len": _U32.unpack_from(frame, 20)[0],
        "dimension": _U32.unpack_from(frame, 24)[0],
        "vector_len": _U32.unpack_from(frame, 28)[0],
    }


def _validate_frame(
    frame: bytes | bytearray,
    dtype: DType,
    dimension: int,
    name_bytes: bytes,
    *,
    require_contiguous: bool = True,
) -> None:
    """Checks that a benchmark frame has the expected v1 structure."""
    header = _parse_header(frame)
    expected_vector_len = dimension * dtype.byte_size
    expected_frame_len = _FIXED_FRAME_SIZE - 4 + len(name_bytes)
    expected_frame_len += expected_vector_len
    if header["frame_len"] != expected_frame_len:
        raise ValueError("frame_len field is inconsistent")
    if require_contiguous and len(frame) - 4 != expected_frame_len:
        raise ValueError("encoded frame length is inconsistent")
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


def _validate_frame_parts(
    frame_parts: FrameParts,
    dtype: DType,
    dimension: int,
    name_bytes: bytes,
) -> None:
    """Checks that frame-parts metadata describes a valid v1 frame."""
    _validate_frame(
        frame_parts.header,
        dtype,
        dimension,
        name_bytes,
        require_contiguous=False,
    )
    if frame_parts.name != name_bytes:
        raise ValueError("frame-parts name is invalid")
    if frame_parts.vector.nbytes != dimension * dtype.byte_size:
        raise ValueError("frame-parts vector length is invalid")
    total_len = len(frame_parts.header) + len(frame_parts.name)
    total_len += frame_parts.vector.nbytes
    if total_len - 4 != _U32.unpack_from(frame_parts.header, 0)[0]:
        raise ValueError("frame-parts total length is inconsistent")


def _validate_inputs(inputs: Sequence[BenchmarkInput]) -> None:
    """Runs lightweight correctness checks before pyperf registration."""
    for benchmark_input in inputs:
        dtype = DType.F32
        dimension = benchmark_input.dimension
        name_bytes = benchmark_input.name_bytes
        production_list = encode_vector_message(
            benchmark_input.name,
            benchmark_input.vector_list,
            dtype,
        )
        production_view = encode_vector_message_le_bytes(
            benchmark_input.name,
            dtype,
            dimension,
            benchmark_input.vector_view,
        )
        no_crc_bytes = _encode_no_crc_bytes(
            name_bytes,
            dtype,
            dimension,
            benchmark_input.vector_view,
        )
        no_crc_bytearray = _encode_no_crc_bytearray(
            name_bytes,
            dtype,
            dimension,
            benchmark_input.vector_view,
        )
        frame_parts = _encode_frame_parts_no_vector_copy(
            name_bytes,
            dtype,
            dimension,
            benchmark_input.vector_view,
        )

        _validate_frame(production_list, dtype, dimension, name_bytes)
        _validate_frame(production_view, dtype, dimension, name_bytes)
        _validate_frame(no_crc_bytes, dtype, dimension, name_bytes)
        _validate_frame(no_crc_bytearray, dtype, dimension, name_bytes)
        _validate_frame_parts(frame_parts, dtype, dimension, name_bytes)


def _consume_frame(frame: bytes | bytearray) -> int:
    """Touches the encoded frame so benchmark return values are used."""
    return len(frame) ^ frame[0] ^ frame[-1]


def _consume_frame_parts(frame_parts: FrameParts) -> int:
    """Touches each frame part without joining vector bytes."""
    return (
        len(frame_parts.header)
        ^ len(frame_parts.name)
        ^ frame_parts.vector.nbytes
        ^ frame_parts.header[0]
        ^ frame_parts.vector[0]
    )


def _bench_list_current(benchmark_input: BenchmarkInput) -> int:
    """Current convenience path: traverses list[float] and packs values."""
    frame = encode_vector_message(
        benchmark_input.name,
        benchmark_input.vector_list,
        DType.F32,
    )
    return _consume_frame(frame)


def _bench_memoryview_current(benchmark_input: BenchmarkInput) -> int:
    """Current path: copies vector bytes, computes crc32, returns bytes."""
    frame = encode_vector_message_le_bytes(
        benchmark_input.name,
        DType.F32,
        benchmark_input.dimension,
        benchmark_input.vector_view,
    )
    return _consume_frame(frame)


def _bench_memoryview_no_crc_bytes(benchmark_input: BenchmarkInput) -> int:
    """Prototype: copies vector bytes and returns bytes, but skips crc32."""
    frame = _encode_no_crc_bytes(
        benchmark_input.name_bytes,
        DType.F32,
        benchmark_input.dimension,
        benchmark_input.vector_view,
    )
    return _consume_frame(frame)


def _bench_memoryview_no_crc_bytearray(benchmark_input: BenchmarkInput) -> int:
    """Prototype: skips crc32 and the final bytearray-to-bytes copy."""
    frame = _encode_no_crc_bytearray(
        benchmark_input.name_bytes,
        DType.F32,
        benchmark_input.dimension,
        benchmark_input.vector_view,
    )
    return _consume_frame(frame)


def _bench_frame_parts_no_vector_copy(benchmark_input: BenchmarkInput) -> int:
    """Prototype lower bound: creates header/name without copying vector bytes."""
    frame_parts = _encode_frame_parts_no_vector_copy(
        benchmark_input.name_bytes,
        DType.F32,
        benchmark_input.dimension,
        benchmark_input.vector_view,
    )
    return _consume_frame_parts(frame_parts)


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
            f"message_memoryview_current_dim_{dimension}",
            _bench_memoryview_current,
            benchmark_input,
        )
        runner.bench_func(
            f"message_memoryview_no_crc_bytes_dim_{dimension}",
            _bench_memoryview_no_crc_bytes,
            benchmark_input,
        )
        runner.bench_func(
            f"message_memoryview_no_crc_bytearray_dim_{dimension}",
            _bench_memoryview_no_crc_bytearray,
            benchmark_input,
        )
        runner.bench_func(
            f"message_frame_parts_no_vector_copy_dim_{dimension}",
            _bench_frame_parts_no_vector_copy,
            benchmark_input,
        )


if __name__ == "__main__":
    main()
