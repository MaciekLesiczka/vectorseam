"""Binary frame packing for vector queries.

Protocol v1 encodes one vector query into a single binary frame:

  frame_len:   u32_le     number of bytes after this field
  magic:       4 bytes    ASCII "VQS1"
  version:     u32_le     1
  dtype:       u32_le
  name_len:    u32_le     UTF-8 byte length of validated cohort name
  dimension:   u32_le
  vector_len:  u32_le
  name:        name_len bytes, UTF-8, no padding
  vector:      vector_len bytes, raw little-endian vector bytes

There is no byte alignment or padding in protocol v1.
"""

from __future__ import annotations

import array
import enum
import re
import struct
import sys
from collections.abc import Iterable
from typing import TypeAlias

BufferLike: TypeAlias = bytes | bytearray | memoryview | array.array

_FIXED_FRAME_HEADER_LEN = 28
_HEADER = struct.Struct("<I4sIIIII")
_MAGIC = b"VQS1"
_MAX_NAME_BYTES = 255
_PLAIN_NAME_SEGMENT = r"[a-z0-9][a-z0-9_-]*"
# Must stay in sync with vectorseam-core's reserved cohort keys.
_RESERVED_COHORT_KEYS = ("window", "part", "cohorts")
_RESERVED_COHORT_KEYS_PATTERN = "|".join(_RESERVED_COHORT_KEYS)
_NAME_SEGMENT_PATTERN = (
    r"(?=[^/]{1,63}(?:/|\Z))(?:"
    rf"{_PLAIN_NAME_SEGMENT}|"
    rf"(?!(?:{_RESERVED_COHORT_KEYS_PATTERN})=)"
    rf"{_PLAIN_NAME_SEGMENT}={_PLAIN_NAME_SEGMENT}"
    r")"
)
_NAME_GRAMMAR_ERROR = (
    "name must match cohort grammar: 1 to 8 '/'-separated segments; each "
    "segment must be either [a-z0-9][a-z0-9_-]* or key=value where key and "
    "value each match [a-z0-9][a-z0-9_-]*; each segment must be 1 to 63 "
    "bytes including '='; whole name must be at most 255 bytes; no empty "
    "segments, leading '/', trailing '/', multiple '=', or reserved keys "
    "window, part, or cohorts"
)
_NAME_PATTERN = re.compile(
    rf"\A{_NAME_SEGMENT_PATTERN}(?:/{_NAME_SEGMENT_PATTERN}){{0,7}}\Z",
    re.ASCII,
)
_U32_MAX = 0xFFFFFFFF
_VERSION = 1


class DType(enum.IntEnum):
    """Protocol v1 vector element dtype values."""

    F32 = (1, 4, "f")
    F64 = (2, 8, "d")
    F16 = (3, 2, None)
    BF16 = (4, 2, None)
    I8 = (5, 1, None)
    U8 = (6, 1, None)

    def __new__(
        cls,
        value: int,
        byte_size: int,
        array_typecode: str | None,
    ) -> DType:
        member = int.__new__(cls, value)
        member._value_ = value
        member.byte_size = byte_size
        member.array_typecode = array_typecode
        return member

    byte_size: int
    array_typecode: str | None


def encode_vector_frame(
    name: str,
    dtype: DType,
    dimension: int,
    vector: BufferLike,
) -> bytes:
    """Encodes already-packed little-endian vector bytes into a v1 frame.

    This is the production marshalling path. Callers that already have
    little-endian vector memory avoid boxed-float traversal and conversion.
    No byte-order conversion is performed. The returned object is immutable
    bytes.

    Example with NumPy:

      vector_le = numpy.ascontiguousarray(
          numpy.asarray(vector, dtype=numpy.dtype("<f4"))
      )
      frame = encode_vector_frame(
          "prod", DType.F32, vector_le.size, memoryview(vector_le)
      )

    The caller is responsible for providing little-endian bytes. The passed
    buffer must be treated as immutable while this function is encoding it; do
    not mutate the array from another reference or thread during the call.

    Args:
        name: Client-defined cohort name. Must be 1 to 8 '/'-separated
            segments; each segment must be either [a-z0-9][a-z0-9_-]* or
            key=value where key and value independently match that same plain
            segment rule; each segment must be 1 to 63 bytes including '='
            and the whole name must be at most 255 bytes. Pair keys `window`,
            `part`, and `cohorts` are reserved.
        dtype: Element dtype of the packed vector bytes.
        dimension: Number of vector elements.
        vector: Raw little-endian vector bytes.

    Returns:
        An immutable bytes object containing the complete binary frame.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If name does not match the cohort grammar, vector is
            empty, byte length does not match dimension and dtype, or a length
            field cannot fit in u32.
    """
    dtype = _coerce_dtype(dtype)
    name_bytes = _encode_name(name)
    _validate_dimension(dimension)
    vector_view = _byte_view(vector)
    _validate_vector_len(dimension, dtype, vector_view.nbytes)
    return _encode_vector_frame_from_view(
        name_bytes, dtype, dimension, vector_view
    )


def encode_vector_frame_from_iterable(
    name: str,
    vector: Iterable[float],
    dtype: DType = DType.F32,
) -> bytes:
    """Encodes a Python float iterable into a protocol v1 binary frame.

    This convenience API is intended for experiments, examples, and small call
    sites. It traverses boxed Python floats and packs them into F32 bytes before
    building the frame. Use `encode_vector_frame` for production call sites
    that already have packed vector memory.

    Args:
        name: Client-defined cohort name. Must be 1 to 8 '/'-separated
            segments; each segment must be either [a-z0-9][a-z0-9_-]* or
            key=value where key and value independently match that same plain
            segment rule; each segment must be 1 to 63 bytes including '='
            and the whole name must be at most 255 bytes. Pair keys `window`,
            `part`, and `cohorts` are reserved.
        vector: Iterable of Python floats to encode as F32.
        dtype: Element dtype to encode. Only DType.F32 is supported here.

    Returns:
        An immutable bytes object containing the complete binary frame.

    Raises:
        TypeError: If an argument has an invalid type or vector cannot be
            encoded as F32.
        ValueError: If name does not match the cohort grammar, vector is
            empty, or a length field cannot fit in u32.
        NotImplementedError: If dtype is recognized but is not DType.F32.
    """
    dtype = _coerce_dtype(dtype)
    if dtype is not DType.F32:
        raise NotImplementedError(
            "encode_vector_frame_from_iterable only supports F32"
        )

    vector_values = _to_f32_array(vector)
    if sys.byteorder != "little":
        vector_values.byteswap()

    return encode_vector_frame(
        name,
        dtype,
        len(vector_values),
        vector_values,
    )


def _coerce_dtype(dtype: DType) -> DType:
    """Validates dtype is a DType value."""
    if not isinstance(dtype, DType):
        raise TypeError("dtype must be a DType")
    return dtype


def _encode_vector_frame_from_view(
    name_bytes: bytes,
    dtype: DType,
    dimension: int,
    vector: memoryview,
) -> bytes:
    """Builds an immutable frame from validated vector bytes."""
    name_len = len(name_bytes)
    vector_len = vector.nbytes

    frame_len = _FIXED_FRAME_HEADER_LEN - 4 + name_len + vector_len
    _validate_u32("frame_len", frame_len)

    header = _HEADER.pack(
        frame_len,
        _MAGIC,
        _VERSION,
        dtype.value,
        name_len,
        dimension,
        vector_len,
    )
    return b"".join((header, name_bytes, vector))


def _to_f32_array(vector: Iterable[float]) -> array.array:
    """Converts a float iterable to an F32 array for binary export."""
    if not isinstance(vector, Iterable):
        raise TypeError("vector must be an iterable")

    try:
        vector_values = array.array("f", vector)
    except TypeError as error:
        raise TypeError("vector must contain F32 values") from error
    if vector_values.itemsize != DType.F32.byte_size:
        raise ValueError("F32 item size must be 4 bytes")
    if len(vector_values) == 0:
        raise ValueError("vector must not be empty")
    _validate_dimension(len(vector_values))
    return vector_values


def _byte_view(vector: BufferLike) -> memoryview:
    """Returns a byte-oriented memoryview over contiguous vector data."""
    try:
        return memoryview(vector).cast("B")
    except TypeError as error:
        raise TypeError("vector must expose contiguous bytes") from error


def _encode_name(name: str) -> bytes:
    """Encodes and validates a frame name."""
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    if len(name) > _MAX_NAME_BYTES or _NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(_NAME_GRAMMAR_ERROR)
    name_bytes = name.encode("utf-8")
    _validate_u32("name_len", len(name_bytes))
    return name_bytes


def _validate_dimension(dimension: int) -> None:
    """Raises if dimension is invalid."""
    if not isinstance(dimension, int) or isinstance(dimension, bool):
        raise TypeError("dimension must be an integer")
    if dimension == 0:
        raise ValueError("vector must not be empty")
    _validate_u32("dimension", dimension)


def _validate_u32(field_name: str, value: int) -> None:
    """Raises if value cannot be represented as u32."""
    if not 0 <= value <= _U32_MAX:
        raise ValueError(f"{field_name} must fit in u32")


def _validate_vector_len(dimension: int, dtype: DType, vector_len: int) -> None:
    """Raises if vector_len does not match dimension and dtype."""
    expected_vector_len = dimension * dtype.byte_size
    if vector_len != expected_vector_len:
        raise ValueError("vector_len must match dimension * dtype_size")
    _validate_u32("vector_len", vector_len)


__all__ = [
    "BufferLike",
    "DType",
    "encode_vector_frame",
    "encode_vector_frame_from_iterable",
]
