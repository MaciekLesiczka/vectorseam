"""Hot-path vector capture producer.

This module covers capture, sampling, marshalling, and non-blocking enqueue.
It does not implement socket sending, background sender threads, retry logic,
collector communication, or IPC transport.

The primary capture path expects already-packed little-endian vector data.
Sampling happens before marshalling. Sampled vectors are marshalled into the
immutable ``bytes`` frames returned by ``encode_vector_message`` and stored in
a memory-bounded queue. The queue is bounded by total queued bytes; when a
frame would exceed capacity, capture drops it. ``VectorCaptureProducer`` is
thread-safe; queue state is protected by an OS-level mutex. Use
``get_vector_capture_producer`` for the process-wide producer instance.
"""

from __future__ import annotations

import enum
import random
import sys
import threading
from collections import deque
from typing import Protocol

import numpy

from vectorseam.message import BufferLike, DType, encode_vector_message

_DEFAULT_MAX_QUEUE_BYTES = 16 * 1024 * 1024


class SamplingPolicy(Protocol):
    """Sampling policy used by ``VectorCaptureProducer``."""

    def should_sample(self, name: str) -> bool:
        """Returns whether a vector named ``name`` should be captured."""


class ProbabilitySampler(SamplingPolicy):
    """Samples captures with a fixed probability."""

    def __init__(
        self,
        sample_rate: float = 1.0,
        rng: random.Random | None = None,
    ) -> None:
        """Initializes a probability sampler.

        Args:
            sample_rate: Probability in the inclusive range [0.0, 1.0].
            rng: Optional random number generator.

        Raises:
            TypeError: If sample_rate is not numeric or rng is invalid.
            ValueError: If sample_rate is outside [0.0, 1.0].
        """
        if isinstance(sample_rate, bool) or not isinstance(
            sample_rate, int | float
        ):
            raise TypeError("sample_rate must be a number")
        sample_rate = float(sample_rate)
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError("sample_rate must be between 0.0 and 1.0")
        if rng is not None and not isinstance(rng, random.Random):
            raise TypeError("rng must be a random.Random instance")

        self._sample_rate = sample_rate
        self._rng = rng if rng is not None else random.Random()
        self._lock = threading.Lock()

    @property
    def sample_rate(self) -> float:
        """Configured sampling probability."""
        return self._sample_rate

    def should_sample(self, name: str) -> bool:
        """Returns whether a capture should be sampled."""
        if self._sample_rate == 1.0:
            return True
        if self._sample_rate == 0.0:
            return False

        with self._lock:
            return self._rng.random() < self._sample_rate


class CaptureResult(enum.Enum):
    """Outcome of a capture attempt."""

    ENQUEUED = "enqueued"
    NOT_SAMPLED = "not_sampled"
    QUEUE_FULL = "queue_full"


class VectorCaptureProducer:
    """Thread-safe producer for non-blocking byte-bounded vector capture."""

    def __init__(
        self,
        *,
        sampler: SamplingPolicy | None = None,
        max_queue_bytes: int = _DEFAULT_MAX_QUEUE_BYTES,
    ) -> None:
        """Initializes a vector capture producer.

        Args:
            sampler: Sampling policy. When omitted, all captures are sampled.
            max_queue_bytes: Maximum total bytes allowed in the queue.

        Raises:
            TypeError: If sampler or max_queue_bytes has an invalid type.
            ValueError: If max_queue_bytes is not positive.
        """
        if sampler is not None and not callable(
            getattr(sampler, "should_sample", None)
        ):
            raise TypeError("sampler must provide should_sample(name)")
        if not isinstance(max_queue_bytes, int) or isinstance(
            max_queue_bytes, bool
        ):
            raise TypeError("max_queue_bytes must be an integer")
        if max_queue_bytes <= 0:
            raise ValueError("max_queue_bytes must be greater than 0")

        self._sampler = sampler if sampler is not None else ProbabilitySampler()
        self._max_queue_bytes = max_queue_bytes
        self._queue: deque[bytes] = deque()
        self._lock = threading.Lock()
        self._queued_bytes = 0

    @property
    def queued_bytes(self) -> int:
        """Current total queued frame bytes."""
        with self._lock:
            return self._queued_bytes

    @property
    def queued_frames(self) -> int:
        """Current queued frame count."""
        with self._lock:
            return len(self._queue)

    def capture_vector(
        self,
        name: str,
        vector: BufferLike,
        *,
        dimension: int,
        dtype: DType = DType.F32,
    ) -> CaptureResult:
        """Samples, marshals, and attempts to enqueue a vector frame.

        Args:
            name: Client-defined UTF-8 cohort or stratum name.
            vector: Already-packed little-endian vector bytes.
            dimension: Number of vector elements.
            dtype: Element dtype of the packed vector bytes.

        Returns:
            Capture result indicating whether the frame was enqueued, skipped
            by sampling, or dropped because the queue was full.

        Raises:
            TypeError: If marshalling rejects an argument type.
            ValueError: If marshalling rejects the name, dimension, or vector.
        """
        if not self._sampler.should_sample(name):
            return CaptureResult.NOT_SAMPLED

        frame = encode_vector_message(name, dtype, dimension, vector)
        frame_bytes = len(frame)

        with self._lock:
            if self._queued_bytes + frame_bytes > self._max_queue_bytes:
                return CaptureResult.QUEUE_FULL

            self._queue.append(frame)
            self._queued_bytes += frame_bytes
            return CaptureResult.ENQUEUED

    def try_dequeue(self) -> bytes | None:
        """Returns the next queued frame without blocking, if one exists."""
        with self._lock:
            if not self._queue:
                return None
            frame = self._queue.popleft()
            self._queued_bytes -= len(frame)
            return frame

    def drain(self, max_bytes: int | None = None) -> list[bytes]:
        """Dequeues queued frames without blocking.

        Args:
            max_bytes: Optional byte budget. Frames are never split; if the
                next frame would exceed the budget, draining stops.

        Returns:
            A list of immutable frame bytes.

        Raises:
            TypeError: If max_bytes is not an integer or None.
            ValueError: If max_bytes is negative.
        """
        if max_bytes is not None:
            if not isinstance(max_bytes, int) or isinstance(max_bytes, bool):
                raise TypeError("max_bytes must be an integer or None")
            if max_bytes < 0:
                raise ValueError("max_bytes must be non-negative")

        frames: list[bytes] = []
        drained_bytes = 0
        with self._lock:
            while self._queue:
                next_frame = self._queue[0]
                next_frame_bytes = len(next_frame)
                if (
                    max_bytes is not None
                    and drained_bytes + next_frame_bytes > max_bytes
                ):
                    break
                frames.append(self._queue.popleft())
                drained_bytes += next_frame_bytes
                self._queued_bytes -= next_frame_bytes
        return frames


_PROCESS_PRODUCER = VectorCaptureProducer()


def get_vector_capture_producer() -> VectorCaptureProducer:
    """Returns the process-wide vector capture producer."""
    return _PROCESS_PRODUCER


def capture_vector(
    name: str,
    vector: BufferLike | numpy.ndarray,
    *,
    dimension: int | None = None,
    dtype: DType | None = None,
    producer: VectorCaptureProducer | None = None,
) -> CaptureResult:
    """Captures a vector with the process-wide producer by default.

    This is the paste-friendly SDK API for application call sites. For a
    one-dimensional ``numpy.ndarray``, dimension and dtype are inferred from
    array metadata. Non-NumPy buffers still require an explicit ``dimension``.
    Lower-level marshalling validates name, dimension, dtype, and vector
    length.

    Args:
        name: Client-defined UTF-8 cohort or stratum name.
        vector: One-dimensional NumPy array or already-packed little-endian
            vector bytes.
        dimension: Number of vector elements. Required for non-NumPy buffers.
        dtype: Element dtype of the packed vector bytes. Inferred for NumPy
            arrays when omitted. Defaults to F32 for non-NumPy buffers.
        producer: Optional producer. When omitted, the process-wide producer is
            used.

    Returns:
        Capture result indicating whether the frame was enqueued, skipped by
        sampling, or dropped because the queue was full.

    Raises:
        TypeError: If producer is invalid, or marshalling rejects an argument.
        ValueError: If marshalling rejects the name, dimension, or vector.
    """
    if producer is None:
        producer = get_vector_capture_producer()
    if not isinstance(producer, VectorCaptureProducer):
        raise TypeError("producer must be a VectorCaptureProducer")

    if isinstance(vector, numpy.ndarray):
        vector, dimension, dtype = _normalize_numpy_vector(
            vector,
            dimension=dimension,
            dtype=dtype,
        )
    else:
        if dimension is None:
            raise TypeError("dimension is required for non-NumPy vectors")
        dtype = DType.F32 if dtype is None else _validate_dtype(dtype)

    return producer.capture_vector(
        name,
        vector,
        dimension=dimension,
        dtype=dtype,
    )


def _normalize_numpy_vector(
    vector: numpy.ndarray,
    *,
    dimension: int | None,
    dtype: DType | None,
) -> tuple[numpy.ndarray, int, DType]:
    """Returns a contiguous little-endian NumPy vector and SDK metadata."""
    if vector.ndim != 1:
        raise ValueError("NumPy vector must be one-dimensional")
    if dimension is not None and dimension != vector.size:
        raise ValueError("dimension must match NumPy vector size")

    inferred_dtype = _dtype_from_numpy(vector.dtype)
    dtype = inferred_dtype if dtype is None else _validate_dtype(dtype)
    if dtype is not inferred_dtype:
        raise ValueError("dtype must match NumPy vector dtype")

    target_dtype = _little_endian_numpy_dtype(vector.dtype)
    if vector.dtype != target_dtype:
        vector = vector.astype(target_dtype, copy=False)
    if not vector.flags.c_contiguous:
        vector = numpy.ascontiguousarray(vector)

    return vector, vector.size, dtype


def _dtype_from_numpy(dtype: numpy.dtype) -> DType:
    """Returns the SDK dtype for a supported NumPy dtype."""
    dtype = numpy.dtype(dtype)
    if dtype.kind == "f" and dtype.itemsize == DType.F32.byte_size:
        return DType.F32
    if dtype.kind == "f" and dtype.itemsize == DType.F64.byte_size:
        return DType.F64
    if dtype.kind == "f" and dtype.itemsize == DType.F16.byte_size:
        return DType.F16
    if dtype.kind == "i" and dtype.itemsize == DType.I8.byte_size:
        return DType.I8
    if dtype.kind == "u" and dtype.itemsize == DType.U8.byte_size:
        return DType.U8
    raise TypeError("NumPy vector dtype must be F16, F32, F64, I8, or U8")


def _little_endian_numpy_dtype(dtype: numpy.dtype) -> numpy.dtype:
    """Returns the little-endian equivalent of dtype."""
    dtype = numpy.dtype(dtype)
    if dtype.byteorder in ("<", "|"):
        return dtype
    if dtype.byteorder == "=" and sys.byteorder == "little":
        return dtype
    return dtype.newbyteorder("<")


def _validate_dtype(dtype: DType) -> DType:
    """Raises if dtype is not a DType."""
    if not isinstance(dtype, DType):
        raise TypeError("dtype must be a DType")
    return dtype


__all__ = [
    "CaptureResult",
    "ProbabilitySampler",
    "SamplingPolicy",
    "VectorCaptureProducer",
    "capture_vector",
    "get_vector_capture_producer",
]
