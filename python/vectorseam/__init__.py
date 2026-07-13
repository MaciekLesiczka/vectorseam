"""Client SDK."""

from vectorseam.frame import (
    BufferLike,
    DType,
    encode_vector_frame,
    encode_vector_frame_from_iterable,
)
from vectorseam.vector_capture import (
    AdaptiveSampler,
    CaptureResult,
    ProbabilitySampler,
    SamplingPolicy,
    VectorCaptureProducer,
    capture_vector,
    get_vector_capture_producer,
)
from vectorseam.vector_sender import VectorSocketSender

__all__ = [
    "AdaptiveSampler",
    "BufferLike",
    "CaptureResult",
    "DType",
    "ProbabilitySampler",
    "SamplingPolicy",
    "VectorCaptureProducer",
    "VectorSocketSender",
    "capture_vector",
    "encode_vector_frame",
    "encode_vector_frame_from_iterable",
    "get_vector_capture_producer",
]
