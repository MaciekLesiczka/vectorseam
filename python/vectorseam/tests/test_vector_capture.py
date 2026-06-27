"""Tests for hot-path vector capture producer."""

import array
import random
import struct
import threading
import unittest

import numpy

from vectorseam import (
    CaptureResult,
    DType,
    ProbabilitySampler,
    VectorCaptureProducer,
    capture_vector,
    encode_vector_message,
    get_vector_capture_producer,
)


def _packed_f32(values: list[float]) -> array.array:
    vector = array.array("f", values)
    if vector.itemsize != DType.F32.byte_size:
        raise AssertionError("test F32 array item size is invalid")
    if struct.pack("=f", 1.0) != struct.pack("<f", 1.0):
        vector.byteswap()
    return vector


def _unpack_u32(frame: bytes, offset: int) -> int:
    return struct.unpack_from("<I", frame, offset)[0]


def _clear_queue(producer: VectorCaptureProducer) -> None:
    producer.drain()


class ProbabilitySamplerTest(unittest.TestCase):
    """Verifies probability sampler behavior."""

    def test_sample_rate_one_samples_everything(self) -> None:
        sampler = ProbabilitySampler(1.0)

        self.assertTrue(sampler.should_sample("a"))
        self.assertTrue(sampler.should_sample("b"))

    def test_sample_rate_zero_samples_nothing(self) -> None:
        sampler = ProbabilitySampler(0.0)

        self.assertFalse(sampler.should_sample("a"))
        self.assertFalse(sampler.should_sample("b"))

    def test_invalid_sample_rates_raise_value_error(self) -> None:
        for sample_rate in (-0.1, 1.1):
            with self.subTest(sample_rate=sample_rate):
                with self.assertRaises(ValueError):
                    ProbabilitySampler(sample_rate)

    def test_seeded_rng_is_deterministic(self) -> None:
        first = ProbabilitySampler(0.5, rng=random.Random(1234))
        second = ProbabilitySampler(0.5, rng=random.Random(1234))

        first_decisions = [first.should_sample("raw") for _ in range(10)]
        second_decisions = [second.should_sample("raw") for _ in range(10)]

        self.assertEqual(first_decisions, second_decisions)


class VectorCaptureProducerTest(unittest.TestCase):
    """Verifies capture, queueing, draining, and occupancy behavior."""

    def test_sample_rate_zero_does_not_enqueue(self) -> None:
        producer = VectorCaptureProducer(
            sampler=ProbabilitySampler(0.0),
            max_queue_bytes=1024,
        )
        vector = _packed_f32([1.0, 2.0])

        result = producer.capture_vector("raw", vector, dimension=2)

        self.assertEqual(CaptureResult.NOT_SAMPLED, result)
        self.assertIsNone(producer.try_dequeue())
        self.assertEqual(0, producer.queued_bytes)
        self.assertEqual(0, producer.queued_frames)

    def test_sample_rate_one_enqueues_immutable_frame(self) -> None:
        producer = VectorCaptureProducer(
            sampler=ProbabilitySampler(1.0),
            max_queue_bytes=1024,
        )
        vector = _packed_f32([1.0, 2.0])
        expected_frame = encode_vector_message("raw", DType.F32, 2, vector)

        result = producer.capture_vector("raw", vector, dimension=2)

        self.assertEqual(CaptureResult.ENQUEUED, result)
        self.assertEqual(len(expected_frame), producer.queued_bytes)
        self.assertEqual(1, producer.queued_frames)
        queued_frame = producer.try_dequeue()
        self.assertIsInstance(queued_frame, bytes)
        self.assertEqual(expected_frame, queued_frame)

    def test_try_dequeue_reduces_queued_byte_count(self) -> None:
        vector = _packed_f32([1.0, 2.0])
        frame = encode_vector_message("raw", DType.F32, 2, vector)
        producer = VectorCaptureProducer(max_queue_bytes=len(frame) * 2)

        producer.capture_vector("raw", vector, dimension=2)
        self.assertEqual(len(frame), producer.queued_bytes)

        dequeued = producer.try_dequeue()

        self.assertEqual(frame, dequeued)
        self.assertEqual(0, producer.queued_bytes)
        self.assertEqual(0, producer.queued_frames)

    def test_drain_reduces_queued_byte_count(self) -> None:
        vector = _packed_f32([1.0, 2.0])
        raw_frame = encode_vector_message("raw", DType.F32, 2, vector)
        alt_frame = encode_vector_message("alt", DType.F32, 2, vector)
        producer = VectorCaptureProducer(
            max_queue_bytes=len(raw_frame) + len(alt_frame)
        )

        producer.capture_vector("raw", vector, dimension=2)
        producer.capture_vector("alt", vector, dimension=2)

        drained = producer.drain(max_bytes=len(raw_frame))

        self.assertEqual([raw_frame], drained)
        self.assertEqual(len(alt_frame), producer.queued_bytes)
        self.assertEqual(1, producer.queued_frames)
        self.assertEqual([alt_frame], producer.drain())
        self.assertEqual(0, producer.queued_bytes)
        self.assertEqual(0, producer.queued_frames)

    def test_drain_returns_empty_when_next_frame_exceeds_budget(self) -> None:
        vector = _packed_f32([1.0, 2.0])
        frame = encode_vector_message("raw", DType.F32, 2, vector)
        producer = VectorCaptureProducer(max_queue_bytes=len(frame))

        producer.capture_vector("raw", vector, dimension=2)

        self.assertEqual([], producer.drain(max_bytes=len(frame) - 1))
        self.assertEqual(len(frame), producer.queued_bytes)
        self.assertEqual(1, producer.queued_frames)

    def test_drain_rejects_invalid_max_bytes(self) -> None:
        producer = VectorCaptureProducer(max_queue_bytes=1024)

        with self.assertRaises(TypeError):
            producer.drain(max_bytes=1.5)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            producer.drain(max_bytes=True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            producer.drain(max_bytes=-1)

    def test_queue_full_drops_without_enqueuing_second_frame(self) -> None:
        vector = _packed_f32([1.0, 2.0])
        frame = encode_vector_message("raw", DType.F32, 2, vector)
        producer = VectorCaptureProducer(max_queue_bytes=len(frame))

        first = producer.capture_vector("raw", vector, dimension=2)
        second = producer.capture_vector("raw", vector, dimension=2)

        self.assertEqual(CaptureResult.ENQUEUED, first)
        self.assertEqual(CaptureResult.QUEUE_FULL, second)
        self.assertEqual(len(frame), producer.queued_bytes)
        self.assertEqual(1, producer.queued_frames)

    def test_single_frame_larger_than_queue_capacity_is_queue_full(self) -> None:
        vector = _packed_f32([1.0, 2.0])
        frame = encode_vector_message("raw", DType.F32, 2, vector)
        producer = VectorCaptureProducer(max_queue_bytes=len(frame) - 1)

        result = producer.capture_vector("raw", vector, dimension=2)

        self.assertEqual(CaptureResult.QUEUE_FULL, result)
        self.assertEqual(0, producer.queued_bytes)
        self.assertEqual(0, producer.queued_frames)
        self.assertIsNone(producer.try_dequeue())

    def test_dimension_is_encoded_from_required_argument(self) -> None:
        producer = VectorCaptureProducer(max_queue_bytes=1024)
        vector = _packed_f32([1.0, 2.0, 3.0])

        result = producer.capture_vector("raw", vector, dimension=3)
        frame = producer.try_dequeue()

        self.assertEqual(CaptureResult.ENQUEUED, result)
        if frame is None:
            self.fail("expected a queued frame")
        self.assertEqual(3, _unpack_u32(frame, 20))

    def test_invalid_vector_byte_length_raises_value_error(self) -> None:
        producer = VectorCaptureProducer(max_queue_bytes=1024)

        with self.assertRaisesRegex(ValueError, "vector_len"):
            producer.capture_vector("raw", b"\x00\x00\x00", dimension=1)

    def test_concurrent_capture_keeps_occupancy_consistent(self) -> None:
        vector = _packed_f32([1.0])
        frame = encode_vector_message("threaded", DType.F32, 1, vector)
        producer = VectorCaptureProducer(max_queue_bytes=1_000_000)
        thread_count = 8
        captures_per_thread = 50
        errors: list[BaseException] = []
        errors_lock = threading.Lock()

        def capture_many() -> None:
            try:
                for _ in range(captures_per_thread):
                    result = producer.capture_vector(
                        "threaded",
                        vector,
                        dimension=1,
                    )
                    self.assertEqual(CaptureResult.ENQUEUED, result)
            except BaseException as error:  # pylint: disable=broad-exception-caught
                with errors_lock:
                    errors.append(error)

        threads = [
            threading.Thread(target=capture_many)
            for _ in range(thread_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        if errors:
            raise errors[0]

        expected_captures = thread_count * captures_per_thread

        self.assertEqual(expected_captures, producer.queued_frames)
        self.assertEqual(expected_captures * len(frame), producer.queued_bytes)
        self.assertEqual(expected_captures, len(producer.drain()))
        self.assertEqual(0, producer.queued_bytes)

    def test_get_vector_capture_producer_returns_process_singleton(self) -> None:
        first = get_vector_capture_producer()
        second = get_vector_capture_producer()
        _clear_queue(first)
        vector = _packed_f32([1.0])
        frame = encode_vector_message("singleton", DType.F32, 1, vector)

        result = first.capture_vector("singleton", vector, dimension=1)

        self.assertIs(first, second)
        self.assertEqual(CaptureResult.ENQUEUED, result)
        self.assertEqual(1, second.queued_frames)
        self.assertEqual(frame, second.try_dequeue())
        self.assertEqual(0, first.queued_bytes)

    def test_capture_vector_uses_process_singleton(self) -> None:
        producer = get_vector_capture_producer()
        _clear_queue(producer)
        vector = _packed_f32([1.0, 2.0])
        frame = encode_vector_message("one_liner", DType.F32, 2, vector)

        result = capture_vector("one_liner", vector, dimension=2)

        self.assertEqual(CaptureResult.ENQUEUED, result)
        self.assertEqual(frame, producer.try_dequeue())
        self.assertEqual(0, producer.queued_bytes)

    def test_capture_vector_infers_numpy_metadata(self) -> None:
        producer = VectorCaptureProducer(max_queue_bytes=1024)
        vector = numpy.asarray([1.0, 2.0], dtype=numpy.dtype("<f4"))
        frame = encode_vector_message("numpy", DType.F32, 2, vector)

        result = capture_vector("numpy", vector, producer=producer)

        self.assertEqual(CaptureResult.ENQUEUED, result)
        self.assertEqual(frame, producer.try_dequeue())

    def test_capture_vector_accepts_non_contiguous_numpy_vector(self) -> None:
        producer = VectorCaptureProducer(max_queue_bytes=1024)
        source = numpy.asarray([1.0, 9.0, 2.0, 8.0], dtype=numpy.dtype("<f4"))
        vector = source[::2]
        expected = numpy.ascontiguousarray(vector)
        frame = encode_vector_message("numpy_slice", DType.F32, 2, expected)

        result = capture_vector("numpy_slice", vector, producer=producer)

        self.assertEqual(CaptureResult.ENQUEUED, result)
        self.assertEqual(frame, producer.try_dequeue())

    def test_capture_vector_rejects_numpy_dimension_mismatch(self) -> None:
        vector = numpy.asarray([1.0, 2.0], dtype=numpy.dtype("<f4"))

        with self.assertRaises(ValueError):
            capture_vector("bad_dimension", vector, dimension=3)

    def test_capture_vector_rejects_numpy_dtype_mismatch(self) -> None:
        vector = numpy.asarray([1.0, 2.0], dtype=numpy.dtype("<f8"))

        with self.assertRaises(ValueError):
            capture_vector("bad_dtype", vector, dtype=DType.F32)

    def test_capture_vector_rejects_multidimensional_numpy_array(self) -> None:
        vector = numpy.asarray([[1.0, 2.0]], dtype=numpy.dtype("<f4"))

        with self.assertRaises(ValueError):
            capture_vector("bad_shape", vector)

    def test_capture_vector_requires_dimension_for_non_numpy(self) -> None:
        vector = _packed_f32([1.0])

        with self.assertRaises(TypeError):
            capture_vector("raw_without_dimension", vector)

    def test_capture_vector_rejects_invalid_producer(self) -> None:
        vector = _packed_f32([1.0])

        with self.assertRaises(TypeError):
            capture_vector(
                "bad_producer",
                vector,
                dimension=1,
                producer=object(),  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
