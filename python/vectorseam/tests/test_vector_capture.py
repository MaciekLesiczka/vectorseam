"""Tests for hot-path vector capture producer."""

import array
import random
import struct
import threading
import unittest

from vectorseam import (
    CaptureResult,
    DType,
    ProbabilitySampler,
    VectorCaptureProducer,
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
    while producer.try_dequeue() is not None:
        pass


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

    def test_multiple_try_dequeue_calls_reduce_queued_byte_count(self) -> None:
        vector = _packed_f32([1.0, 2.0])
        raw_frame = encode_vector_message("raw", DType.F32, 2, vector)
        alt_frame = encode_vector_message("alt", DType.F32, 2, vector)
        producer = VectorCaptureProducer(
            max_queue_bytes=len(raw_frame) + len(alt_frame)
        )

        producer.capture_vector("raw", vector, dimension=2)
        producer.capture_vector("alt", vector, dimension=2)

        first = producer.try_dequeue()

        self.assertEqual(raw_frame, first)
        self.assertEqual(len(alt_frame), producer.queued_bytes)
        self.assertEqual(1, producer.queued_frames)
        self.assertEqual(alt_frame, producer.try_dequeue())
        self.assertEqual(0, producer.queued_bytes)

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
        dequeued_count = 0
        while producer.try_dequeue() is not None:
            dequeued_count += 1
        self.assertEqual(expected_captures, dequeued_count)
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


if __name__ == "__main__":
    unittest.main()
