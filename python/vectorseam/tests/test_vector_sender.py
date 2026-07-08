"""Tests for background Unix socket vector sender."""

import array
import os
import socket
import struct
import tempfile
import threading
import time
import unittest
from unittest import mock

from vectorseam import (
    CaptureResult,
    DType,
    VectorCaptureProducer,
    VectorSocketSender,
    encode_vector_frame,
)
from vectorseam import vector_sender


def _packed_f32(values: list[float]) -> array.array:
    vector = array.array("f", values)
    if vector.itemsize != DType.F32.byte_size:
        raise AssertionError("test F32 array item size is invalid")
    if struct.pack("=f", 1.0) != struct.pack("<f", 1.0):
        vector.byteswap()
    return vector


def _wait_until(predicate, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return predicate()


class _UnixStreamServer:
    """Small Unix socket server used by sender tests."""

    def __init__(
        self,
        socket_path: str,
        *,
        expected_bytes: int | None = None,
        close_after_accept: bool = False,
    ) -> None:
        self.received = bytearray()
        self.accepted = threading.Event()
        self.done = threading.Event()
        self.errors: list[BaseException] = []
        self._socket_path = socket_path
        self._expected_bytes = expected_bytes
        self._close_after_accept = close_after_accept
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(socket_path)
        self._server.listen(1)
        self._server.settimeout(0.05)
        self._conn: socket.socket | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self.done.set()
        if self._conn is not None:
            try:
                self._conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._conn.close()
        self._server.close()
        self._thread.join(1.0)
        try:
            os.unlink(self._socket_path)
        except FileNotFoundError:
            pass

    def _run(self) -> None:
        try:
            self._serve()
        except BaseException as error:  # pylint: disable=broad-exception-caught
            self.errors.append(error)
            self.done.set()

    def _serve(self) -> None:
        while not self.done.is_set():
            try:
                conn, _ = self._server.accept()
                break
            except TimeoutError:
                continue
            except OSError:
                return
        else:
            return

        self._conn = conn
        self.accepted.set()
        if self._close_after_accept:
            conn.close()
            self.done.set()
            return

        conn.settimeout(0.01)
        while not self.done.is_set():
            if (
                self._expected_bytes is not None
                and len(self.received) >= self._expected_bytes
            ):
                self.done.set()
                break

            try:
                chunk = conn.recv(4096)
            except TimeoutError:
                continue
            except OSError:
                break
            if not chunk:
                break
            self.received.extend(chunk)

        try:
            conn.close()
        except OSError:
            pass


@unittest.skipUnless(hasattr(socket, "AF_UNIX"), "requires Unix sockets")
class VectorSocketSenderTest(unittest.TestCase):
    """Verifies sender validation, lifecycle, batching, and failures."""

    def test_constructor_validation(self) -> None:
        producer = VectorCaptureProducer()

        with self.assertRaises(ValueError):
            VectorSocketSender(socket_path="", producer=producer)
        with self.assertRaises(TypeError):
            VectorSocketSender(
                socket_path=123,  # type: ignore[arg-type]
                producer=producer,
            )

        for timing_name in (
            "flush_interval_seconds",
            "idle_sleep_seconds",
            "reconnect_interval_seconds",
            "send_timeout_seconds",
        ):
            with self.subTest(timing_name=timing_name):
                with self.assertRaises(ValueError):
                    VectorSocketSender(
                        socket_path="/tmp/vectorseam.sock",
                        producer=producer,
                        **{timing_name: 0.0},
                    )
                with self.assertRaises(TypeError):
                    VectorSocketSender(
                        socket_path="/tmp/vectorseam.sock",
                        producer=producer,
                        **{timing_name: True},
                    )

        with self.assertRaises(ValueError):
            VectorSocketSender(
                socket_path="/tmp/vectorseam.sock",
                producer=producer,
                max_batch_bytes=0,
            )
        with self.assertRaises(TypeError):
            VectorSocketSender(
                socket_path="/tmp/vectorseam.sock",
                producer=producer,
                max_batch_bytes=1.5,  # type: ignore[arg-type]
            )
        with self.assertRaises(TypeError):
            VectorSocketSender(
                socket_path="/tmp/vectorseam.sock",
                producer=object(),  # type: ignore[arg-type]
            )

    def test_lifecycle_start_is_idempotent_and_stop_is_safe(self) -> None:
        producer = VectorCaptureProducer()
        sender = VectorSocketSender(
            socket_path="/tmp/vectorseam-missing.sock",
            producer=producer,
            flush_interval_seconds=0.001,
            idle_sleep_seconds=0.001,
            reconnect_interval_seconds=0.01,
        )

        sender.stop()
        sender.start()
        self.assertTrue(_wait_until(lambda: sender.is_running))
        first_thread = sender._thread  # pylint: disable=protected-access

        sender.start()

        self.assertIs(
            first_thread,
            sender._thread,  # pylint: disable=protected-access
        )
        sender.stop()
        self.assertFalse(sender.is_running)

    def test_successful_send(self) -> None:
        with _TempSocketPath() as socket_path:
            vector = _packed_f32([1.0, 2.0])
            expected = encode_vector_frame("raw", DType.F32, 2, vector)
            producer = VectorCaptureProducer(max_queue_bytes=len(expected))
            self.assertEqual(
                CaptureResult.ENQUEUED,
                producer.capture_vector("raw", vector, dimension=2),
            )
            server = _UnixStreamServer(
                socket_path,
                expected_bytes=len(expected),
            )
            server.start()
            sender = _new_fast_sender(socket_path, producer)

            try:
                sender.start()
                self.assertTrue(
                    _wait_until(lambda: bytes(server.received) == expected)
                )
            finally:
                sender.stop()
                server.stop()

            self.assertEqual([], server.errors)
            self.assertEqual(0, producer.queued_frames)
            self.assertEqual(0, producer.queued_bytes)

    def test_micro_batching_sends_concatenated_stream(self) -> None:
        with _TempSocketPath() as socket_path:
            vector = _packed_f32([1.0, 2.0])
            first = encode_vector_frame("first", DType.F32, 2, vector)
            second = encode_vector_frame("second", DType.F32, 2, vector)
            expected = first + second
            producer = VectorCaptureProducer(max_queue_bytes=len(expected))
            self.assertEqual(
                CaptureResult.ENQUEUED,
                producer.capture_vector("first", vector, dimension=2),
            )
            server = _UnixStreamServer(
                socket_path,
                expected_bytes=len(expected),
            )
            server.start()
            sender = _new_fast_sender(
                socket_path,
                producer,
                flush_interval_seconds=0.02,
            )

            try:
                sender.start()
                self.assertEqual(
                    CaptureResult.ENQUEUED,
                    producer.capture_vector("second", vector, dimension=2),
                )
                self.assertTrue(
                    _wait_until(lambda: bytes(server.received) == expected)
                )
            finally:
                sender.stop()
                server.stop()

            self.assertEqual([], server.errors)

    def test_empty_queue_does_not_crash(self) -> None:
        with _TempSocketPath() as socket_path:
            producer = VectorCaptureProducer()
            server = _UnixStreamServer(socket_path)
            server.start()
            sender = _new_fast_sender(socket_path, producer)

            try:
                sender.start()
                self.assertTrue(_wait_until(lambda: sender.is_running))
                self.assertTrue(_wait_until(lambda: server.accepted.is_set()))
            finally:
                sender.stop()
                server.stop()

            self.assertEqual([], server.errors)

    def test_connection_failure_does_not_crash(self) -> None:
        with _TempSocketPath(create=False) as socket_path:
            sender = _new_fast_sender(socket_path, VectorCaptureProducer())

            try:
                sender.start()
                self.assertTrue(_wait_until(lambda: sender.is_running))
                time.sleep(0.02)
            finally:
                sender.stop()

            self.assertFalse(sender.is_running)

    def test_send_failure_drops_batch(self) -> None:
        vector = _packed_f32([1.0, 2.0])
        frame = encode_vector_frame("raw", DType.F32, 2, vector)
        producer = VectorCaptureProducer(max_queue_bytes=len(frame))
        self.assertEqual(
            CaptureResult.ENQUEUED,
            producer.capture_vector("raw", vector, dimension=2),
        )
        fake_socket = _FailingSendSocket()
        sender = _new_fast_sender("/tmp/vectorseam.sock", producer)

        with mock.patch.object(
            vector_sender.socket,
            "socket",
            return_value=fake_socket,
        ):
            try:
                sender.start()
                self.assertTrue(
                    _wait_until(lambda: fake_socket.send_attempted.is_set())
                )
                self.assertTrue(
                    _wait_until(lambda: producer.queued_frames == 0)
                )
                self.assertTrue(sender.is_running)
            finally:
                sender.stop()

        self.assertTrue(fake_socket.closed)

    def test_partial_send_timeout_closes_connection(self) -> None:
        vector = _packed_f32([1.0, 2.0])
        frame = encode_vector_frame("raw", DType.F32, 2, vector)
        producer = VectorCaptureProducer(max_queue_bytes=len(frame))
        self.assertEqual(
            CaptureResult.ENQUEUED,
            producer.capture_vector("raw", vector, dimension=2),
        )
        fake_socket = _PartialTimeoutSocket()
        sender = _new_fast_sender("/tmp/vectorseam.sock", producer)

        with mock.patch.object(
            vector_sender.socket,
            "socket",
            return_value=fake_socket,
        ):
            try:
                sender.start()
                self.assertTrue(
                    _wait_until(lambda: fake_socket.send_attempts == 2)
                )
                self.assertTrue(
                    _wait_until(lambda: producer.queued_frames == 0)
                )
            finally:
                sender.stop()

        self.assertTrue(fake_socket.closed)
        self.assertGreater(fake_socket.sent_bytes, 0)
        self.assertLess(fake_socket.sent_bytes, len(frame))

    def test_socket_connected_after_stop_is_closed_not_stored(self) -> None:
        fake_socket = _FakeConnectedSocket()
        sender = _new_fast_sender(
            "/tmp/vectorseam.sock",
            VectorCaptureProducer(),
        )
        sender._stop_event.set()  # pylint: disable=protected-access

        with mock.patch.object(
            vector_sender.socket,
            "socket",
            return_value=fake_socket,
        ):
            self.assertIsNone(
                sender._ensure_connected()  # pylint: disable=protected-access
            )

        self.assertTrue(fake_socket.closed)
        self.assertIsNone(sender._sock)  # pylint: disable=protected-access

    def test_unhandled_worker_exception_closes_socket(self) -> None:
        fake_socket = _FakeConnectedSocket()
        sender = _new_fast_sender(
            "/tmp/vectorseam.sock",
            _RaisingDrainProducer(),
        )

        with mock.patch.object(
            vector_sender.socket,
            "socket",
            return_value=fake_socket,
        ):
            sender.start()
            self.assertTrue(_wait_until(lambda: fake_socket.closed))

        self.assertFalse(sender.is_running)
        self.assertIsNone(sender._sock)  # pylint: disable=protected-access

    def test_oversized_frame_is_sent_alone(self) -> None:
        with _TempSocketPath() as socket_path:
            vector = _packed_f32([1.0, 2.0])
            frame = encode_vector_frame("raw", DType.F32, 2, vector)
            producer = VectorCaptureProducer(max_queue_bytes=len(frame))
            self.assertEqual(
                CaptureResult.ENQUEUED,
                producer.capture_vector("raw", vector, dimension=2),
            )
            server = _UnixStreamServer(
                socket_path,
                expected_bytes=len(frame),
            )
            server.start()
            sender = _new_fast_sender(
                socket_path,
                producer,
                max_batch_bytes=len(frame) - 1,
            )

            try:
                sender.start()
                self.assertTrue(
                    _wait_until(lambda: bytes(server.received) == frame)
                )
            finally:
                sender.stop()
                server.stop()

            self.assertEqual([], server.errors)
            self.assertEqual(0, producer.queued_frames)
            self.assertEqual(0, producer.queued_bytes)


class _TempSocketPath:
    """Context manager for short temporary Unix socket paths."""

    def __init__(self, *, create: bool = True) -> None:
        self.path = ""
        self._create = create
        self._directory: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> str:
        directory = tempfile.TemporaryDirectory()
        self._directory = directory
        self.path = os.path.join(directory.name, "s.sock")
        if self._create:
            return self.path
        return os.path.join(directory.name, "missing.sock")

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.path:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass
        if self._directory is not None:
            self._directory.cleanup()


class _FailingSendSocket:
    """Socket test double that connects but fails every send."""

    def __init__(self) -> None:
        self.closed = False
        self.send_attempted = threading.Event()

    def connect(self, socket_path: str) -> None:
        pass

    def settimeout(self, timeout: float) -> None:
        pass

    def send(self, batch: memoryview) -> int:
        self.send_attempted.set()
        raise OSError("send failed")

    def close(self) -> None:
        self.closed = True


class _PartialTimeoutSocket:
    """Socket test double that writes once and then times out."""

    def __init__(self) -> None:
        self.closed = False
        self.send_attempts = 0
        self.sent_bytes = 0

    def connect(self, socket_path: str) -> None:
        pass

    def settimeout(self, timeout: float) -> None:
        pass

    def send(self, batch: memoryview) -> int:
        self.send_attempts += 1
        if self.send_attempts == 1:
            self.sent_bytes = min(1, len(batch))
            return self.sent_bytes
        raise TimeoutError("send timed out")

    def close(self) -> None:
        self.closed = True


class _FakeConnectedSocket:
    """Socket test double that connects successfully."""

    def __init__(self) -> None:
        self.closed = False

    def connect(self, socket_path: str) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _RaisingDrainProducer(VectorCaptureProducer):
    """Producer that raises after connection for worker exception tests."""

    def drain(self, max_bytes: int | None = None) -> list[bytes]:
        raise RuntimeError("drain failed")


def _new_fast_sender(
    socket_path: str,
    producer: VectorCaptureProducer,
    *,
    flush_interval_seconds: float = 0.001,
    max_batch_bytes: int = 128 * 1024,
) -> VectorSocketSender:
    return VectorSocketSender(
        socket_path=socket_path,
        producer=producer,
        flush_interval_seconds=flush_interval_seconds,
        max_batch_bytes=max_batch_bytes,
        idle_sleep_seconds=0.001,
        reconnect_interval_seconds=0.01,
        send_timeout_seconds=0.01,
    )


if __name__ == "__main__":
    unittest.main()
