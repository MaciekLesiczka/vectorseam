"""Best-effort Unix socket sender for captured vector frames.

``VectorSocketSender`` drains already-marshalled immutable ``bytes`` frames
from ``VectorCaptureProducer`` on a daemon background thread and sends them to
a collector over a Unix-domain ``SOCK_STREAM`` socket. Delivery is best-effort:
frames in a batch are dropped if sending fails.
"""

from __future__ import annotations

import socket
import threading
import time
from numbers import Real

from vectorseam.vector_capture import (
    VectorCaptureProducer,
    get_vector_capture_producer,
)


class VectorSocketSender:
    """Background Unix-domain stream sender for captured vector frames.

    The sender is safe to start and stop from multiple threads. Lifecycle state
    is protected by a ``threading.Lock``. Queue state remains owned by
    ``VectorCaptureProducer``, whose queue is protected by an OS-level mutex.
    """

    def __init__(
        self,
        *,
        socket_path: str,
        producer: VectorCaptureProducer | None = None,
        flush_interval_seconds: float = 0.01,
        max_batch_bytes: int = 128 * 1024,
        idle_sleep_seconds: float = 0.01,
        reconnect_interval_seconds: float = 1.0,
        send_timeout_seconds: float = 1.0,
    ) -> None:
        """Initializes a background sender.

        Args:
            socket_path: Unix-domain socket path used by the collector.
            producer: Capture producer to drain. Defaults to the process-wide
                producer.
            flush_interval_seconds: Debounce window used after the first frame
                is available.
            max_batch_bytes: Target maximum batch size. Frames are never split.
            idle_sleep_seconds: Sleep interval when no frame is available.
            reconnect_interval_seconds: Delay after a connection failure.
            send_timeout_seconds: Maximum time to spend sending one batch.

        Raises:
            TypeError: If an argument has an invalid type.
            ValueError: If a numeric argument is not positive or socket_path is
                empty.
        """
        if not isinstance(socket_path, str):
            raise TypeError("socket_path must be a string")
        if not socket_path:
            raise ValueError("socket_path must be non-empty")

        if producer is None:
            producer = get_vector_capture_producer()
        if not isinstance(producer, VectorCaptureProducer):
            raise TypeError("producer must be a VectorCaptureProducer")

        for name, value in (
            ("flush_interval_seconds", flush_interval_seconds),
            ("idle_sleep_seconds", idle_sleep_seconds),
            ("reconnect_interval_seconds", reconnect_interval_seconds),
            ("send_timeout_seconds", send_timeout_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a number")
            if value <= 0:
                raise ValueError(f"{name} must be greater than 0")

        if not isinstance(max_batch_bytes, int) or isinstance(
            max_batch_bytes, bool
        ):
            raise TypeError("max_batch_bytes must be an integer")
        if max_batch_bytes <= 0:
            raise ValueError("max_batch_bytes must be greater than 0")

        self._socket_path = socket_path
        self._producer = producer
        self._flush_interval_seconds = float(flush_interval_seconds)
        self._max_batch_bytes = max_batch_bytes
        self._idle_sleep_seconds = float(idle_sleep_seconds)
        self._reconnect_interval_seconds = float(reconnect_interval_seconds)
        self._send_timeout_seconds = float(send_timeout_seconds)

        self._stop_event = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._socket_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None

    def start(self) -> None:
        """Starts the daemon sender thread if it is not already running."""
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="vectorseam-vector-socket-sender",
                daemon=True,
            )
            self._thread.start()

    def stop(self, *, timeout: float | None = 1.0) -> None:
        """Signals the sender thread to stop, joins it, and closes the socket.

        Args:
            timeout: Optional join timeout in seconds. ``None`` waits without a
                timeout.

        Raises:
            TypeError: If timeout is not a number or None.
            ValueError: If timeout is negative.
        """
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, Real):
                raise TypeError("timeout must be a number or None")
            if timeout < 0:
                raise ValueError("timeout must be non-negative")

        self._stop_event.set()
        with self._lifecycle_lock:
            thread = self._thread

        if thread is not None:
            thread.join(timeout)

        self._close_socket()

    @property
    def is_running(self) -> bool:
        """Whether the sender thread is currently alive."""
        with self._lifecycle_lock:
            return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        """Runs the sender loop."""
        try:
            while not self._stop_event.is_set():
                sock = self._ensure_connected()
                if sock is None:
                    continue

                frames = self._drain_initial_frames()
                if not frames:
                    self._stop_event.wait(self._idle_sleep_seconds)
                    continue

                batch_bytes = sum(len(frame) for frame in frames)
                if self._stop_event.wait(self._flush_interval_seconds):
                    break

                remaining_budget = self._max_batch_bytes - batch_bytes
                if remaining_budget > 0:
                    more_frames = self._producer.drain(remaining_budget)
                    frames.extend(more_frames)

                self._send_batch(sock, frames)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        finally:
            self._close_socket()

    def _ensure_connected(self) -> socket.socket | None:
        """Returns a connected socket, or None after a failed attempt."""
        with self._socket_lock:
            if self._sock is not None:
                return self._sock

        sock: socket.socket | None = None
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(self._socket_path)
        except OSError:
            if sock is not None:
                sock.close()
            self._close_socket()
            self._stop_event.wait(self._reconnect_interval_seconds)
            return None

        with self._socket_lock:
            if self._stop_event.is_set():
                sock.close()
                return None
            self._sock = sock

        return sock

    def _drain_initial_frames(self) -> list[bytes]:
        """Drains an initial batch, allowing one oversized frame to progress."""
        frames = self._producer.drain(self._max_batch_bytes)
        if frames:
            return frames

        frame = self._producer.try_dequeue()
        if frame is None:
            return []
        return [frame]

    def _send_batch(
        self,
        sock: socket.socket,
        frames: list[bytes],
    ) -> None:
        """Sends one contiguous batch and drops it on send failure."""
        batch = b"".join(frames)
        if not self._send_all_bounded(sock, batch):
            self._close_socket(sock)

    def _send_all_bounded(self, sock: socket.socket, batch: bytes) -> bool:
        """Sends a batch with a total deadline.

        Any timeout or error is connection-fatal. A timeout may happen after a
        partial write, and reusing that stream would desynchronize framing.
        """
        deadline = time.monotonic() + self._send_timeout_seconds
        batch_view = memoryview(batch)
        sent_bytes = 0

        while sent_bytes < len(batch):
            if self._stop_event.is_set():
                return False

            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                return False

            try:
                sock.settimeout(remaining_seconds)
                sent_now = sock.send(batch_view[sent_bytes:])
            except OSError:
                return False
            if sent_now == 0:
                return False
            sent_bytes += sent_now

        return True

    def _close_socket(self, sock: socket.socket | None = None) -> None:
        """Closes a socket and clears it if it is current."""
        with self._socket_lock:
            if sock is None:
                sock = self._sock
                self._sock = None
            elif self._sock is sock:
                self._sock = None

        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


__all__ = [
    "VectorSocketSender",
]
