"""Tests for the effective-recommendation HTTP client."""

import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from vectorseam import RecommendationClient


class _RecommendationServer:
    """Small HTTP server returning scripted recommendation responses."""

    def __init__(self, status: int = 200, body: bytes = b"60") -> None:
        self.status = status
        self.body = body
        self.paths: list[str] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                server.paths.append(self.path)
                self.send_response(server.status)
                self.send_header("Content-Length", str(len(server.body)))
                self.end_headers()
                self.wfile.write(server.body)

            def log_message(self, *args: object) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)


class RecommendationClientTest(unittest.TestCase):
    def _client(self, server: _RecommendationServer, **kwargs) -> RecommendationClient:
        options = {"default_ef_search": 100, "timeout_seconds": 5.0}
        options.update(kwargs)
        return RecommendationClient(port=server.port, **options)

    def test_reads_recommendation_and_caches_it(self) -> None:
        server = _RecommendationServer(body=b"60")
        self.addCleanup(server.close)
        client = self._client(server, ttl_seconds=300.0)

        self.assertEqual(client.ef_search("superuser"), 60)
        self.assertEqual(client.ef_search("superuser"), 60)
        self.assertEqual(server.paths, ["/v1/ef-search/superuser"])

    def test_expired_entry_is_refetched(self) -> None:
        server = _RecommendationServer(body=b"60")
        self.addCleanup(server.close)
        client = self._client(server, ttl_seconds=0.0)

        self.assertEqual(client.ef_search("superuser"), 60)
        self.assertEqual(client.ef_search("superuser"), 60)
        self.assertEqual(len(server.paths), 2)

    def test_hierarchical_cohort_path_is_preserved(self) -> None:
        server = _RecommendationServer(body=b"60")
        self.addCleanup(server.close)

        self.assertEqual(self._client(server).ef_search("tenant/a"), 60)
        self.assertEqual(server.paths, ["/v1/ef-search/tenant/a"])

    def test_missing_recommendation_falls_back_to_default(self) -> None:
        server = _RecommendationServer(status=404, body=b"")
        self.addCleanup(server.close)

        self.assertEqual(self._client(server).ef_search("superuser"), 100)

    def test_bad_request_raises_and_does_not_consume_cohort_capacity(self) -> None:
        server = _RecommendationServer(status=400, body=b"")
        self.addCleanup(server.close)
        client = self._client(server, max_cohorts=1)

        with self.assertRaisesRegex(ValueError, "invalid cohort"):
            client.ef_search("not valid")
        server.status = 200
        server.body = b"60"

        self.assertEqual(client.ef_search("valid"), 60)
        self.assertEqual(
            server.paths,
            ["/v1/ef-search/not%20valid", "/v1/ef-search/valid"],
        )

    def test_unreachable_collector_falls_back_to_default(self) -> None:
        server = _RecommendationServer()
        port = server.port
        server.close()
        client = RecommendationClient(
            port=port, default_ef_search=100, timeout_seconds=5.0
        )

        self.assertEqual(client.ef_search("superuser"), 100)

    def test_out_of_range_and_malformed_bodies_are_rejected(self) -> None:
        for body in (b"0", b"1001", b"sixty", b""):
            with self.subTest(body=body):
                server = _RecommendationServer(body=body)
                self.addCleanup(server.close)

                self.assertEqual(self._client(server).ef_search("superuser"), 100)

    def test_last_known_value_survives_a_later_failure(self) -> None:
        server = _RecommendationServer(body=b"60")
        self.addCleanup(server.close)
        client = self._client(server, ttl_seconds=0.0)

        self.assertEqual(client.ef_search("superuser"), 60)
        server.status = 503
        server.body = b""

        self.assertEqual(client.ef_search("superuser"), 60)

    def test_cohorts_beyond_the_cap_get_the_default_without_a_lookup(
        self,
    ) -> None:
        server = _RecommendationServer(body=b"60")
        self.addCleanup(server.close)
        client = self._client(server, ttl_seconds=0.0, max_cohorts=2)

        self.assertEqual(client.ef_search("a"), 60)
        self.assertEqual(client.ef_search("b"), 60)
        self.assertEqual(client.ef_search("c"), 100)
        self.assertEqual(client.ef_search("c"), 100)
        # Cohorts admitted before the cap keep refreshing.
        self.assertEqual(client.ef_search("a"), 60)
        self.assertEqual(
            server.paths,
            [
                "/v1/ef-search/a",
                "/v1/ef-search/b",
                "/v1/ef-search/a",
            ],
        )

    def test_invalid_options_are_rejected(self) -> None:
        for kwargs in (
            {"host": ""},
            {"port": 0},
            {"default_ef_search": 0},
            {"default_ef_search": 1001},
            {"ttl_seconds": -1.0},
            {"timeout_seconds": 0.0},
            {"max_cohorts": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    RecommendationClient(**kwargs)


if __name__ == "__main__":
    unittest.main()
