"""Tests for the demo replay driver."""

import contextlib
import io
import pathlib
import tempfile
import unittest
from unittest import mock
import urllib.error

from demo.driver import __main__ as driver


class DriverTest(unittest.TestCase):
    """Verifies replay input handling and CLI value validation."""

    def test_load_queries_preserves_file_order_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = pathlib.Path(temporary_directory) / "queries.txt"
            path.write_text("first query\nsecond query\n", encoding="utf-8")

            queries = driver._load_queries(path)

        self.assertEqual(["first query", "second query"], queries)

    def test_load_queries_rejects_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = pathlib.Path(temporary_directory) / "queries.txt"
            path.touch()

            with self.assertRaisesRegex(ValueError, "queries file is empty"):
                driver._load_queries(path)

    def test_search_url_appends_endpoint_to_base_url(self) -> None:
        self.assertEqual(
            "http://127.0.0.1:8000/search",
            driver._search_url("http://127.0.0.1:8000/"),
        )

    @mock.patch("demo.driver.__main__.urllib.request.urlopen")
    def test_send_query_returns_http_error_body(
        self, urlopen: mock.Mock
    ) -> None:
        urlopen.side_effect = urllib.error.HTTPError(
            url="http://127.0.0.1:8000/search",
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=io.BytesIO(b'{"detail":"database unavailable"}'),
        )

        status, error = driver._send_query(
            "http://127.0.0.1:8000/search",
            "query",
            1.0,
        )

        self.assertEqual(500, status)
        self.assertEqual(
            'HTTP 500 Internal Server Error: {"detail":"database unavailable"}',
            error,
        )

    @mock.patch("demo.driver.__main__.time.sleep")
    @mock.patch("demo.driver.__main__._send_query")
    def test_replay_logs_latest_error_in_each_interval(
        self, send_query: mock.Mock, unused_sleep: mock.Mock
    ) -> None:
        send_query.side_effect = [
            urllib.error.URLError("connection refused"),
            (503, "HTTP 503 Service Unavailable: retry later"),
            (200, None),
            (200, None),
            KeyboardInterrupt(),
        ]
        output = io.StringIO()

        with self.assertRaises(KeyboardInterrupt), contextlib.redirect_stdout(
            output
        ):
            driver.replay(
                queries=["query"],
                url="http://127.0.0.1:8000",
                qps=5.0,
                seed=7,
                log_every=2,
                timeout_seconds=1.0,
            )

        lines = output.getvalue().splitlines()
        self.assertIn("requests=2 errors=2", lines[0])
        self.assertIn(
            "last_error='HTTP 503 Service Unavailable: retry later'",
            lines[0],
        )
        self.assertIn("requests=4 errors=2", lines[1])
        self.assertIn("last_error=None", lines[1])


if __name__ == "__main__":
    unittest.main()
