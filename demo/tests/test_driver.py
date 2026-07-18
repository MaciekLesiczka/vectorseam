"""Tests for the demo replay driver."""

import pathlib
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
