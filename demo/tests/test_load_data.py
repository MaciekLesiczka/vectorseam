"""Tests for demo data preparation."""

import argparse
import pathlib
import tempfile
import unittest
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

from demo.scripts import load_data


class LoadDataTest(unittest.TestCase):
    """Verifies query emission and required-input errors."""

    def test_write_queries_preserves_parquet_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            input_path = root / "queries.parquet"
            output_path = root / "data" / "queries.txt"
            table = pa.table({"text": ["query two", "query one"]})
            pq.write_table(table, input_path)

            row_count = load_data._write_queries(input_path, output_path)

            self.assertEqual(2, row_count)
            self.assertEqual(
                "query two\nquery one\n",
                output_path.read_text(encoding="utf-8"),
            )

    def test_require_file_names_missing_path(self) -> None:
        missing_path = pathlib.Path("/definitely/missing/docs.parquet")

        with self.assertRaisesRegex(
            load_data.DemoDataError,
            str(missing_path),
        ):
            load_data._require_file(missing_path)

    def test_main_loads_both_cohort_tables(self) -> None:
        args = argparse.Namespace(
            docs=pathlib.Path("superuser-docs.parquet"),
            queries=pathlib.Path("superuser-queries.parquet"),
            embeddings=pathlib.Path("superuser-embeddings.parquet"),
            queries_output=pathlib.Path("queries.txt"),
            reddit_docs=pathlib.Path("reddit-docs.parquet"),
            reddit_queries=pathlib.Path("reddit-queries.parquet"),
            reddit_embeddings=pathlib.Path("reddit-embeddings.parquet"),
            reddit_queries_output=pathlib.Path("queries_reddit.txt"),
            dsn="postgresql://demo",
            parallel_workers=4,
        )
        with (
            mock.patch.object(load_data, "_parse_args", return_value=args),
            mock.patch.object(load_data, "_require_file"),
            mock.patch.object(
                load_data, "_write_queries", return_value=2000
            ) as write_queries,
            mock.patch.object(
                load_data,
                "_load_database",
                return_value=(300_000, 1.0),
            ) as load_database,
        ):
            result = load_data.main()

        self.assertEqual(0, result)
        self.assertEqual(2, write_queries.call_count)
        self.assertEqual(
            [
                load_data.SUPERUSER_TABLE_NAME,
                load_data.REDDIT_TABLE_NAME,
            ],
            [
                call.kwargs["table_name"]
                for call in load_database.call_args_list
            ],
        )


if __name__ == "__main__":
    unittest.main()
