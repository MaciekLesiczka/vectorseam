"""Tests for demo data preparation."""

import pathlib
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
