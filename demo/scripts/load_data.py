"""Load the SuperUser demo corpus into Postgres and emit query text."""

from __future__ import annotations

import argparse
import csv
import io
import os
import pathlib
import sys
import time
from typing import Any

import psycopg
from psycopg import sql
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
BENCHMARK_DATA = REPO_ROOT / "python" / "ann-recall-latency" / "data"
DEFAULT_DOCS_PATH = (
    BENCHMARK_DATA / "processed" / "stackexchange" / "docs.parquet"
)
DEFAULT_QUERIES_PATH = (
    BENCHMARK_DATA / "processed" / "stackexchange" / "queries.parquet"
)
DEFAULT_EMBEDDINGS_PATH = (
    BENCHMARK_DATA
    / "embeddings"
    / "stackexchange"
    / "BAAI_bge-small-en-v1.5__5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    / "docs.parquet"
)
DEFAULT_QUERY_OUTPUT = REPO_ROOT / "demo" / "data" / "queries.txt"
DEFAULT_DSN = "postgresql://postgres:password@127.0.0.1:5432/postgres"
TABLE_NAME = "docs_superuser"
INDEX_NAME = "docs_superuser_embedding_hnsw_idx"
EMBEDDING_DIMENSION = 384
COPY_BATCH_ROWS = 1000
PROGRESS_ROWS = 10_000


class DemoDataError(RuntimeError):
    """Raised when demo input data is missing or inconsistent."""


def _require_file(path: pathlib.Path) -> None:
    """Raises an error naming a required input path."""
    if not path.is_file():
        raise DemoDataError(f"missing required input: {path}")


def _write_queries(input_path: pathlib.Path, output_path: pathlib.Path) -> int:
    """Writes query text in Parquet file order, one query per line."""
    table = pq.read_table(input_path, columns=["text"])
    texts = table.column("text").combine_chunks().to_pylist()
    for row_number, text in enumerate(texts, start=1):
        if text is None:
            raise DemoDataError(
                f"query text is null at row {row_number} in {input_path}"
            )
        if "\n" in text or "\r" in text:
            raise DemoDataError(
                f"query text contains a newline at row {row_number} "
                f"in {input_path}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as file_obj:
        for text in texts:
            file_obj.write(text)
            file_obj.write("\n")
    temporary_path.replace(output_path)
    return len(texts)


def _load_document_bodies(path: pathlib.Path) -> dict[int, str]:
    """Loads raw document bodies keyed by document ID."""
    table = pq.read_table(path, columns=["doc_id", "text"])
    data = table.to_pydict()
    bodies: dict[int, str] = {}
    for row_number, (doc_id, body) in enumerate(
        zip(data["doc_id"], data["text"]), start=1
    ):
        if doc_id is None:
            raise DemoDataError(
                f"document id is null at row {row_number} in {path}"
            )
        if body is None:
            raise DemoDataError(
                f"document text is null at row {row_number} in {path}"
            )
        doc_id = int(doc_id)
        if doc_id in bodies:
            raise DemoDataError(f"duplicate document id {doc_id} in {path}")
        bodies[doc_id] = str(body)
    return bodies


def _format_vector(values: list[float]) -> str:
    """Formats vector values for pgvector input."""
    return "[" + ",".join(f"{float(value):.9g}" for value in values) + "]"


def _copy_csv_rows(rows: list[list[Any]]) -> str:
    """Serializes a COPY batch as CSV."""
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    return output.getvalue()


def _validate_embedding_schema(
    parquet_file: pq.ParquetFile, path: pathlib.Path
) -> None:
    """Validates the required embedding column before database mutation."""
    schema = parquet_file.schema_arrow
    for field_name in ("doc_id", "embedding"):
        if field_name not in schema.names:
            raise DemoDataError(f"missing {field_name!r} column in {path}")
    embedding_type = schema.field("embedding").type
    if (
        not pa.types.is_fixed_size_list(embedding_type)
        or embedding_type.list_size != EMBEDDING_DIMENSION
        or not pa.types.is_float32(embedding_type.value_type)
    ):
        raise DemoDataError(
            "embedding column must be fixed_size_list<float32>"
            f"[{EMBEDDING_DIMENSION}], got {embedding_type}"
        )


def _create_table(connection: psycopg.Connection[Any]) -> None:
    """Drops and recreates the idempotent demo table."""
    with connection.cursor() as cursor:
        cursor.execute(
            sql.SQL("DROP TABLE IF EXISTS {} CASCADE;").format(
                sql.Identifier(TABLE_NAME)
            )
        )
        cursor.execute(
            sql.SQL(
                """
                CREATE TABLE {} (
                    doc_id bigint PRIMARY KEY,
                    body text NOT NULL,
                    embedding vector({}) NOT NULL
                );
                """
            ).format(
                sql.Identifier(TABLE_NAME),
                sql.Literal(EMBEDDING_DIMENSION),
            )
        )
    connection.commit()


def _copy_documents(
    connection: psycopg.Connection[Any],
    embeddings_path: pathlib.Path,
    bodies: dict[int, str],
    docs_path: pathlib.Path,
) -> int:
    """Joins embeddings to raw bodies by ID and streams them through COPY."""
    parquet_file = pq.ParquetFile(embeddings_path)
    _validate_embedding_schema(parquet_file, embeddings_path)
    expected_rows = len(bodies)
    copied_rows = 0
    copy_statement = sql.SQL(
        "COPY {} (doc_id, body, embedding) "
        "FROM STDIN WITH (FORMAT csv, NULL '')"
    ).format(sql.Identifier(TABLE_NAME))

    with connection.cursor() as cursor:
        with cursor.copy(copy_statement) as copy:
            for batch in parquet_file.iter_batches(
                batch_size=COPY_BATCH_ROWS,
                columns=["doc_id", "embedding"],
            ):
                data = pa.Table.from_batches([batch]).to_pydict()
                rows: list[list[Any]] = []
                for doc_id, embedding in zip(data["doc_id"], data["embedding"]):
                    if doc_id is None:
                        raise DemoDataError(
                            f"null document id in {embeddings_path}"
                        )
                    doc_id = int(doc_id)
                    try:
                        body = bodies.pop(doc_id)
                    except KeyError as error:
                        raise DemoDataError(
                            f"embedding document id {doc_id} is missing or "
                            f"duplicated relative to {docs_path}"
                        ) from error
                    rows.append([doc_id, body, _format_vector(embedding)])
                copy.write(_copy_csv_rows(rows))
                copied_rows += len(rows)
                if copied_rows % PROGRESS_ROWS == 0:
                    print(f"copied {copied_rows:,} documents", flush=True)

    if bodies:
        missing_ids = sorted(bodies)[:5]
        raise DemoDataError(
            f"{len(bodies):,} raw documents have no embedding in "
            f"{embeddings_path}; first IDs: {missing_ids}"
        )
    if copied_rows != expected_rows:
        raise DemoDataError(
            f"joined row count {copied_rows:,} does not match raw document "
            f"count {expected_rows:,}"
        )
    connection.commit()
    return copied_rows


def _build_index(
    connection: psycopg.Connection[Any],
    parallel_workers: int,
) -> float:
    """Builds and analyzes the benchmark-compatible HNSW index."""
    started_at = time.monotonic()
    with connection.cursor() as cursor:
        cursor.execute("SET maintenance_work_mem = '1GB';")
        cursor.execute(
            sql.SQL("SET max_parallel_maintenance_workers = {};").format(
                sql.Literal(parallel_workers)
            )
        )
        cursor.execute(
            sql.SQL(
                """
                CREATE INDEX {} ON {}
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
                """
            ).format(
                sql.Identifier(INDEX_NAME),
                sql.Identifier(TABLE_NAME),
            )
        )
        cursor.execute(
            sql.SQL("ANALYZE {};").format(sql.Identifier(TABLE_NAME))
        )
    connection.commit()
    return time.monotonic() - started_at


def _load_database(
    *,
    dsn: str,
    docs_path: pathlib.Path,
    embeddings_path: pathlib.Path,
    parallel_workers: int,
) -> tuple[int, float]:
    """Loads the joined corpus and returns row count and index build time."""
    bodies = _load_document_bodies(docs_path)
    with psycopg.connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        connection.commit()
        _create_table(connection)
        try:
            row_count = _copy_documents(
                connection,
                embeddings_path,
                bodies,
                docs_path,
            )
        except Exception:
            connection.rollback()
            raise
        index_seconds = _build_index(connection, parallel_workers)
    return row_count, index_seconds


def _parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=pathlib.Path, default=DEFAULT_DOCS_PATH)
    parser.add_argument(
        "--queries", type=pathlib.Path, default=DEFAULT_QUERIES_PATH
    )
    parser.add_argument(
        "--embeddings",
        type=pathlib.Path,
        default=DEFAULT_EMBEDDINGS_PATH,
    )
    parser.add_argument(
        "--queries-output",
        type=pathlib.Path,
        default=DEFAULT_QUERY_OUTPUT,
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("DATABASE_URL", DEFAULT_DSN),
        help="Postgres connection string.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=4,
        help="Postgres parallel index-build workers. Defaults to 4.",
    )
    args = parser.parse_args()
    if args.parallel_workers < 0:
        parser.error("--parallel-workers must be non-negative")
    if not args.dsn:
        parser.error("--dsn must be non-empty")
    return args


def main() -> int:
    """Writes query text, loads documents, and builds the HNSW index."""
    args = _parse_args()
    try:
        for path in (args.docs, args.queries, args.embeddings):
            _require_file(path)
        query_count = _write_queries(args.queries, args.queries_output)
        print(
            f"wrote {query_count:,} queries to {args.queries_output}",
            flush=True,
        )
        row_count, index_seconds = _load_database(
            dsn=args.dsn,
            docs_path=args.docs,
            embeddings_path=args.embeddings,
            parallel_workers=args.parallel_workers,
        )
    except (DemoDataError, OSError, psycopg.Error) as error:
        print(f"load_data.py: error: {error}", file=sys.stderr)
        return 1

    print(
        f"loaded {row_count:,} rows into {TABLE_NAME}; "
        f"HNSW index built in {index_seconds:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
