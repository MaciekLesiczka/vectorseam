"""Load embedded benchmark documents into Postgres and build HNSW indexes."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import pathlib
import sys
import time
from typing import Any

import psycopg
from psycopg import sql
import pyarrow as pa
import pyarrow.parquet as pq

from common import (
    BenchmarkError,
    cache_matches,
    DEFAULT_CONFIG,
    DEFAULT_DATA_DIR,
    embedding_paths,
    file_fingerprint,
    format_vector,
    index_name,
    load_yaml,
    table_name,
)


DEFAULT_DSN = "postgresql://postgres:password@localhost:5432/postgres"
MANIFEST_TABLE = "ann_recall_latency_manifests"
COPY_BATCH_ROWS = 1000


def _quote_array(values: list[str] | None) -> str:
    if not values:
        return "{}"
    escaped = [
        '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
        for value in values
    ]
    return "{" + ",".join(escaped) + "}"


def _none_to_empty(value: Any) -> Any:
    return "" if value is None else value


def _copy_csv_rows(rows: list[list[Any]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    return output.getvalue()


def _postgres_versions(conn: psycopg.Connection[Any]) -> dict[str, str]:
    with conn.cursor() as cur:
        cur.execute("SELECT version();")
        postgres_version = str(cur.fetchone()[0])
        cur.execute(
            """
            SELECT extversion
            FROM pg_extension
            WHERE extname = 'vector';
            """
        )
        row = cur.fetchone()
    if row is None:
        raise BenchmarkError("pgvector extension is not installed")
    return {
        "postgres_version": postgres_version,
        "pgvector_version": str(row[0]),
    }


def _ensure_extension(conn: psycopg.Connection[Any]) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()


def _ensure_manifest_table(conn: psycopg.Connection[Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {} (
                    dataset text PRIMARY KEY,
                    manifest jsonb NOT NULL,
                    updated_at timestamptz NOT NULL DEFAULT now()
                );
                """
            ).format(sql.Identifier(MANIFEST_TABLE))
        )
    conn.commit()


def _load_db_manifest(
    conn: psycopg.Connection[Any], dataset: str
) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT manifest FROM {} WHERE dataset = %s;").format(
                sql.Identifier(MANIFEST_TABLE)
            ),
            (dataset,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    manifest = row[0]
    if isinstance(manifest, str):
        return json.loads(manifest)
    return dict(manifest)


def _write_db_manifest(
    conn: psycopg.Connection[Any], dataset: str, manifest: dict[str, Any]
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                INSERT INTO {} (dataset, manifest, updated_at)
                VALUES (%s, %s, now())
                ON CONFLICT (dataset) DO UPDATE
                SET manifest = EXCLUDED.manifest,
                    updated_at = EXCLUDED.updated_at;
                """
            ).format(sql.Identifier(MANIFEST_TABLE)),
            (dataset, json.dumps(manifest)),
        )
    conn.commit()


def _table_exists(conn: psycopg.Connection[Any], table_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s);", (table_name,))
        return cur.fetchone()[0] is not None


def _index_exists(conn: psycopg.Connection[Any], index_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s);", (index_name,))
        return cur.fetchone()[0] is not None


def _cache_expectations(
    dataset: str,
    config: dict[str, Any],
    data_dir: pathlib.Path,
    versions: dict[str, str],
) -> dict[str, Any]:
    paths = embedding_paths(data_dir, dataset, config)
    hnsw_config = config["benchmark"]["hnsw"]
    model_config = config["model"]
    return {
        "dataset": dataset,
        "table_name": table_name(dataset),
        "index_name": index_name(dataset),
        "embedding_dim": int(model_config["embedding_dim"]),
        "hnsw": {
            "m": int(hnsw_config["m"]),
            "ef_construction": int(hnsw_config["ef_construction"]),
            "ops": "vector_cosine_ops",
        },
        "input_docs": file_fingerprint(paths["docs"]),
        "postgres": versions,
    }


def _create_table(
    conn: psycopg.Connection[Any], dataset: str, dimension: int
) -> None:
    table = sql.Identifier(table_name(dataset))
    with conn.cursor() as cur:
        cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE;").format(table))
        if dataset == "stackexchange":
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE {} (
                        doc_id bigint PRIMARY KEY,
                        embedding vector({}),
                        tags text[] NOT NULL,
                        score integer,
                        creation_date timestamptz,
                        answer_count integer
                    );
                    """
                ).format(table, sql.Literal(dimension))
            )
        elif dataset == "reddit":
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE {} (
                        doc_id bigint PRIMARY KEY,
                        embedding vector({}),
                        community text,
                        score integer,
                        datetime timestamptz,
                        data_type text
                    );
                    """
                ).format(table, sql.Literal(dimension))
            )
        else:
            raise BenchmarkError(f"Unsupported dataset: {dataset}")
    conn.commit()


def _batch_to_stackexchange_rows(batch: pa.Table) -> list[list[Any]]:
    data = batch.to_pydict()
    rows = []
    for index, doc_id in enumerate(data["doc_id"]):
        rows.append(
            [
                doc_id,
                format_vector(data["embedding"][index]),
                _quote_array(data["tags"][index]),
                _none_to_empty(data["score"][index]),
                _none_to_empty(data["creation_date"][index]),
                _none_to_empty(data["answer_count"][index]),
            ]
        )
    return rows


def _batch_to_reddit_rows(batch: pa.Table) -> list[list[Any]]:
    data = batch.to_pydict()
    rows = []
    for index, doc_id in enumerate(data["doc_id"]):
        rows.append(
            [
                doc_id,
                format_vector(data["embedding"][index]),
                _none_to_empty(data["community"][index]),
                _none_to_empty(data["score"][index]),
                _none_to_empty(data["datetime"][index]),
                _none_to_empty(data["data_type"][index]),
            ]
        )
    return rows


def _copy_statement(dataset: str) -> sql.Composed:
    table = sql.Identifier(table_name(dataset))
    if dataset == "stackexchange":
        columns = [
            "doc_id",
            "embedding",
            "tags",
            "score",
            "creation_date",
            "answer_count",
        ]
    elif dataset == "reddit":
        columns = [
            "doc_id",
            "embedding",
            "community",
            "score",
            "datetime",
            "data_type",
        ]
    else:
        raise BenchmarkError(f"Unsupported dataset: {dataset}")
    return sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT csv, NULL '')").format(
        table, sql.SQL(", ").join(sql.Identifier(column) for column in columns)
    )


def _load_rows(
    conn: psycopg.Connection[Any],
    dataset: str,
    docs_path: pathlib.Path,
) -> int:
    parquet_file = pq.ParquetFile(docs_path)
    total_rows = 0
    started_at = time.monotonic()
    with conn.cursor() as cur:
        with cur.copy(_copy_statement(dataset)) as copy:
            for batch in parquet_file.iter_batches(batch_size=COPY_BATCH_ROWS):
                table = pa.Table.from_batches([batch])
                if dataset == "stackexchange":
                    rows = _batch_to_stackexchange_rows(table)
                else:
                    rows = _batch_to_reddit_rows(table)
                copy.write(_copy_csv_rows(rows))
                total_rows += len(rows)
                if total_rows % 5000 == 0:
                    print(f"{dataset}: copied {total_rows:,} rows", flush=True)
    conn.commit()
    elapsed = max(time.monotonic() - started_at, 0.001)
    print(
        f"{dataset}: copied {total_rows:,} rows in {elapsed:.1f}s",
        flush=True,
    )
    return total_rows


def _build_index(
    conn: psycopg.Connection[Any], dataset: str, config: dict[str, Any]
) -> None:
    hnsw_config = config["benchmark"]["hnsw"]
    table = sql.Identifier(table_name(dataset))
    index = sql.Identifier(index_name(dataset))
    started_at = time.monotonic()
    print(f"{dataset}: building HNSW index {index_name(dataset)}", flush=True)
    with conn.cursor() as cur:
        cur.execute("SET maintenance_work_mem = '1GB';")
        cur.execute(
            sql.SQL(
                """
                CREATE INDEX {} ON {}
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = {}, ef_construction = {});
                """
            ).format(
                index,
                table,
                sql.Literal(int(hnsw_config["m"])),
                sql.Literal(int(hnsw_config["ef_construction"])),
            )
        )
        cur.execute(sql.SQL("ANALYZE {};").format(table))
    conn.commit()
    elapsed = max(time.monotonic() - started_at, 0.001)
    print(f"{dataset}: built HNSW index in {elapsed:.1f}s", flush=True)


def _load_one(
    conn: psycopg.Connection[Any],
    dataset: str,
    config: dict[str, Any],
    data_dir: pathlib.Path,
    force: bool,
) -> dict[str, Any]:
    paths = embedding_paths(data_dir, dataset, config)
    for path in paths.values():
        if not path.exists():
            raise BenchmarkError(f"Missing Stage 2 input: {path}")

    _ensure_extension(conn)
    _ensure_manifest_table(conn)
    versions = _postgres_versions(conn)
    expectations = _cache_expectations(dataset, config, data_dir, versions)
    db_manifest = _load_db_manifest(conn, dataset)
    if (
        not force
        and db_manifest is not None
        and cache_matches(db_manifest, expectations)
        and _table_exists(conn, table_name(dataset))
        and _index_exists(conn, index_name(dataset))
    ):
        print(f"Using cached Postgres table/index for {dataset}.", flush=True)
        return db_manifest

    dimension = int(config["model"]["embedding_dim"])
    _create_table(conn, dataset, dimension)
    row_count = _load_rows(conn, dataset, paths["docs"])
    _build_index(conn, dataset, config)
    manifest = {
        **expectations,
        "row_count": row_count,
        "created_at_unix": int(time.time()),
    }
    _write_db_manifest(conn, dataset, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("stackexchange", "reddit", "all"),
        default="all",
        help="Dataset to load. Defaults to both datasets.",
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("DATABASE_URL", DEFAULT_DSN),
        help="Postgres connection string.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop/reload tables and rebuild indexes even when cached.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(DEFAULT_CONFIG)
    selected = (
        ["stackexchange", "reddit"]
        if args.dataset == "all"
        else [args.dataset]
    )
    with psycopg.connect(args.dsn, autocommit=False) as conn:
        for dataset in selected:
            manifest = _load_one(
                conn=conn,
                dataset=dataset,
                config=config,
                data_dir=DEFAULT_DATA_DIR,
                force=args.force,
            )
            print(
                f"Wrote {dataset} Postgres table: "
                f"{manifest['row_count']:,} rows",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BenchmarkError, OSError, psycopg.Error) as exc:
        print(f"pg_load.py: error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
