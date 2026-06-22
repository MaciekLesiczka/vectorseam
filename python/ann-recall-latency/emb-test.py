"""Inspect Reddit HNSW results for selected benchmark queries."""

from __future__ import annotations

import argparse
import os
import pathlib
from typing import Any

import numpy as np
import psycopg
from psycopg import sql
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from common import (
    BenchmarkError,
    configured_embedding_dir,
    DEFAULT_CONFIG,
    DEFAULT_DATA_DIR,
    fixed_size_list_matrix,
    format_vector,
    load_yaml,
)


DEFAULT_DSN = "postgresql://postgres:password@localhost:5432/postgres"
DEFAULT_QUERY_IDS = (1209752, 1834845)
DATASET = "reddit"


def _load_query_rows(
    data_dir: pathlib.Path, config: dict[str, Any], query_ids: list[int]
) -> dict[int, dict[str, Any]]:
    path = (
        configured_embedding_dir(data_dir, DATASET, config) / "queries.parquet"
    )
    table = pq.read_table(path)
    ids = table.column("query_id").combine_chunks().to_pylist()
    embeddings = fixed_size_list_matrix(table, "embedding")
    rows: dict[int, dict[str, Any]] = {}
    requested = set(query_ids)

    for index, query_id in enumerate(ids):
        query_id_int = int(query_id)
        if query_id_int in requested:
            rows[query_id_int] = {
                "text": table.column("text")[index].as_py(),
                "embedding": embeddings[index],
            }

    missing = requested - set(rows)
    if missing:
        raise BenchmarkError(
            f"Missing query embeddings for IDs: {sorted(missing)}"
        )
    return rows


def _run_hnsw_query(
    conn: psycopg.Connection[Any],
    query_vector: np.ndarray,
    ef_search: int,
    k: int,
) -> list[tuple[int, float]]:
    vector_text = format_vector(query_vector)
    statement = sql.SQL(
        """
        SELECT doc_id, embedding <=> %s::vector AS distance
        FROM {}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """
    ).format(sql.Identifier(f"docs_{DATASET}"))

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SET LOCAL hnsw.ef_search = {};").format(
                    sql.Literal(int(ef_search))
                )
            )
            cur.execute(statement, (vector_text, vector_text, k))
            return [(int(row[0]), float(row[1])) for row in cur.fetchall()]


def _load_doc_rows(
    data_dir: pathlib.Path, doc_ids: list[int]
) -> dict[int, dict[str, Any]]:
    path = data_dir / "processed" / DATASET / "docs.parquet"
    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table(
        columns=[
            "doc_id",
            "text",
            "community",
            "score",
            "datetime",
            "data_type",
        ],
        filter=ds.field("doc_id").isin(doc_ids),
    )
    rows = table.to_pylist()
    return {int(row["doc_id"]): row for row in rows}


def _print_result(
    query_id: int,
    query_text: str,
    results: list[tuple[int, float]],
    docs_by_id: dict[int, dict[str, Any]],
) -> None:
    print("=" * 100)
    print(f"query_id={query_id}")
    print(f"query_text={query_text}")
    print()

    for rank, (doc_id, distance) in enumerate(results, start=1):
        doc = docs_by_id.get(doc_id)
        if doc is None:
            raise BenchmarkError(f"Missing processed doc text for doc_id={doc_id}")
        text = " ".join(str(doc["text"]).split())
        print(f"{rank:02d}. doc_id={doc_id} distance={distance:.6f}")
        print(
            "    "
            f"community={doc.get('community')} "
            f"score={doc.get('score')} "
            f"datetime={doc.get('datetime')} "
            f"data_type={doc.get('data_type')}"
        )
        print(f"    text={text}")
        print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print Reddit document text selected by pgvector HNSW for selected "
            "query IDs."
        )
    )
    parser.add_argument(
        "--query-id",
        action="append",
        type=int,
        dest="query_ids",
        help="Query ID to inspect. Can be supplied more than once.",
    )
    parser.add_argument("--ef-search", type=int, default=400)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--dsn",
        default=os.environ.get("ANN_RECALL_LATENCY_DSN", DEFAULT_DSN),
        help="Postgres DSN. Defaults to ANN_RECALL_LATENCY_DSN or local Docker.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    query_ids = args.query_ids or list(DEFAULT_QUERY_IDS)
    config = load_yaml(DEFAULT_CONFIG)
    query_rows = _load_query_rows(DEFAULT_DATA_DIR, config, query_ids)

    with psycopg.connect(args.dsn) as conn:
        per_query_results = {
            query_id: _run_hnsw_query(
                conn=conn,
                query_vector=query_rows[query_id]["embedding"],
                ef_search=args.ef_search,
                k=args.k,
            )
            for query_id in query_ids
        }

    doc_ids = sorted(
        {
            doc_id
            for results in per_query_results.values()
            for doc_id, _ in results
        }
    )
    docs_by_id = _load_doc_rows(DEFAULT_DATA_DIR, doc_ids)

    print(
        f"dataset={DATASET} ef_search={args.ef_search} k={args.k} "
        f"queries={query_ids}"
    )
    for query_id in query_ids:
        _print_result(
            query_id=query_id,
            query_text=str(query_rows[query_id]["text"]),
            results=per_query_results[query_id],
            docs_by_id=docs_by_id,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
