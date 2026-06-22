"""Sweep pgvector HNSW ef_search and record recall/latency rows."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time
from typing import Any

import numpy as np
import psycopg
from psycopg import sql
import pyarrow as pa
import pyarrow.parquet as pq

from common import (
    BenchmarkError,
    cache_matches,
    configured_embedding_dir,
    DEFAULT_CONFIG,
    DEFAULT_DATA_DIR,
    DEFAULT_RESULTS_DIR,
    file_fingerprint,
    fixed_size_list_matrix,
    format_vector,
    load_json,
    load_yaml,
    table_name,
    write_json,
)


DEFAULT_DSN = "postgresql://postgres:password@localhost:5432/postgres"
DB_MANIFEST_TABLE = "ann_recall_latency_manifests"


def _query_embeddings_path(
    data_dir: pathlib.Path, dataset: str, config: dict[str, Any]
) -> pathlib.Path:
    return (
        configured_embedding_dir(data_dir, dataset, config) / "queries.parquet"
    )


def _query_embedding_manifest_path(
    data_dir: pathlib.Path, dataset: str, config: dict[str, Any]
) -> pathlib.Path:
    return (
        configured_embedding_dir(data_dir, dataset, config) / "manifest.json"
    )


def _ground_truth_path(data_dir: pathlib.Path, dataset: str) -> pathlib.Path:
    return data_dir / "ground_truth" / dataset / "topk.parquet"


def _ground_truth_manifest_path(
    data_dir: pathlib.Path, dataset: str
) -> pathlib.Path:
    return data_dir / "ground_truth" / dataset / "manifest.json"


def _result_path(results_dir: pathlib.Path, dataset: str) -> pathlib.Path:
    return results_dir / f"sweep_{dataset}.parquet"


def _manifest_path(results_dir: pathlib.Path, dataset: str) -> pathlib.Path:
    return results_dir / f"sweep_{dataset}.json"


def _embedding_array(table: pa.Table) -> np.ndarray:
    return fixed_size_list_matrix(table, "embedding")


def _load_queries(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["query_id", "embedding"])
    query_ids = np.asarray(table.column("query_id").combine_chunks().to_numpy())
    return query_ids, _embedding_array(table)


def _load_ground_truth(path: pathlib.Path, k: int) -> dict[int, set[int]]:
    table = pq.read_table(path)
    query_ids = table.column("query_id").combine_chunks().to_pylist()
    doc_id_lists = (
        table.column("ground_truth_doc_ids").combine_chunks().to_pylist()
    )
    return {
        int(query_id): {int(doc_id) for doc_id in doc_ids[:k]}
        for query_id, doc_ids in zip(query_ids, doc_id_lists)
    }


def _load_db_manifest(
    conn: psycopg.Connection[Any], dataset: str
) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT manifest FROM {} WHERE dataset = %s;").format(
                sql.Identifier(DB_MANIFEST_TABLE)
            ),
            (dataset,),
        )
        row = cur.fetchone()
    if row is None:
        raise BenchmarkError(f"Missing Stage 3 DB manifest for {dataset}")
    manifest = row[0]
    if isinstance(manifest, str):
        return json.loads(manifest)
    return dict(manifest)


def _cache_expectations(
    *,
    conn: psycopg.Connection[Any],
    dataset: str,
    config: dict[str, Any],
    data_dir: pathlib.Path,
    ef_grid: list[int],
    repeats: int,
) -> dict[str, Any]:
    query_path = _query_embeddings_path(data_dir, dataset, config)
    query_manifest_path = _query_embedding_manifest_path(
        data_dir, dataset, config
    )
    ground_truth_path = _ground_truth_path(data_dir, dataset)
    ground_truth_manifest_path = _ground_truth_manifest_path(data_dir, dataset)
    return {
        "dataset": dataset,
        "table_name": table_name(dataset),
        "k": int(config["benchmark"]["k"]),
        "ef_search": ef_grid,
        "repeats": repeats,
        "query_embeddings": file_fingerprint(query_path),
        "ground_truth": file_fingerprint(ground_truth_path),
        "db_manifest": _load_db_manifest(conn, dataset),
        "query": "ORDER BY embedding <=> $query_vector LIMIT k",
    }


def _is_cached(result_path: pathlib.Path, manifest_path: pathlib.Path) -> bool:
    return result_path.exists() and manifest_path.exists()


def _cached_manifest_matches(
    manifest_path: pathlib.Path, expectations: dict[str, Any]
) -> bool:
    manifest = load_json(manifest_path)
    return cache_matches(manifest, expectations)


def _execute_ann_query(
    conn: psycopg.Connection[Any],
    dataset: str,
    ef_search: int,
    query_vector: np.ndarray,
    k: int,
) -> tuple[list[int], float]:
    vector_text = format_vector(query_vector)
    table = sql.Identifier(table_name(dataset))
    statement = sql.SQL(
        """
        SELECT doc_id
        FROM {}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """
    ).format(table)
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SET LOCAL hnsw.ef_search = {};").format(
                    sql.Literal(int(ef_search))
                )
            )
            started_at = time.perf_counter()
            cur.execute(statement, (vector_text, k))
            rows = cur.fetchall()
            latency_ms = (time.perf_counter() - started_at) * 1000.0
    return [int(row[0]) for row in rows], latency_ms


def _sweep_query(
    conn: psycopg.Connection[Any],
    dataset: str,
    ef_search: int,
    query_id: int,
    query_vector: np.ndarray,
    ground_truth: set[int],
    k: int,
    repeats: int,
) -> dict[str, Any]:
    latencies = []
    returned_doc_ids: list[int] = []
    for _ in range(repeats):
        returned_doc_ids, latency_ms = _execute_ann_query(
            conn, dataset, ef_search, query_vector, k
        )
        latencies.append(latency_ms)
    recall = len(set(returned_doc_ids) & ground_truth) / float(k)
    return {
        "dataset": dataset,
        "query_id": query_id,
        "ef": ef_search,
        "recall": recall,
        "latency_ms": float(statistics.median(latencies)),
        "result_count": len(returned_doc_ids),
    }


def _write_results(rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".parquet.tmp")
    table = pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                ("dataset", pa.string()),
                ("query_id", pa.int64()),
                ("ef", pa.int64()),
                ("recall", pa.float64()),
                ("latency_ms", pa.float64()),
                ("result_count", pa.int64()),
            ]
        ),
    )
    pq.write_table(table, tmp_path, compression="zstd")
    tmp_path.replace(path)


def _run_one(
    *,
    conn: psycopg.Connection[Any],
    dataset: str,
    config: dict[str, Any],
    data_dir: pathlib.Path,
    results_dir: pathlib.Path,
    ef_grid: list[int],
    repeats: int,
    force: bool,
) -> dict[str, Any]:
    query_path = _query_embeddings_path(data_dir, dataset, config)
    ground_truth_path = _ground_truth_path(data_dir, dataset)
    for path in [
        query_path,
        _query_embedding_manifest_path(data_dir, dataset, config),
        ground_truth_path,
        _ground_truth_manifest_path(data_dir, dataset),
    ]:
        if not path.exists():
            raise BenchmarkError(f"Missing input for sweep: {path}")

    result_path = _result_path(results_dir, dataset)
    manifest_path = _manifest_path(results_dir, dataset)
    expectations = _cache_expectations(
        conn=conn,
        dataset=dataset,
        config=config,
        data_dir=data_dir,
        ef_grid=ef_grid,
        repeats=repeats,
    )
    if (
        not force
        and _is_cached(result_path, manifest_path)
        and _cached_manifest_matches(manifest_path, expectations)
    ):
        print(f"Using cached sweep results for {dataset}.", flush=True)
        return load_json(manifest_path)

    k = int(config["benchmark"]["k"])
    query_ids, query_embeddings = _load_queries(query_path)
    ground_truth = _load_ground_truth(ground_truth_path, k)
    rows: list[dict[str, Any]] = []
    started_at = time.monotonic()
    total = len(ef_grid) * len(query_ids)
    completed = 0
    print(
        f"Sweeping {dataset}: {len(query_ids):,} queries x "
        f"{len(ef_grid)} ef values x {repeats} repeats",
        flush=True,
    )
    for ef_search in ef_grid:
        ef_started_at = time.monotonic()
        for query_id, query_vector in zip(query_ids, query_embeddings):
            query_id_int = int(query_id)
            if query_id_int not in ground_truth:
                raise BenchmarkError(
                    f"Missing ground truth for query {query_id_int}"
                )
            rows.append(
                _sweep_query(
                    conn,
                    dataset,
                    ef_search,
                    query_id_int,
                    query_vector,
                    ground_truth[query_id_int],
                    k,
                    repeats,
                )
            )
            completed += 1
        elapsed = max(time.monotonic() - ef_started_at, 0.001)
        print(
            f"{dataset}: ef={ef_search} complete "
            f"({completed:,}/{total:,}) in {elapsed:.1f}s",
            flush=True,
        )

    _write_results(rows, result_path)
    manifest = {
        **expectations,
        "result_path": str(result_path),
        "row_count": len(rows),
        "query_count": len(query_ids),
        "generated_at_unix": int(time.time()),
        "elapsed_seconds": time.monotonic() - started_at,
    }
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("stackexchange", "reddit", "all"),
        default="all",
        help="Dataset to sweep. Defaults to both datasets.",
    )
    parser.add_argument(
        "--results-dir",
        type=pathlib.Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for sweep parquet outputs.",
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("DATABASE_URL", DEFAULT_DSN),
        help="Postgres connection string.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Override sweep.repeats.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate sweep rows even when a matching cache exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(DEFAULT_CONFIG)
    ef_grid = [int(value) for value in config["benchmark"]["ef_search"]]
    repeats = int(args.repeats or config["sweep"]["repeats"])
    if repeats < 1:
        raise BenchmarkError("--repeats must be at least 1")
    selected = (
        ["stackexchange", "reddit"]
        if args.dataset == "all"
        else [args.dataset]
    )
    with psycopg.connect(args.dsn, autocommit=False) as conn:
        for dataset in selected:
            manifest = _run_one(
                conn=conn,
                dataset=dataset,
                config=config,
                data_dir=DEFAULT_DATA_DIR,
                results_dir=args.results_dir,
                ef_grid=ef_grid,
                repeats=repeats,
                force=args.force,
            )
            print(
                f"Wrote {dataset} sweep rows: {manifest['row_count']:,}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BenchmarkError, OSError, psycopg.Error, ValueError) as exc:
        print(f"sweep.py: error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
