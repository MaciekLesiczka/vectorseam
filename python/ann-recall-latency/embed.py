"""Embed normalized benchmark documents and queries."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
import sentence_transformers

from common import (
    BenchmarkError,
    cache_matches,
    DEFAULT_CONFIG,
    DEFAULT_DATA_DIR,
    MANIFEST_NAME,
    file_fingerprint,
    load_json,
    load_yaml,
    model_key,
    write_json,
)


PROGRESS_INTERVAL_SECONDS = 5.0


def _processed_dir(data_dir: pathlib.Path, dataset: str) -> pathlib.Path:
    return data_dir / "processed" / dataset


def _embedding_dir(
    embedding_root: pathlib.Path,
    dataset: str,
    model_name: str,
    revision: str | None,
) -> pathlib.Path:
    return embedding_root / dataset / model_key(model_name, revision)


def _manifest_path(output_dir: pathlib.Path) -> pathlib.Path:
    return output_dir / MANIFEST_NAME


def _preprocess_text(text: Any) -> str:
    """Shared identity preprocessing for docs and queries."""
    if text is None:
        return ""
    return str(text)


def _progress(message: str, last_at: float) -> float:
    now = time.monotonic()
    if now - last_at >= PROGRESS_INTERVAL_SECONDS:
        print(message, file=sys.stderr, flush=True)
        return now
    return last_at


def _embedding_array(embeddings: np.ndarray, dimension: int) -> pa.Array:
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32, copy=False)
    if embeddings.ndim != 2 or embeddings.shape[1] != dimension:
        raise BenchmarkError(
            f"Expected embeddings with shape (*, {dimension}), "
            f"got {embeddings.shape}"
        )
    flat = pa.array(embeddings.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, dimension)


def _append_embeddings(
    batch: pa.Table, embeddings: np.ndarray, dimension: int
) -> pa.Table:
    embedding_column = _embedding_array(embeddings, dimension)
    return batch.append_column("embedding", embedding_column)


def _read_table(path: pathlib.Path, max_rows: int | None) -> pa.Table:
    table = pq.read_table(path)
    if max_rows is not None:
        return table.slice(0, max_rows)
    return table


def _table_texts(batch: pa.Table) -> list[str]:
    return [
        _preprocess_text(value)
        for value in batch.column("text").combine_chunks().to_pylist()
    ]


def _encode_table(
    *,
    table: pa.Table,
    output_path: pathlib.Path,
    model: SentenceTransformer,
    batch_size: int,
    dimension: int,
    normalize_embeddings: bool,
    label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".parquet.tmp")
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    started_at = time.monotonic()
    last_progress_at = started_at

    try:
        for offset in range(0, table.num_rows, batch_size):
            batch = table.slice(offset, batch_size)
            embeddings = model.encode(
                _table_texts(batch),
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=False,
            )
            embedded_batch = _append_embeddings(batch, embeddings, dimension)
            if writer is None:
                writer = pq.ParquetWriter(
                    tmp_path, embedded_batch.schema, compression="zstd"
                )
            writer.write_table(embedded_batch)
            rows_written += embedded_batch.num_rows
            last_progress_at = _progress(
                f"{label}: embedded {rows_written:,}/{table.num_rows:,} rows",
                last_progress_at,
            )
    finally:
        if writer is not None:
            writer.close()

    tmp_path.replace(output_path)
    elapsed = max(time.monotonic() - started_at, 0.001)
    print(
        f"{label}: embedded {rows_written:,} rows in {elapsed:.1f}s",
        flush=True,
    )


def _input_paths(
    data_dir: pathlib.Path, dataset: str
) -> dict[str, pathlib.Path]:
    base_dir = _processed_dir(data_dir, dataset)
    return {
        "docs": base_dir / "docs.parquet",
        "queries": base_dir / "queries.parquet",
        "manifest": base_dir / MANIFEST_NAME,
    }


def _cache_expectations(
    *,
    dataset: str,
    config: dict[str, Any],
    data_dir: pathlib.Path,
) -> dict[str, Any]:
    model_config = config["model"]
    model_name = str(model_config["name"])
    revision = model_config.get("revision")
    revision_value = str(revision) if revision else None
    paths = _input_paths(data_dir, dataset)
    return {
        "dataset": dataset,
        "model_name": model_name,
        "model_revision": revision_value,
        "embedding_dim": int(model_config["embedding_dim"]),
        "device": str(model_config.get("device") or "cpu"),
        "normalize_embeddings": bool(model_config["normalize_embeddings"]),
        "preprocessing": "identity_after_stage1_normalization",
        "sentence_transformers_version": sentence_transformers.__version__,
        "input_docs": file_fingerprint(paths["docs"]),
        "input_queries": file_fingerprint(paths["queries"]),
    }


def _is_cached(output_dir: pathlib.Path, expectations: dict[str, Any]) -> bool:
    docs_path = output_dir / "docs.parquet"
    queries_path = output_dir / "queries.parquet"
    manifest_path = _manifest_path(output_dir)
    if (
        not docs_path.exists()
        or not queries_path.exists()
        or not manifest_path.exists()
    ):
        return False
    manifest = load_json(manifest_path)
    return cache_matches(manifest, expectations)


def _load_model(
    model_name: str, revision: str | None, device: str
) -> SentenceTransformer:
    kwargs: dict[str, Any] = {}
    if revision:
        kwargs["revision"] = revision
    if device:
        kwargs["device"] = device
    return SentenceTransformer(model_name, **kwargs)


def _embed_one(
    *,
    dataset: str,
    config: dict[str, Any],
    data_dir: pathlib.Path,
    embedding_root: pathlib.Path,
    force: bool,
) -> dict[str, Any]:
    model_config = config["model"]
    model_name = str(model_config["name"])
    revision = model_config.get("revision")
    revision_value = str(revision) if revision else None
    device = str(model_config.get("device") or "cpu")
    output_dir = _embedding_dir(
        embedding_root, dataset, model_name, revision_value
    )
    expectations = _cache_expectations(
        dataset=dataset,
        config=config,
        data_dir=data_dir,
    )
    if not force and _is_cached(output_dir, expectations):
        print(f"Using cached {dataset} embeddings.", flush=True)
        return load_json(_manifest_path(output_dir))

    paths = _input_paths(data_dir, dataset)
    for path in paths.values():
        if not path.exists():
            raise BenchmarkError(f"Missing Stage 1 input: {path}")

    model = _load_model(model_name, revision_value, device)
    dimension = int(model_config["embedding_dim"])
    batch_size = int(model_config["batch_size"])
    normalize_embeddings = bool(model_config["normalize_embeddings"])

    docs_table = _read_table(paths["docs"], None)
    queries_table = _read_table(paths["queries"], None)
    print(
        f"Embedding {dataset}: {docs_table.num_rows:,} docs, "
        f"{queries_table.num_rows:,} queries with {model_name} on {device}",
        flush=True,
    )
    _encode_table(
        table=docs_table,
        output_path=output_dir / "docs.parquet",
        model=model,
        batch_size=batch_size,
        dimension=dimension,
        normalize_embeddings=normalize_embeddings,
        label=f"{dataset} docs",
    )
    _encode_table(
        table=queries_table,
        output_path=output_dir / "queries.parquet",
        model=model,
        batch_size=batch_size,
        dimension=dimension,
        normalize_embeddings=normalize_embeddings,
        label=f"{dataset} queries",
    )

    manifest = {
        **expectations,
        "docs_path": str(output_dir / "docs.parquet"),
        "queries_path": str(output_dir / "queries.parquet"),
        "doc_count": docs_table.num_rows,
        "query_count": queries_table.num_rows,
        "generated_at_unix": int(time.time()),
    }
    write_json(_manifest_path(output_dir), manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("stackexchange", "reddit", "all"),
        default="all",
        help="Dataset to embed. Defaults to both datasets.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate embeddings even when a matching cache exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(DEFAULT_CONFIG)
    embedding_root = DEFAULT_DATA_DIR / "embeddings"
    selected = (
        ["stackexchange", "reddit"]
        if args.dataset == "all"
        else [args.dataset]
    )
    for dataset in selected:
        manifest = _embed_one(
            dataset=dataset,
            config=config,
            data_dir=DEFAULT_DATA_DIR,
            embedding_root=embedding_root,
            force=args.force,
        )
        print(
            f"Wrote {dataset} embeddings: {manifest['doc_count']:,} docs, "
            f"{manifest['query_count']:,} queries",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BenchmarkError, OSError, ValueError) as exc:
        print(f"embed.py: error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
