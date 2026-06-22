"""Compute exact brute-force top-k neighbors for benchmark queries."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from common import (
    BenchmarkError,
    cache_matches,
    DEFAULT_CONFIG,
    DEFAULT_DATA_DIR,
    MANIFEST_NAME,
    embedding_paths,
    file_fingerprint,
    fixed_size_list_matrix,
    load_json,
    load_yaml,
    write_json,
)


def _output_dir(data_dir: pathlib.Path, dataset: str) -> pathlib.Path:
    return data_dir / "ground_truth" / dataset


def _manifest_path(output_dir: pathlib.Path) -> pathlib.Path:
    return output_dir / MANIFEST_NAME


def _read_embeddings(
    path: pathlib.Path, id_column: str
) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=[id_column, "embedding"])
    ids = np.asarray(table.column(id_column).combine_chunks().to_numpy())
    embeddings = fixed_size_list_matrix(table, "embedding")
    return ids, embeddings


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "mps":
        if not torch.backends.mps.is_available():
            raise BenchmarkError(
                "ground_truth.device=mps but MPS is unavailable"
            )
        return torch.device("mps")
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise BenchmarkError(
                "ground_truth.device=cuda but CUDA is unavailable"
            )
        return torch.device("cuda")
    if requested_device == "cpu":
        return torch.device("cpu")
    raise BenchmarkError(f"Unsupported ground truth device: {requested_device}")


def _cache_expectations(
    dataset: str,
    config: dict[str, Any],
    data_dir: pathlib.Path,
    actual_device: str,
) -> dict[str, Any]:
    paths = embedding_paths(data_dir, dataset, config)
    ground_truth_config = config["ground_truth"]
    return {
        "dataset": dataset,
        "k": int(config["benchmark"]["k"]),
        "device": actual_device,
        "query_batch_size": int(ground_truth_config["query_batch_size"]),
        "metric": "cosine_via_dot_product_on_normalized_embeddings",
        "input_docs": file_fingerprint(paths["docs"]),
        "input_queries": file_fingerprint(paths["queries"]),
        "torch_version": torch.__version__,
    }


def _is_cached(output_dir: pathlib.Path, expectations: dict[str, Any]) -> bool:
    result_path = output_dir / "topk.parquet"
    manifest_path = _manifest_path(output_dir)
    if not result_path.exists() or not manifest_path.exists():
        return False
    manifest = load_json(manifest_path)
    cached_k = int(manifest.get("k", 0))
    requested_k = int(expectations["k"])
    if cached_k < requested_k:
        return False
    comparable_expectations = dict(expectations)
    comparable_expectations.pop("k")
    return cache_matches(manifest, comparable_expectations)


def _write_ground_truth(
    output_path: pathlib.Path,
    query_ids: np.ndarray,
    top_doc_ids: list[list[int]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".parquet.tmp")
    table = pa.Table.from_pydict(
        {
            "query_id": pa.array(query_ids, type=pa.int64()),
            "ground_truth_doc_ids": pa.array(
                top_doc_ids, type=pa.list_(pa.int64())
            ),
        }
    )
    pq.write_table(table, tmp_path, compression="zstd")
    tmp_path.replace(output_path)


def _compute_exact_topk(
    *,
    doc_ids: np.ndarray,
    doc_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    k: int,
    query_batch_size: int,
    device: torch.device,
    dataset: str,
) -> list[list[int]]:
    doc_tensor = torch.from_numpy(doc_embeddings).to(device)
    top_doc_ids: list[list[int]] = []
    started_at = time.monotonic()
    for start in range(0, query_embeddings.shape[0], query_batch_size):
        end = min(start + query_batch_size, query_embeddings.shape[0])
        query_tensor = torch.from_numpy(query_embeddings[start:end]).to(device)
        # Stage 2 writes unit-normalized embeddings, so dot product equals cosine.
        scores = query_tensor @ doc_tensor.T
        indices = torch.topk(
            scores, k=k, dim=1, largest=True, sorted=True
        ).indices
        for row in indices.cpu().numpy():
            top_doc_ids.append([int(doc_ids[index]) for index in row])
        print(
            f"{dataset}: exact top-k for "
            f"{end:,}/{query_embeddings.shape[0]:,} queries",
            flush=True,
        )
    elapsed = max(time.monotonic() - started_at, 0.001)
    print(f"{dataset}: exact top-k complete in {elapsed:.1f}s", flush=True)
    return top_doc_ids


def _compute_one(
    dataset: str,
    config: dict[str, Any],
    data_dir: pathlib.Path,
    force: bool,
) -> dict[str, Any]:
    paths = embedding_paths(data_dir, dataset, config)
    for path in paths.values():
        if not path.exists():
            raise BenchmarkError(f"Missing Stage 2 input: {path}")

    requested_device = str(config["ground_truth"]["device"])
    device = _resolve_device(requested_device)
    output_dir = _output_dir(data_dir, dataset)
    expectations = _cache_expectations(dataset, config, data_dir, device.type)
    if not force and _is_cached(output_dir, expectations):
        print(f"Using cached exact ground truth for {dataset}.", flush=True)
        return load_json(_manifest_path(output_dir))

    k = int(config["benchmark"]["k"])
    query_batch_size = int(config["ground_truth"]["query_batch_size"])
    doc_ids, doc_embeddings = _read_embeddings(paths["docs"], "doc_id")
    query_ids, query_embeddings = _read_embeddings(paths["queries"], "query_id")
    if k > doc_embeddings.shape[0]:
        raise BenchmarkError(
            f"k={k} exceeds document count {doc_embeddings.shape[0]}"
        )

    print(
        f"Computing exact {dataset} top-{k}: "
        f"{query_embeddings.shape[0]:,} queries x "
        f"{doc_embeddings.shape[0]:,} docs on {device.type}",
        flush=True,
    )
    top_doc_ids = _compute_exact_topk(
        doc_ids=doc_ids,
        doc_embeddings=doc_embeddings,
        query_embeddings=query_embeddings,
        k=k,
        query_batch_size=query_batch_size,
        device=device,
        dataset=dataset,
    )

    result_path = output_dir / "topk.parquet"
    _write_ground_truth(result_path, query_ids, top_doc_ids)
    manifest = {
        **expectations,
        "result_path": str(result_path),
        "doc_count": int(doc_embeddings.shape[0]),
        "query_count": int(query_embeddings.shape[0]),
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
        help="Dataset to process. Defaults to both datasets.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate exact ground truth even when a matching cache exists.",
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
    for dataset in selected:
        manifest = _compute_one(
            dataset, config, DEFAULT_DATA_DIR, args.force
        )
        print(
            f"Wrote {dataset} ground truth: "
            f"{manifest['query_count']:,} queries",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BenchmarkError, OSError, RuntimeError, ValueError) as exc:
        print(f"ground_truth.py: error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
