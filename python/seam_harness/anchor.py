"""Run the trusted ANN pipeline over the deterministic F-pg fixture."""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any, Callable

import numpy as np
import psycopg
import torch


_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
_ANCHOR_ROOT = _REPOSITORY_ROOT / "python" / "ann-recall-latency"
_DEFAULT_FIXTURE_ROOT = (
    _REPOSITORY_ROOT / "target" / "seam-fixtures" / "f-pg"
)
_DEFAULT_DSN = "postgresql://postgres:password@localhost:55432/postgres"
_DATASET = "seam_fixture"
_K = 10
_EF_GRID = [10, 20, 40, 80, 160]
_PERCENTILE = 0.90
_VALUE = 0.8
_TRAIN_FRACTION = 0.7
_SPLIT_SEED = 7


def fnv1a64(data: bytes) -> int:
    """Returns the five-line FNV-1a 64 reference hash."""
    value = 0xCBF29CE484222325
    for byte in data:
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


def is_train(vector_hash: int) -> bool:
    """Returns deterministic train membership from the frozen split rule."""
    split_input = f"s:{_SPLIT_SEED}:{vector_hash}".encode("ascii")
    threshold = round(_TRAIN_FRACTION * 10_000)
    return fnv1a64(split_input) % 10_000 < threshold


def _load_anchor_modules() -> tuple[Any, Any, Any, Any]:
    if str(_ANCHOR_ROOT) not in sys.path:
        sys.path.insert(0, str(_ANCHOR_ROOT))
    import analyze  # pylint: disable=import-outside-toplevel
    import common  # pylint: disable=import-outside-toplevel
    import ground_truth  # pylint: disable=import-outside-toplevel
    import sweep  # pylint: disable=import-outside-toplevel

    return ground_truth, sweep, analyze, common


def _vector_hashes(query_embeddings: np.ndarray) -> list[int]:
    hashes = []
    for embedding in query_embeddings:
        little_endian = np.ascontiguousarray(embedding, dtype=np.dtype("<f4"))
        hashes.append(fnv1a64(little_endian.tobytes(order="C")))
    return hashes


def _split_query_ids(
    query_ids: np.ndarray, vector_hashes: list[int]
) -> tuple[set[int], set[int]]:
    train = set()
    test = set()
    for query_id, vector_hash in zip(query_ids, vector_hashes):
        destination = train if is_train(vector_hash) else test
        destination.add(int(query_id))
    return train, test


def _run_calibration_with_hash_split(
    analyze: Any,
    rows: list[dict[str, Any]],
    train_ids: set[int],
    test_ids: set[int],
) -> dict[str, Any]:
    original_datasets = analyze.DATASETS
    original_split: Callable[..., Any] = analyze._split_query_ids
    original_target_recall = analyze.TARGET_RECALL

    def fixture_split(
        unused_rows: list[dict[str, Any]], unused_dataset: str
    ) -> tuple[set[int], set[int]]:
        return train_ids, test_ids

    try:
        analyze.DATASETS = (_DATASET,)
        analyze._split_query_ids = fixture_split
        analyze.TARGET_RECALL = _VALUE
        calibration = analyze._calibration_rows(rows)
    finally:
        analyze.DATASETS = original_datasets
        analyze._split_query_ids = original_split
        analyze.TARGET_RECALL = original_target_recall
    if len(calibration) != 1:
        raise RuntimeError("anchor calibration must emit exactly one dataset")
    return calibration[0]


def run_anchor(
    fixture_root: pathlib.Path,
    output_path: pathlib.Path,
    dsn: str,
) -> None:
    """Runs existing anchor math and writes Rust-readable comparison JSON."""
    ground_truth, sweep, analyze, common = _load_anchor_modules()
    doc_ids, doc_embeddings = ground_truth._read_embeddings(
        fixture_root / "docs.parquet", "doc_id"
    )
    query_ids, query_embeddings = ground_truth._read_embeddings(
        fixture_root / "queries.parquet", "query_id"
    )
    top_doc_ids = ground_truth._compute_exact_topk(
        doc_ids=doc_ids,
        doc_embeddings=doc_embeddings,
        query_embeddings=query_embeddings,
        k=_K,
        query_batch_size=64,
        device=torch.device("cpu"),
        dataset=_DATASET,
    )
    ground_truth_path = fixture_root / "anchor" / "ground_truth.parquet"
    ground_truth._write_ground_truth(
        ground_truth_path, query_ids, top_doc_ids
    )
    ground_truth_by_query = sweep._load_ground_truth(ground_truth_path, _K)

    rows = []
    with psycopg.connect(dsn, autocommit=False) as connection:
        for ef_search in _EF_GRID:
            for query_id, query_vector in zip(query_ids, query_embeddings):
                query_id_int = int(query_id)
                rows.append(
                    sweep._sweep_query(
                        connection,
                        _DATASET,
                        ef_search,
                        query_id_int,
                        query_vector,
                        ground_truth_by_query[query_id_int],
                        _K,
                        1,
                    )
                )

    vector_hashes = _vector_hashes(query_embeddings)
    train_ids, test_ids = _split_query_ids(query_ids, vector_hashes)
    summary_rows = analyze._summary_rows(rows)
    calibration = _run_calibration_with_hash_split(
        analyze, rows, train_ids, test_ids
    )
    per_ef = []
    for summary in summary_rows:
        ef_search = int(summary["ef"])
        per_ef.append(
            {
                "ef": ef_search,
                "mean_recall": float(summary["recall_mean"]),
                "train_quantile_recall": analyze._p10_for_subset(
                    rows, _DATASET, ef_search, train_ids
                ),
            }
        )
    comparison = {
        "format_version": 1,
        "dataset": _DATASET,
        "k": _K,
        "ef_grid": _EF_GRID,
        "percentile": _PERCENTILE,
        "value": _VALUE,
        "train_fraction": _TRAIN_FRACTION,
        "split_seed": _SPLIT_SEED,
        "query_order": [int(query_id) for query_id in query_ids],
        "vector_hashes": vector_hashes,
        "train_query_ids": sorted(train_ids),
        "test_query_ids": sorted(test_ids),
        "recall_rows": rows,
        "per_ef": per_ef,
        "recommended_ef": int(calibration["selected_ef"]),
        "train_quantile_recall": float(calibration["train_p10"]),
        "test_quantile_recall": float(calibration["test_p10"]),
        "transferred": bool(calibration["test_clears_0.9"]),
    }
    common.write_json(output_path, comparison)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture-root",
        type=pathlib.Path,
        default=_DEFAULT_FIXTURE_ROOT,
    )
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--dsn", default=_DEFAULT_DSN)
    return parser.parse_args()


def main() -> int:
    """Runs the anchor driver."""
    args = parse_args()
    output = args.output or args.fixture_root / "anchor" / "comparison.json"
    run_anchor(args.fixture_root, output, args.dsn)
    print(f"wrote anchor comparison to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
