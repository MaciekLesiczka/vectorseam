"""Analyze pgvector ANN sweep outputs and generate cross-dataset plots."""

from __future__ import annotations

import argparse
import csv
import pathlib
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from common import BenchmarkError, DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR, load_json


DATASETS = ("stackexchange", "reddit")
TARGET_RECALL = 0.9
MEAN_RECALL_TARGET = 0.95
DEFAULT_FIXED_EF = 40
TRAIN_FRACTION = 0.7
SPLIT_SEED = 7


def _load_sweep_rows(results_dir: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        path = results_dir / f"sweep_{dataset}.parquet"
        if not path.exists():
            raise BenchmarkError(f"Missing sweep output: {path}")
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def _group_by_dataset_ef(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["dataset"]), int(row["ef"]))
        grouped.setdefault(key, []).append(row)
    return grouped


def _percentile(values: list[float], percentile: float) -> float:
    return float(
        np.percentile(np.asarray(values, dtype=np.float64), percentile)
    )


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for (dataset, ef), group_rows in sorted(_group_by_dataset_ef(rows).items()):
        recall = [float(row["recall"]) for row in group_rows]
        latency = [float(row["latency_ms"]) for row in group_rows]
        summaries.append(
            {
                "dataset": dataset,
                "ef": ef,
                "query_count": len(group_rows),
                "recall_mean": float(np.mean(recall)),
                "recall_p90": _percentile(recall, 90),
                "recall_p50": _percentile(recall, 50),
                "recall_p10": _percentile(recall, 10),
                "recall_p05": _percentile(recall, 5),
                "frac_below_0.9": float(np.mean(np.asarray(recall) < 0.9)),
                "latency_mean_ms": float(np.mean(latency)),
                "latency_p50_ms": _percentile(latency, 50),
                "latency_p90_ms": _percentile(latency, 90),
            }
        )
    return summaries


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise BenchmarkError("No rows to write")
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _dataset_summaries(
    summaries: list[dict[str, Any]], dataset: str
) -> list[dict[str, Any]]:
    return [row for row in summaries if row["dataset"] == dataset]


def _plot_recall(
    summary_rows: list[dict[str, Any]], output_path: pathlib.Path
) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    for dataset in DATASETS:
        rows = _dataset_summaries(summary_rows, dataset)
        ef = np.asarray([row["ef"] for row in rows], dtype=np.float64)
        p50 = np.asarray([row["recall_p50"] for row in rows])
        p10 = np.asarray([row["recall_p10"] for row in rows])
        p90 = np.asarray([row["recall_p90"] for row in rows])
        axis.plot(ef, p50, marker="o", label=f"{dataset} p50")
        axis.fill_between(ef, p10, p90, alpha=0.18)
    axis.axhline(TARGET_RECALL, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("hnsw.ef_search")
    axis.set_ylabel("Recall@10")
    axis.set_title("Recall Distribution vs ef_search")
    axis.set_ylim(0.0, 1.03)
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _target_crossing_ef(
    rows: list[dict[str, Any]], target_recall: float
) -> float | None:
    ef = np.asarray([float(row["ef"]) for row in rows], dtype=np.float64)
    recall = np.asarray(
        [float(row["recall_mean"]) for row in rows], dtype=np.float64
    )
    order = np.argsort(ef)
    ef = ef[order]
    recall = recall[order]

    for index in range(len(ef) - 1):
        left_recall = recall[index]
        right_recall = recall[index + 1]
        if left_recall == target_recall:
            return float(ef[index])
        if (left_recall - target_recall) * (
            right_recall - target_recall
        ) <= 0:
            if right_recall == left_recall:
                return float(ef[index])
            fraction = (target_recall - left_recall) / (
                right_recall - left_recall
            )
            return float(ef[index] + fraction * (ef[index + 1] - ef[index]))
    if len(ef) and recall[-1] == target_recall:
        return float(ef[-1])
    return None


def _plot_mean_recall_gap(
    summary_rows: list[dict[str, Any]],
    output_path: pathlib.Path,
    target_recall: float = MEAN_RECALL_TARGET,
) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    crossings = {}
    min_recall = min(float(row["recall_mean"]) for row in summary_rows)
    y_min = max(0.0, min_recall - 0.05)
    for dataset in DATASETS:
        rows = _dataset_summaries(summary_rows, dataset)
        ef = np.asarray([row["ef"] for row in rows], dtype=np.float64)
        recall = np.asarray([row["recall_mean"] for row in rows])
        axis.plot(ef, recall, marker="o", label=f"{dataset} mean")
        crossing = _target_crossing_ef(rows, target_recall)
        if crossing is not None:
            crossings[dataset] = crossing
            axis.vlines(
                crossing,
                y_min,
                target_recall,
                color="gray",
                linestyle=":",
                linewidth=1,
            )
            axis.text(
                crossing,
                target_recall - 0.035,
                f"{dataset}\nef={crossing:.1f}",
                ha="center",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
            )

    axis.axhline(
        target_recall,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"target {target_recall:.2f}",
    )
    if len(crossings) == len(DATASETS):
        left = min(crossings.values())
        right = max(crossings.values())
        gap = right - left
        axis.annotate(
            "",
            xy=(right, target_recall),
            xytext=(left, target_recall),
            arrowprops={"arrowstyle": "<->", "color": "black", "linewidth": 1.4},
        )
        axis.text(
            (left + right) / 2.0,
            min(target_recall + 0.02, 1.0),
            f"ef gap = {gap:.1f}",
            ha="center",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )

    axis.set_xlabel("hnsw.ef_search")
    axis.set_ylabel("Mean recall@10")
    axis.set_title("Mean Recall@10 vs ef_search")
    axis.set_ylim(y_min, 1.01)
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _load_doc_counts(data_dir: pathlib.Path) -> dict[str, int]:
    counts = {}
    for dataset in DATASETS:
        path = data_dir / "embeddings" / dataset
        manifests = sorted(path.glob("*/manifest.json"))
        if not manifests:
            raise BenchmarkError(f"Missing embedding manifest under {path}")
        manifest = load_json(manifests[0])
        counts[dataset] = int(manifest["doc_count"])
    return counts


def _plot_latency(
    summary_rows: list[dict[str, Any]],
    doc_counts: dict[str, int],
    output_path: pathlib.Path,
) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    max_latency = max(float(row["latency_p90_ms"]) for row in summary_rows)
    y_max = max_latency * 1.15
    for index, dataset in enumerate(DATASETS):
        rows = _dataset_summaries(summary_rows, dataset)
        ef = np.asarray([row["ef"] for row in rows], dtype=np.float64)
        latency_p50 = np.asarray([row["latency_p50_ms"] for row in rows])
        latency_p90 = np.asarray([row["latency_p90_ms"] for row in rows])
        label = f"{dataset} p50, N_DOCS={doc_counts[dataset]:,}"
        (line,) = axis.plot(ef, latency_p50, marker="o", label=label)
        color = line.get_color()
        axis.plot(
            ef,
            latency_p90,
            color=color,
            linestyle="--",
            marker="^",
            alpha=0.75,
            label=f"{dataset} p90",
        )
        crossing = _target_crossing_ef(rows, MEAN_RECALL_TARGET)
        if crossing is not None:
            axis.axvline(
                crossing,
                color=color,
                linestyle=":",
                linewidth=1.4,
            )
            axis.text(
                crossing,
                y_max * (0.92 - 0.12 * index),
                f"{dataset}\nmean recall=0.95\nef={crossing:.1f}",
                ha="center",
                va="top",
                fontsize=8,
                color=color,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
            )
    axis.set_xlabel("hnsw.ef_search")
    axis.set_ylabel("Query latency (ms)")
    axis.set_title("Latency vs ef_search")
    axis.set_ylim(0.0, y_max)
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_fixed_ef(
    rows: list[dict[str, Any]], fixed_ef: int, output_path: pathlib.Path
) -> None:
    values = []
    labels = []
    for dataset in DATASETS:
        recall = [
            float(row["recall"])
            for row in rows
            if row["dataset"] == dataset and int(row["ef"]) == fixed_ef
        ]
        if not recall:
            raise BenchmarkError(f"No rows for {dataset} at ef={fixed_ef}")
        values.append(recall)
        labels.append(dataset)
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.boxplot(values, tick_labels=labels, showmeans=True)
    axis.axhline(TARGET_RECALL, color="black", linestyle="--", linewidth=1)
    axis.set_ylabel("Recall@10")
    axis.set_title(f"Recall Distribution at ef_search={fixed_ef}")
    axis.set_ylim(0.0, 1.03)
    axis.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _split_query_ids(
    rows: list[dict[str, Any]], dataset: str
) -> tuple[set[int], set[int]]:
    query_ids = sorted(
        {int(row["query_id"]) for row in rows if row["dataset"] == dataset}
    )
    rng = np.random.default_rng(SPLIT_SEED)
    shuffled = np.asarray(query_ids, dtype=np.int64)
    rng.shuffle(shuffled)
    split_index = int(len(shuffled) * TRAIN_FRACTION)
    train = {int(value) for value in shuffled[:split_index]}
    test = {int(value) for value in shuffled[split_index:]}
    return train, test


def _p10_for_subset(
    rows: list[dict[str, Any]], dataset: str, ef: int, query_ids: set[int]
) -> float:
    recall = [
        float(row["recall"])
        for row in rows
        if row["dataset"] == dataset
        and int(row["ef"]) == ef
        and int(row["query_id"]) in query_ids
    ]
    if not recall:
        raise BenchmarkError(f"No recall rows for {dataset} ef={ef}")
    return _percentile(recall, 10)


def _calibration_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calibration = []
    ef_values = sorted({int(row["ef"]) for row in rows})
    for dataset in DATASETS:
        train_ids, test_ids = _split_query_ids(rows, dataset)
        train_p10_by_ef = {}
        for ef in ef_values:
            train_p10_by_ef[ef] = _p10_for_subset(
                rows, dataset, ef, train_ids
            )
        clearing = [
            ef for ef, p10 in train_p10_by_ef.items() if p10 >= TARGET_RECALL
        ]
        selected_ef = min(clearing) if clearing else max(ef_values)
        train_p10 = train_p10_by_ef[selected_ef]
        test_p10 = _p10_for_subset(rows, dataset, selected_ef, test_ids)
        calibration.append(
            {
                "dataset": dataset,
                "selected_ef": selected_ef,
                "train_p10": train_p10,
                "test_p10": test_p10,
                "test_clears_0.9": test_p10 >= TARGET_RECALL,
                "train_queries": len(train_ids),
                "test_queries": len(test_ids),
                "selection_note": (
                    "min ef with train p10>=0.9"
                    if clearing
                    else "no ef cleared train p10>=0.9"
                ),
            }
        )
    return calibration


def _print_summary(summary_rows: list[dict[str, Any]]) -> None:
    headers = [
        "dataset",
        "ef",
        "mean",
        "p50",
        "p10",
        "p05",
        "frac<0.9",
        "lat_p50",
    ]
    print(",".join(headers))
    for row in summary_rows:
        print(
            ",".join(
                [
                    row["dataset"],
                    str(row["ef"]),
                    f"{row['recall_mean']:.4f}",
                    f"{row['recall_p50']:.4f}",
                    f"{row['recall_p10']:.4f}",
                    f"{row['recall_p05']:.4f}",
                    f"{row['frac_below_0.9']:.4f}",
                    f"{row['latency_p50_ms']:.4f}",
                ]
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=pathlib.Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing sweep outputs and generated reports.",
    )
    parser.add_argument(
        "--fixed-ef",
        type=int,
        default=DEFAULT_FIXED_EF,
        help="ef_search value for the fixed-ef distribution plot.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = _load_sweep_rows(args.results_dir)
    summary_rows = _summary_rows(rows)
    calibration = _calibration_rows(rows)
    doc_counts = _load_doc_counts(DEFAULT_DATA_DIR)
    _write_csv(args.results_dir / "summary_by_dataset_ef.csv", summary_rows)
    _write_csv(args.results_dir / "calibration_transfer.csv", calibration)
    _plot_recall(summary_rows, args.results_dir / "recall_vs_ef.png")
    _plot_mean_recall_gap(
        summary_rows, args.results_dir / "mean_recall_vs_ef_gap.png"
    )
    _plot_latency(
        summary_rows, doc_counts, args.results_dir / "latency_vs_ef.png"
    )
    _plot_fixed_ef(
        rows,
        args.fixed_ef,
        args.results_dir / f"recall_fixed_ef_{args.fixed_ef}.png",
    )
    _print_summary(summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
