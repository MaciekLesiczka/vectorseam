"""Shared helpers for ANN recall-latency benchmark stages."""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from typing import Any, Iterable, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    import numpy as np
    import pyarrow as pa


BENCH_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_CONFIG = BENCH_DIR / "config.yaml"
DEFAULT_DATA_DIR = BENCH_DIR / "data"
DEFAULT_RESULTS_DIR = BENCH_DIR / "results"
MANIFEST_NAME = "manifest.json"
CHUNK_BYTES = 1024 * 1024
PATH_CACHE_KEYS = frozenset(
    {
        "config_path",
        "docs_path",
        "local_path",
        "path",
        "queries_path",
        "result_path",
    }
)


class BenchmarkError(RuntimeError):
    """Raised when a benchmark stage cannot complete reproducibly."""


def load_yaml(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)
    if not isinstance(config, dict):
        raise BenchmarkError(f"Config is not a YAML mapping: {path}")
    return config


def load_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        loaded = json.load(file_obj)
    if not isinstance(loaded, dict):
        raise BenchmarkError(f"Expected JSON object: {path}")
    return loaded


def write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as file_obj:
        json.dump(value, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    tmp_path.replace(path)


def file_digest(path: pathlib.Path, algorithm: str = "sha1") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: pathlib.Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha1": file_digest(path),
    }


def cache_value_matches(actual: Any, expected: Any) -> bool:
    """Return whether actual satisfies expected after dropping local paths.

    ANN benchmark artifacts can be expensive to regenerate. Their cache keys
    should depend on content and configuration, not on where the repo is
    checked out locally.
    """
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(
            cache_value_matches(actual.get(key), item)
            for key, item in expected.items()
            if key not in PATH_CACHE_KEYS
        )
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False
        return all(
            cache_value_matches(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    return actual == expected


def cache_matches(
    manifest: dict[str, Any], expectations: dict[str, Any]
) -> bool:
    """Return whether manifest satisfies path-insensitive expectations."""
    return all(
        cache_value_matches(manifest.get(key), value)
        for key, value in expectations.items()
    )


def safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def model_key(model_name: str, revision: str | None) -> str:
    key = safe_key(model_name)
    if revision:
        key = f"{key}__{safe_key(revision)}"
    return key


def configured_embedding_dir(
    data_dir: pathlib.Path, dataset: str, config: dict[str, Any]
) -> pathlib.Path:
    model_config = config["model"]
    revision = model_config.get("revision")
    return (
        data_dir
        / "embeddings"
        / dataset
        / model_key(
            str(model_config["name"]), str(revision) if revision else None
        )
    )


def embedding_paths(
    data_dir: pathlib.Path, dataset: str, config: dict[str, Any]
) -> dict[str, pathlib.Path]:
    base_dir = configured_embedding_dir(data_dir, dataset, config)
    return {
        "docs": base_dir / "docs.parquet",
        "queries": base_dir / "queries.parquet",
        "manifest": base_dir / "manifest.json",
    }


def table_name(dataset: str) -> str:
    return f"docs_{dataset}"


def index_name(dataset: str) -> str:
    return f"docs_{dataset}_embedding_hnsw_idx"


def format_vector(values: Iterable[float]) -> str:
    return "[" + ",".join(f"{float(value):.9g}" for value in values) + "]"


def fixed_size_list_matrix(
    table: "pa.Table", column_name: str
) -> "np.ndarray":
    import numpy as np
    import pyarrow as pa

    column = table.column(column_name).combine_chunks()
    if not pa.types.is_fixed_size_list(column.type):
        raise BenchmarkError(
            f"Expected fixed-size {column_name} column: {column.type}"
        )
    dimension = int(column.type.list_size)
    values = column.values.to_numpy(zero_copy_only=False)
    return np.array(values, dtype=np.float32, copy=True).reshape(
        table.num_rows, dimension
    )
