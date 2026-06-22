"""Download pinned raw inputs for the ANN recall-latency benchmark."""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile
import sys
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from tqdm import tqdm

from common import (
    BenchmarkError,
    CHUNK_BYTES,
    DEFAULT_CONFIG,
    DEFAULT_DATA_DIR,
    MANIFEST_NAME,
    file_digest,
    load_json,
    load_yaml,
    write_json,
)


def _read_json_url(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def _download_url(url: str, destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, delete=False, suffix=".tmp"
    ) as tmp_file:
        tmp_path = pathlib.Path(tmp_file.name)
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                total_header = response.headers.get("Content-Length")
                total = int(total_header) if total_header else None
                progress = tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=destination.name,
                )
                with progress:
                    while True:
                        chunk = response.read(CHUNK_BYTES)
                        if not chunk:
                            break
                        tmp_file.write(chunk)
                        progress.update(len(chunk))
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    tmp_path.replace(destination)


def _verify_file(
    path: pathlib.Path,
    *,
    size: int | None,
    sha1: str | None,
    md5: str | None,
) -> dict[str, Any]:
    actual_size = path.stat().st_size
    if size is not None and actual_size != size:
        raise BenchmarkError(f"{path} size {actual_size} != expected {size}")

    actual_sha1 = file_digest(path, "sha1")
    if sha1 is not None and actual_sha1.lower() != sha1.lower():
        raise BenchmarkError(f"{path} sha1 {actual_sha1} != expected {sha1}")

    actual_md5 = file_digest(path, "md5")
    if md5 is not None and actual_md5.lower() != md5.lower():
        raise BenchmarkError(f"{path} md5 {actual_md5} != expected {md5}")

    return {"size": actual_size, "sha1": actual_sha1, "md5": actual_md5}


def _archive_metadata(config: dict[str, Any]) -> dict[str, Any]:
    item = _require_str(config, "archive_item")
    prefix = _require_str(config, "archive_prefix")
    file_name = _require_str(config, "file_name")
    target_name = f"{prefix}/{file_name}"
    metadata = _read_json_url(f"https://archive.org/metadata/{item}")

    for candidate in metadata.get("files", []):
        if candidate.get("name") == target_name:
            return candidate
    raise BenchmarkError(
        f"Archive.org item {item} does not contain {target_name}"
    )


def _require_str(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise BenchmarkError(f"Missing required string config key: {key}")
    return value


def _download_stackexchange(
    config: dict[str, Any], data_dir: pathlib.Path
) -> dict[str, Any]:
    metadata = _archive_metadata(config)
    expected_size = int(metadata["size"]) if metadata.get("size") else None
    expected_sha1 = config.get("sha1") or metadata.get("sha1")
    expected_md5 = config.get("md5") or metadata.get("md5")

    file_name = _require_str(config, "file_name")
    destination = data_dir / "raw" / "stackexchange" / file_name
    if not destination.exists():
        _download_url(_require_str(config, "url"), destination)

    verified = _verify_file(
        destination,
        size=expected_size,
        sha1=expected_sha1,
        md5=expected_md5,
    )
    return {
        "dataset": "stackexchange",
        "source": "archive.org",
        "archive_item": config["archive_item"],
        "dump_date": config["dump_date"],
        "site": config["site"],
        "file_name": file_name,
        "url": config["url"],
        "local_path": str(destination),
        "expected": {
            "size": expected_size,
            "sha1": expected_sha1,
            "md5": expected_md5,
        },
        "verified": verified,
    }


def _hf_repo_info(repo_id: str, revision: str) -> dict[str, Any]:
    url = f"https://huggingface.co/api/datasets/{repo_id}/revision/{revision}"
    return _read_json_url(url)


def _download_hf_file(
    repo_id: str, revision: str, repo_path: str, destination: pathlib.Path
) -> None:
    quoted_repo_path = urllib.parse.quote(repo_path, safe="/")
    url = (
        f"https://huggingface.co/datasets/{repo_id}/resolve/"
        f"{revision}/{quoted_repo_path}"
    )
    _download_url(url, destination)


def _hf_parquet_files(repo_id: str) -> list[dict[str, Any]]:
    quoted_repo_id = urllib.parse.quote(repo_id, safe="/")
    url = (
        "https://datasets-server.huggingface.co/parquet"
        f"?dataset={quoted_repo_id}"
    )
    response = _read_json_url(url)
    failed = response.get("failed") or []
    pending = response.get("pending") or []
    if failed or pending:
        raise BenchmarkError(
            f"Hugging Face parquet export has failed={failed} pending={pending}"
        )
    parquet_files = response.get("parquet_files")
    if not isinstance(parquet_files, list) or not parquet_files:
        raise BenchmarkError(f"No parquet files published for {repo_id}")
    return parquet_files


def _download_reddit_parquet(
    repo_id: str, revision: str, data_dir: pathlib.Path
) -> dict[str, Any]:
    parquet_files = _hf_parquet_files(repo_id)
    base_dir = data_dir / "raw" / "reddit" / revision
    files = []

    for parquet_file in parquet_files:
        config_name = str(parquet_file["config"])
        split = str(parquet_file["split"])
        filename = str(parquet_file["filename"])
        destination = base_dir / config_name / split / filename
        if not destination.exists():
            _download_url(str(parquet_file["url"]), destination)
        verified = _verify_file(
            destination,
            size=int(parquet_file["size"]),
            sha1=None,
            md5=None,
        )
        files.append(
            {
                "config": config_name,
                "split": split,
                "filename": filename,
                "url": parquet_file["url"],
                "local_path": str(destination),
                "size": verified["size"],
                "sha1": verified["sha1"],
                "md5": verified["md5"],
            }
        )

    return {
        "dataset": "reddit",
        "source": "huggingface",
        "repo_id": repo_id,
        "revision": revision,
        "parquet_export": "refs/convert/parquet",
        "files": files,
    }


def _download_reddit(
    config: dict[str, Any], data_dir: pathlib.Path
) -> dict[str, Any]:
    repo_id = _require_str(config, "repo_id")
    revision = _require_str(config, "revision")
    info = _hf_repo_info(repo_id, revision)
    resolved_revision = info.get("sha")
    if resolved_revision != revision:
        raise BenchmarkError(
            "Configured Reddit revision "
            f"{revision} resolved to {resolved_revision}"
        )
    if config.get("use_parquet_endpoint"):
        return _download_reddit_parquet(repo_id, revision, data_dir)

    repo_path = _require_str(config, "repo_path")
    siblings = info.get("siblings", [])
    sibling_by_path = {
        str(sibling.get("rfilename", "")): sibling for sibling in siblings
    }
    sibling = sibling_by_path.get(repo_path)
    if sibling is None:
        raise BenchmarkError(
            f"Hugging Face repo {repo_id} does not contain {repo_path}"
        )

    base_dir = data_dir / "raw" / "reddit" / revision
    destination = base_dir / repo_path
    if not destination.exists():
        _download_hf_file(repo_id, revision, repo_path, destination)
    size = sibling.get("size")
    verified = _verify_file(
        destination,
        size=int(size) if size is not None else None,
        sha1=None,
        md5=None,
    )

    return {
        "dataset": "reddit",
        "source": "huggingface",
        "repo_id": repo_id,
        "revision": revision,
        "repo_path": repo_path,
        "local_path": str(destination),
        "size": verified["size"],
        "sha1": verified["sha1"],
        "md5": verified["md5"],
    }


def _manifest_path(data_dir: pathlib.Path) -> pathlib.Path:
    return data_dir / MANIFEST_NAME


def _load_manifest(data_dir: pathlib.Path) -> dict[str, Any]:
    path = _manifest_path(data_dir)
    if not path.exists():
        return {"datasets": {}}
    manifest = load_json(path)
    manifest.setdefault("datasets", {})
    return manifest


def _write_manifest(data_dir: pathlib.Path, manifest: dict[str, Any]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    write_json(_manifest_path(data_dir), manifest)


def _download_one(
    dataset: str, config: dict[str, Any], data_dir: pathlib.Path
) -> dict[str, Any]:
    datasets = config.get("datasets")
    if not isinstance(datasets, dict) or dataset not in datasets:
        raise BenchmarkError(f"Unknown dataset: {dataset}")

    dataset_config = datasets[dataset]
    if dataset == "stackexchange":
        return _download_stackexchange(dataset_config, data_dir)
    if dataset == "reddit":
        return _download_reddit(dataset_config, data_dir)
    raise BenchmarkError(f"Unsupported dataset: {dataset}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("stackexchange", "reddit", "all"),
        default="all",
        help="Dataset to download. Defaults to both datasets.",
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

    manifest = _load_manifest(DEFAULT_DATA_DIR)
    manifest["generated_at_unix"] = int(time.time())
    manifest["config_path"] = str(DEFAULT_CONFIG.resolve())
    manifest.setdefault("datasets", {})

    for dataset in selected:
        print(f"Downloading {dataset} raw input...", flush=True)
        manifest["datasets"][dataset] = _download_one(
            dataset, config, DEFAULT_DATA_DIR
        )
        _write_manifest(DEFAULT_DATA_DIR, manifest)
        print(f"Wrote {dataset} manifest entry.", flush=True)

    print(f"Manifest: {_manifest_path(DEFAULT_DATA_DIR)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BenchmarkError, OSError, urllib.error.URLError) as exc:
        print(f"download.py: error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
