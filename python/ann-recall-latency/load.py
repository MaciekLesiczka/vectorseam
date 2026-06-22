"""Parse raw benchmark inputs into normalized parquet files."""

from __future__ import annotations

import argparse
import copy
import html
from html.parser import HTMLParser
import json
import pathlib
import random
import re
import subprocess
import sys
import time
from typing import Any, Iterable
import xml.etree.ElementTree as ET
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq

from common import (
    BenchmarkError,
    cache_matches,
    DEFAULT_CONFIG,
    DEFAULT_DATA_DIR,
    MANIFEST_NAME,
    load_json,
    load_yaml,
    write_json,
)


RAW_MANIFEST = DEFAULT_DATA_DIR / MANIFEST_NAME
STACKEXCHANGE_POSTS_XML = "Posts.xml"
REDDIT_JSON_NAME = "corpus-webis-tldr-17.json"
REDDIT_REPO_PATH = "data/corpus-webis-tldr-17.zip"
PROGRESS_INTERVAL_SECONDS = 5.0


class _TextExtractor(HTMLParser):
    """Small HTML-to-text extractor for StackExchange post bodies."""

    _BLOCK_TAGS = {
        "blockquote",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "ol",
        "p",
        "pre",
        "tr",
        "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del attrs
        if tag in self._BLOCK_TAGS:
            self._parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._BLOCK_TAGS:
            self._parts.append(" ")

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def text(self) -> str:
        return _normalize_text("".join(self._parts))


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    text = html.unescape(value)
    return re.sub(r"\s+", " ", text).strip()


def _strip_html(value: str | None) -> str:
    if not value:
        return ""
    parser = _TextExtractor()
    parser.feed(value)
    parser.close()
    return parser.text()


def _parse_int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _parse_tags(value: str | None) -> list[str]:
    if not value:
        return []
    return re.findall(r"<([^>]+)>", html.unescape(value))


def _is_low_signal(text: str) -> bool:
    normalized = text.strip().lower()
    return normalized in {
        "",
        ".",
        "?",
        "[deleted]",
        "[removed]",
        "deleted",
        "removed",
        "n/a",
        "na",
        "none",
        "null",
    }


def _accept_text(text: str, min_chars: int) -> bool:
    return len(text) >= min_chars and not _is_low_signal(text)


def _seed(config: dict[str, Any], dataset: str, stream_name: str) -> int:
    base_seed = int(config["benchmark"]["seed"])
    salt = sum(ord(char) for char in f"{dataset}:{stream_name}")
    return base_seed + salt


def _reservoir_consider(
    reservoir: list[dict[str, Any]],
    item: dict[str, Any],
    *,
    seen: int,
    limit: int,
    rng: random.Random,
) -> None:
    if limit <= 0:
        return
    if len(reservoir) < limit:
        reservoir.append(item)
        return
    replacement_index = rng.randrange(seen)
    if replacement_index < limit:
        reservoir[replacement_index] = item


def _write_table(
    rows: list[dict[str, Any]], path: pathlib.Path, schema: pa.Schema
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".parquet.tmp")
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, tmp_path, compression="zstd")
    tmp_path.replace(path)


def _stackexchange_doc_schema() -> pa.Schema:
    return pa.schema(
        [
            ("doc_id", pa.int64()),
            ("source_id", pa.string()),
            ("text", pa.string()),
            ("tags", pa.list_(pa.string())),
            ("score", pa.int64()),
            ("creation_date", pa.string()),
            ("answer_count", pa.int64()),
        ]
    )


def _stackexchange_query_schema() -> pa.Schema:
    return pa.schema(
        [
            ("query_id", pa.int64()),
            ("source_id", pa.string()),
            ("text", pa.string()),
            ("tags", pa.list_(pa.string())),
            ("score", pa.int64()),
            ("creation_date", pa.string()),
            ("answer_count", pa.int64()),
        ]
    )


def _reddit_doc_schema() -> pa.Schema:
    return pa.schema(
        [
            ("doc_id", pa.int64()),
            ("source_id", pa.string()),
            ("text", pa.string()),
            ("community", pa.string()),
            ("score", pa.int64()),
            ("datetime", pa.string()),
            ("data_type", pa.string()),
            ("subreddit_id", pa.string()),
        ]
    )


def _reddit_query_schema() -> pa.Schema:
    return pa.schema(
        [
            ("query_id", pa.int64()),
            ("source_id", pa.string()),
            ("text", pa.string()),
            ("community", pa.string()),
            ("score", pa.int64()),
            ("datetime", pa.string()),
            ("data_type", pa.string()),
            ("subreddit_id", pa.string()),
        ]
    )


def _write_manifest(
    dataset: str,
    output_dir: pathlib.Path,
    manifest: dict[str, Any],
) -> None:
    path = output_dir / dataset / "manifest.json"
    write_json(path, manifest)


def _processed_manifest_path(
    dataset: str, output_dir: pathlib.Path
) -> pathlib.Path:
    return output_dir / dataset / "manifest.json"


def _raw_source_manifest(
    dataset: str, raw_manifest: dict[str, Any]
) -> dict[str, Any]:
    entry = raw_manifest["datasets"][dataset]
    if dataset == "stackexchange":
        return {
            "source": entry["source"],
            "archive_item": entry["archive_item"],
            "dump_date": entry["dump_date"],
            "site": entry["site"],
            "verified": entry["verified"],
        }
    if dataset == "reddit":
        file_entry = _reddit_source_file(entry)
        return {
            "source": entry["source"],
            "repo_id": entry["repo_id"],
            "revision": entry["revision"],
            "repo_path": file_entry["repo_path"],
            "size": file_entry["size"],
            "sha1": file_entry["sha1"],
        }
    raise BenchmarkError(f"Unsupported dataset: {dataset}")


def _reddit_source_file(entry: dict[str, Any]) -> dict[str, Any]:
    if "files" not in entry:
        return entry
    for file_entry in entry["files"]:
        if file_entry["repo_path"] == REDDIT_REPO_PATH:
            return file_entry
    raise BenchmarkError("Reddit raw manifest does not include corpus zip")


def _normalize_cached_manifest(
    dataset: str, manifest: dict[str, Any]
) -> dict[str, Any]:
    if dataset != "reddit" or "raw_source" not in manifest:
        return manifest
    normalized = dict(manifest)
    normalized["raw_source"] = _raw_source_manifest(
        dataset, {"datasets": {dataset: normalized["raw_source"]}}
    )
    return normalized


def _cache_expectations(
    dataset: str,
    config: dict[str, Any],
    raw_manifest: dict[str, Any],
    max_source_records: int | None,
) -> dict[str, Any]:
    benchmark = config["benchmark"]
    expectations = {
        "dataset": dataset,
        "requested_doc_count": int(benchmark["n_docs"]),
        "requested_query_count": int(benchmark["n_queries"]),
        "doc_sample_seed": _seed(config, dataset, "docs"),
        "query_sample_seed": _seed(config, dataset, "queries"),
        "min_doc_chars": int(benchmark["min_doc_chars"]),
        "min_query_chars": int(benchmark["min_query_chars"]),
        "max_source_records": max_source_records,
        "raw_source": _raw_source_manifest(dataset, raw_manifest),
    }
    dataset_config = config["datasets"][dataset]
    for field_name in ("document_field", "query_field"):
        if field_name in dataset_config:
            expectations[field_name] = str(dataset_config[field_name])
    return expectations


def _is_cached(
    dataset: str,
    output_dir: pathlib.Path,
    expectations: dict[str, Any],
) -> bool:
    manifest_path = _processed_manifest_path(dataset, output_dir)
    docs_path = output_dir / dataset / "docs.parquet"
    queries_path = output_dir / dataset / "queries.parquet"
    if (
        not manifest_path.exists()
        or not docs_path.exists()
        or not queries_path.exists()
    ):
        return False
    manifest = _normalize_cached_manifest(dataset, load_json(manifest_path))
    return cache_matches(manifest, expectations)


def _progress(message: str, last_at: float) -> float:
    now = time.monotonic()
    if now - last_at >= PROGRESS_INTERVAL_SECONDS:
        print(message, file=sys.stderr, flush=True)
        return now
    return last_at


def _stackexchange_archive_path(raw_manifest: dict[str, Any]) -> pathlib.Path:
    entry = raw_manifest["datasets"]["stackexchange"]
    return pathlib.Path(entry["local_path"])


def _iter_stackexchange_rows(
    archive_path: pathlib.Path,
    max_questions: int | None = None,
) -> Iterable[dict[str, str]]:
    command = ["bsdtar", "-xOf", str(archive_path), STACKEXCHANGE_POSTS_XML]
    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
        if process.stdout is None:
            raise BenchmarkError("bsdtar did not provide stdout")
        stopped_early = False
        question_count = 0
        try:
            for _, element in ET.iterparse(process.stdout, events=("end",)):
                if element.tag == "row":
                    attrs = dict(element.attrib)
                    if attrs.get("PostTypeId") == "1":
                        if (
                            max_questions is not None
                            and question_count >= max_questions
                        ):
                            stopped_early = True
                            process.terminate()
                            break
                        question_count += 1
                    yield attrs
                    element.clear()
        finally:
            if process.stdout is not None:
                process.stdout.close()
            return_code = process.wait()
            if return_code != 0 and not stopped_early:
                raise BenchmarkError(
                    f"bsdtar failed with exit code {return_code}"
                )


def _load_stackexchange(
    config: dict[str, Any],
    raw_manifest: dict[str, Any],
    output_dir: pathlib.Path,
    max_source_records: int | None,
) -> dict[str, Any]:
    benchmark = config["benchmark"]
    doc_limit = int(benchmark["n_docs"])
    query_limit = int(benchmark["n_queries"])
    min_doc_chars = int(benchmark["min_doc_chars"])
    min_query_chars = int(benchmark["min_query_chars"])
    docs_rng = random.Random(_seed(config, "stackexchange", "docs"))
    queries_rng = random.Random(_seed(config, "stackexchange", "queries"))
    docs: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    seen_docs = 0
    seen_queries = 0
    total_questions = 0
    last_progress_at = time.monotonic()

    archive_path = _stackexchange_archive_path(raw_manifest)
    for attrs in _iter_stackexchange_rows(archive_path, max_source_records):
        if attrs.get("PostTypeId") != "1":
            continue
        total_questions += 1
        body_text = _strip_html(attrs.get("Body"))
        title_text = _normalize_text(attrs.get("Title"))
        source_id = _parse_int(attrs.get("Id"))
        tags = _parse_tags(attrs.get("Tags"))
        score = _parse_int(attrs.get("Score"))
        answer_count = _parse_int(attrs.get("AnswerCount"))
        creation_date = attrs.get("CreationDate", "")

        if _accept_text(body_text, min_doc_chars):
            seen_docs += 1
            _reservoir_consider(
                docs,
                {
                    "doc_id": source_id,
                    "source_id": str(source_id),
                    "text": body_text,
                    "tags": tags,
                    "score": score,
                    "creation_date": creation_date,
                    "answer_count": answer_count,
                },
                seen=seen_docs,
                limit=doc_limit,
                rng=docs_rng,
            )
        if _accept_text(title_text, min_query_chars):
            seen_queries += 1
            _reservoir_consider(
                queries,
                {
                    "query_id": source_id,
                    "source_id": str(source_id),
                    "text": title_text,
                    "tags": tags,
                    "score": score,
                    "creation_date": creation_date,
                    "answer_count": answer_count,
                },
                seen=seen_queries,
                limit=query_limit,
                rng=queries_rng,
            )
        last_progress_at = _progress(
            f"stackexchange: scanned {total_questions:,} questions",
            last_progress_at,
        )

    dataset_dir = output_dir / "stackexchange"
    docs.sort(key=lambda row: row["doc_id"])
    queries.sort(key=lambda row: row["query_id"])
    _write_table(
        docs, dataset_dir / "docs.parquet", _stackexchange_doc_schema()
    )
    _write_table(
        queries, dataset_dir / "queries.parquet", _stackexchange_query_schema()
    )
    manifest = {
        "dataset": "stackexchange",
        **_cache_expectations(
            "stackexchange", config, raw_manifest, max_source_records
        ),
        "docs_path": str(dataset_dir / "docs.parquet"),
        "queries_path": str(dataset_dir / "queries.parquet"),
        "doc_count": len(docs),
        "query_count": len(queries),
        "eligible_doc_count": seen_docs,
        "eligible_query_count": seen_queries,
        "total_questions": total_questions,
    }
    _write_manifest("stackexchange", output_dir, manifest)
    return manifest


def _reddit_archive_path(raw_manifest: dict[str, Any]) -> pathlib.Path:
    entry = raw_manifest["datasets"]["reddit"]
    return pathlib.Path(_reddit_source_file(entry)["local_path"])


def _iter_reddit_records(
    archive_path: pathlib.Path,
) -> Iterable[dict[str, Any]]:
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(REDDIT_JSON_NAME) as file_obj:
            for line in file_obj:
                if not line.strip():
                    continue
                record = json.loads(line)
                if isinstance(record, dict):
                    yield record


def _reddit_doc_id(index: int) -> int:
    return index + 1


def _load_reddit(
    config: dict[str, Any],
    raw_manifest: dict[str, Any],
    output_dir: pathlib.Path,
    max_source_records: int | None,
) -> dict[str, Any]:
    benchmark = config["benchmark"]
    dataset_config = config["datasets"]["reddit"]
    doc_limit = int(benchmark["n_docs"])
    query_limit = int(benchmark["n_queries"])
    min_doc_chars = int(benchmark["min_doc_chars"])
    min_query_chars = int(benchmark["min_query_chars"])
    document_field = str(dataset_config["document_field"])
    query_field = str(dataset_config["query_field"])
    docs_rng = random.Random(_seed(config, "reddit", "docs"))
    queries_rng = random.Random(_seed(config, "reddit", "queries"))
    docs: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    seen_docs = 0
    seen_queries = 0
    total_records = 0
    last_progress_at = time.monotonic()

    for record in _iter_reddit_records(_reddit_archive_path(raw_manifest)):
        if (
            max_source_records is not None
            and total_records >= max_source_records
        ):
            break
        source_index = total_records
        total_records += 1
        source_id = str(record.get("id") or source_index)
        content = _normalize_text(record.get(document_field))
        query_text = _normalize_text(record.get(query_field))
        subreddit = _normalize_text(record.get("subreddit"))
        subreddit_id = _normalize_text(record.get("subreddit_id"))

        if _accept_text(content, min_doc_chars):
            seen_docs += 1
            _reservoir_consider(
                docs,
                {
                    "doc_id": _reddit_doc_id(source_index),
                    "source_id": source_id,
                    "text": content,
                    "community": subreddit,
                    "score": None,
                    "datetime": None,
                    "data_type": "post",
                    "subreddit_id": subreddit_id,
                },
                seen=seen_docs,
                limit=doc_limit,
                rng=docs_rng,
            )
        if _accept_text(query_text, min_query_chars):
            seen_queries += 1
            _reservoir_consider(
                queries,
                {
                    "query_id": _reddit_doc_id(source_index),
                    "source_id": source_id,
                    "text": query_text,
                    "community": subreddit,
                    "score": None,
                    "datetime": None,
                    "data_type": "post",
                    "subreddit_id": subreddit_id,
                },
                seen=seen_queries,
                limit=query_limit,
                rng=queries_rng,
            )
        last_progress_at = _progress(
            f"reddit: scanned {total_records:,} records",
            last_progress_at,
        )

    dataset_dir = output_dir / "reddit"
    docs.sort(key=lambda row: row["doc_id"])
    queries.sort(key=lambda row: row["query_id"])
    _write_table(docs, dataset_dir / "docs.parquet", _reddit_doc_schema())
    _write_table(
        queries, dataset_dir / "queries.parquet", _reddit_query_schema()
    )
    manifest = {
        "dataset": "reddit",
        **_cache_expectations(
            "reddit", config, raw_manifest, max_source_records
        ),
        "docs_path": str(dataset_dir / "docs.parquet"),
        "queries_path": str(dataset_dir / "queries.parquet"),
        "doc_count": len(docs),
        "query_count": len(queries),
        "eligible_doc_count": seen_docs,
        "eligible_query_count": seen_queries,
        "total_records": total_records,
        "document_field": document_field,
        "query_field": query_field,
    }
    _write_manifest("reddit", output_dir, manifest)
    return manifest


def _load_one(
    dataset: str,
    config: dict[str, Any],
    raw_manifest: dict[str, Any],
    output_dir: pathlib.Path,
    max_source_records: int | None,
    force: bool,
) -> dict[str, Any]:
    expectations = _cache_expectations(
        dataset, config, raw_manifest, max_source_records
    )
    if not force and _is_cached(dataset, output_dir, expectations):
        print(f"Using cached {dataset} parquet outputs.", flush=True)
        return load_json(_processed_manifest_path(dataset, output_dir))
    if dataset == "stackexchange":
        return _load_stackexchange(
            config, raw_manifest, output_dir, max_source_records
        )
    if dataset == "reddit":
        return _load_reddit(
            config, raw_manifest, output_dir, max_source_records
        )
    raise BenchmarkError(f"Unsupported dataset: {dataset}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("stackexchange", "reddit", "all"),
        default="all",
        help="Dataset to parse. Defaults to both datasets.",
    )
    parser.add_argument(
        "--n-docs",
        type=int,
        default=None,
        help="Override benchmark.n_docs for this run.",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=None,
        help="Override benchmark.n_queries for this run.",
    )
    parser.add_argument(
        "--max-source-records",
        type=int,
        default=None,
        help="Stop after this many source records; for smoke tests only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate parquet outputs even when a matching cache exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = copy.deepcopy(load_yaml(DEFAULT_CONFIG))
    if args.n_docs is not None:
        config["benchmark"]["n_docs"] = args.n_docs
    if args.n_queries is not None:
        config["benchmark"]["n_queries"] = args.n_queries
    raw_manifest = load_json(DEFAULT_DATA_DIR / MANIFEST_NAME)
    output_dir = DEFAULT_DATA_DIR / "processed"
    selected = (
        ["stackexchange", "reddit"]
        if args.dataset == "all"
        else [args.dataset]
    )

    for dataset in selected:
        print(f"Parsing {dataset} raw input...", flush=True)
        manifest = _load_one(
            dataset,
            config,
            raw_manifest,
            output_dir,
            args.max_source_records,
            args.force,
        )
        print(
            f"Wrote {dataset}: {manifest['doc_count']:,} docs, "
            f"{manifest['query_count']:,} queries",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BenchmarkError, OSError, ET.ParseError, zipfile.BadZipFile) as exc:
        print(f"load.py: error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
