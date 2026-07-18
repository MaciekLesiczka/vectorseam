"""Replay demo queries against the search API at a fixed rate."""

from __future__ import annotations

import argparse
from collections import deque
import json
import pathlib
import random
import statistics
import sys
import time
import urllib.error
import urllib.request


DEFAULT_QPS = 5.0
DEFAULT_SEED = 7
DEFAULT_LOG_EVERY = 100
DEFAULT_TIMEOUT_SECONDS = 30.0


def _positive_float(value: str) -> float:
    """Parses a positive floating-point command-line value."""
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _positive_int(value: str) -> int:
    """Parses a positive integer command-line value."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _load_queries(path: pathlib.Path) -> list[str]:
    """Reads every query line without changing its text."""
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            queries = [line.rstrip("\r\n") for line in file_obj]
    except OSError as error:
        raise ValueError(
            f"could not read queries file {path}: {error}"
        ) from error
    if not queries:
        raise ValueError(f"queries file is empty: {path}")
    return queries


def _send_query(url: str, query: str, timeout_seconds: float) -> int:
    """Sends one query and returns the HTTP status code."""
    body = json.dumps({"query": query, "k": 10}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request, timeout=timeout_seconds
        ) as response:
            response.read()
            return int(response.status)
    except urllib.error.HTTPError as error:
        error.read()
        return int(error.code)


def _search_url(base_url: str) -> str:
    """Builds the search endpoint URL from the configured API base URL."""
    return base_url.rstrip("/") + "/search"


def replay(
    *,
    queries: list[str],
    url: str,
    qps: float,
    seed: int,
    log_every: int,
    timeout_seconds: float,
) -> None:
    """Replays a shuffled query pool forever."""
    shuffled_queries = list(queries)
    random.Random(seed).shuffle(shuffled_queries)
    search_url = _search_url(url)
    request_interval = 1.0 / qps
    latencies_ms: deque[float] = deque(maxlen=log_every)
    request_count = 0
    error_count = 0
    query_index = 0
    next_request_at = time.monotonic()

    while True:
        query = shuffled_queries[query_index]
        query_index = (query_index + 1) % len(shuffled_queries)
        started_at = time.monotonic()
        request_count += 1
        try:
            status = _send_query(search_url, query, timeout_seconds)
            if not 200 <= status < 300:
                error_count += 1
        except Exception:  # pylint: disable=broad-exception-caught
            error_count += 1
        latencies_ms.append((time.monotonic() - started_at) * 1000.0)

        if request_count % log_every == 0:
            rolling_mean = statistics.fmean(latencies_ms)
            print(
                f"requests={request_count} errors={error_count} "
                f"rolling_mean_latency_ms={rolling_mean:.2f}",
                flush=True,
            )

        next_request_at += request_interval
        delay = next_request_at - time.monotonic()
        if delay > 0.0:
            time.sleep(delay)
        else:
            next_request_at = time.monotonic()


def _parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        type=pathlib.Path,
        required=True,
        help="Path to the one-query-per-line input file.",
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Search API base URL.",
    )
    parser.add_argument(
        "--qps",
        type=_positive_float,
        default=DEFAULT_QPS,
        help=f"Target request rate. Defaults to {DEFAULT_QPS:g}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"One-time shuffle seed. Defaults to {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--log-every",
        type=_positive_int,
        default=DEFAULT_LOG_EVERY,
        help=(
            "Log every N requests and average the most recent N latencies. "
            f"Defaults to {DEFAULT_LOG_EVERY}."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=(
            "Per-request timeout in seconds. "
            f"Defaults to {DEFAULT_TIMEOUT_SECONDS:g}."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Runs the query replay loop."""
    args = _parse_args()
    try:
        queries = _load_queries(args.queries)
    except ValueError as error:
        print(f"driver: error: {error}", file=sys.stderr)
        return 1

    print(
        f"loaded {len(queries):,} queries; replaying at {args.qps:g} qps "
        f"with seed {args.seed}",
        flush=True,
    )
    try:
        replay(
            queries=queries,
            url=args.url,
            qps=args.qps,
            seed=args.seed,
            log_every=args.log_every,
            timeout_seconds=args.timeout,
        )
    except KeyboardInterrupt:
        print("driver: stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
