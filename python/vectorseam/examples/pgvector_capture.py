"""Example pgvector application code instrumented with Vectorseam."""

from __future__ import annotations

import hashlib

import numpy
import psycopg

from vectorseam import CaptureResult, VectorSocketSender, capture_vector

_DIMENSION = 8


def search_products(
    conn: psycopg.Connection,
    query: str,
) -> list[tuple[int, str, float]]:
    """Captures a query vector and runs a pgvector nearest-neighbor query."""
    query_vector, _ = _capture_query_vector(query)
    vector_literal = _pgvector_literal(query_vector)
    rows = conn.execute(
        """
        SELECT id, name, embedding <-> %s::vector AS distance
        FROM products
        ORDER BY embedding <-> %s::vector
        LIMIT 5
        """,
        (vector_literal, vector_literal),
    ).fetchall()
    return [(int(row[0]), str(row[1]), float(row[2])) for row in rows]


def capture_demo_vectors() -> list[CaptureResult]:
    """Captures a few vectors without opening a database connection."""
    return [
        _capture_query_vector(query)[1]
        for query in (
            "waterproof hiking jacket",
            "trail running shoes",
            "insulated coffee mug",
        )
    ]


def run_example(
    database_url: str,
    *,
    socket_path: str = "/tmp/vectorseam.sock",
) -> list[tuple[int, str, float]]:
    """Starts the sender, runs one instrumented query, and stops the sender."""
    sender = VectorSocketSender(socket_path=socket_path)
    sender.start()
    try:
        with psycopg.connect(database_url) as conn:
            return search_products(conn, "waterproof hiking jacket")
    finally:
        sender.stop()


def embed_query(query: str) -> numpy.ndarray:
    """Returns a deterministic little-endian F32 vector for this example."""
    digest = hashlib.sha256(query.encode("utf-8")).digest()
    values = numpy.frombuffer(digest[:_DIMENSION], dtype=numpy.uint8)
    vector = values.astype(numpy.dtype("<f4")) / 255.0
    return numpy.ascontiguousarray(vector, dtype=numpy.dtype("<f4"))


def _capture_query_vector(query: str) -> tuple[numpy.ndarray, CaptureResult]:
    """Embeds and captures a query vector."""
    query_vector = embed_query(query)
    result = capture_vector(
        "products.search.query",
        query_vector,
    )
    return query_vector, result


def _pgvector_literal(vector: numpy.ndarray) -> str:
    """Formats a vector value for a parameterized pgvector cast."""
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


__all__ = [
    "capture_demo_vectors",
    "embed_query",
    "run_example",
    "search_products",
]
