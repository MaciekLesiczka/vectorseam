"""FastAPI search service for the VectorSeam M1 demo."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
import os
import time
from typing import Any

from fastapi import FastAPI, Request
import numpy as np
import psycopg
from psycopg import sql
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers import SentenceTransformer

from vectorseam import (
    ProbabilitySampler,
    VectorCaptureProducer,
    VectorSocketSender,
    capture_vector,
)


MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
EMBEDDING_DIMENSION = 384
MODEL_BATCH_SIZE = 256
COHORT_NAME = "superuser"
DEFAULT_DATABASE_URL = "postgresql://postgres:password@127.0.0.1:5432/postgres"


@dataclass(frozen=True)
class Settings:
    """Startup settings read from the environment."""

    database_url: str
    collector_host: str
    collector_port: int
    ef_search: int

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> Settings:
        """Builds and validates startup settings."""
        if environ is None:
            environ = os.environ

        database_url = environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
        if not database_url:
            raise ValueError("DATABASE_URL must be non-empty")

        collector_host = environ.get("COLLECTOR_HOST", "127.0.0.1")
        if not collector_host:
            raise ValueError("COLLECTOR_HOST must be non-empty")

        collector_port = _parse_environment_int(environ, "COLLECTOR_PORT", 7737)
        if not 1 <= collector_port <= 65535:
            raise ValueError("COLLECTOR_PORT must be between 1 and 65535")

        ef_search = _parse_environment_int(environ, "DEMO_EF_SEARCH", 100)
        if not 1 <= ef_search <= 1000:
            raise ValueError("DEMO_EF_SEARCH must be between 1 and 1000")

        return cls(
            database_url=database_url,
            collector_host=collector_host,
            collector_port=collector_port,
            ef_search=ef_search,
        )


class SearchRequest(BaseModel):
    """Search request body."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    k: int = Field(default=10, ge=1)


class SearchResult(BaseModel):
    """One nearest-neighbor result."""

    doc_id: int
    body: str
    distance: float


class SearchResponse(BaseModel):
    """Search response body."""

    results: list[SearchResult]
    latency_ms: float
    ef_search: int


def _parse_environment_int(
    environ: Mapping[str, str], name: str, default: int
) -> int:
    """Parses one integer-valued environment setting."""
    raw_value = environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


def _preprocess_text(text: Any) -> str:
    """Identity preprocessing copied from the benchmark embedding stage."""
    if text is None:
        return ""
    return str(text)


def _embed_query(
    model: SentenceTransformer,
    query: str,
) -> np.ndarray:
    """Embeds one query exactly as the benchmark embedding stage does."""
    embeddings = model.encode(
        [_preprocess_text(query)],
        batch_size=MODEL_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if embeddings.shape != (1, EMBEDDING_DIMENSION):
        raise RuntimeError(
            "embedding model returned shape "
            f"{embeddings.shape}, expected (1, {EMBEDDING_DIMENSION})"
        )
    return np.ascontiguousarray(embeddings[0], dtype=np.dtype("<f4"))


def _format_vector(vector: np.ndarray) -> str:
    """Formats a vector for pgvector input."""
    return "[" + ",".join(f"{float(value):.9g}" for value in vector) + "]"


def _search_database(
    settings: Settings,
    vector: np.ndarray,
    k: int,
) -> tuple[list[SearchResult], float]:
    """Runs one HNSW search transaction and returns its query latency."""
    vector_literal = _format_vector(vector)
    statement = """
        SELECT doc_id, body, embedding <=> %s::vector AS distance
        FROM docs_superuser
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """

    with psycopg.connect(settings.database_url) as connection:
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL("SET LOCAL hnsw.ef_search = {};").format(
                        sql.Literal(settings.ef_search)
                    )
                )
                started_at = time.perf_counter()
                cursor.execute(
                    statement,
                    (vector_literal, vector_literal, k),
                )
                rows = cursor.fetchall()
                latency_ms = (time.perf_counter() - started_at) * 1000.0

    results = [
        SearchResult(
            doc_id=int(doc_id),
            body=str(body)[:300],
            distance=float(distance),
        )
        for doc_id, body, distance in rows
    ]
    return results, latency_ms


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Owns the model, always-capture producer, and sender lifecycle."""
    settings = Settings.from_environment()
    model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION)
    producer = VectorCaptureProducer(sampler=ProbabilitySampler(1.0))
    sender = VectorSocketSender(
        host=settings.collector_host,
        port=settings.collector_port,
        producer=producer,
    )

    app.state.settings = settings
    app.state.model = model
    app.state.producer = producer
    app.state.sender = sender
    sender.start()
    try:
        yield
    finally:
        sender.stop()


app = FastAPI(title="VectorSeam M1 demo", lifespan=lifespan)


@app.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest, request: Request) -> SearchResponse:
    """Embeds, captures, and searches for one query."""
    vector = _embed_query(request.app.state.model, payload.query)
    capture_vector(
        COHORT_NAME,
        vector,
        producer=request.app.state.producer,
    )
    results, latency_ms = _search_database(
        request.app.state.settings,
        vector,
        payload.k,
    )
    return SearchResponse(
        results=results,
        latency_ms=latency_ms,
        ef_search=request.app.state.settings.ef_search,
    )
