FROM ghcr.io/astral-sh/uv:0.11.15 AS uv

FROM python:3.12-slim

COPY --from=uv /uv /uvx /bin/

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY python ./python
COPY demo ./demo
RUN uv sync --locked --no-dev

CMD ["uvicorn", "demo.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
