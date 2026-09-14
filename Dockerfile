# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_SYNC=1 \
    PYTHONUNBUFFERED=1 \
    # Bake the BM25 assets into the image instead of downloading on first query.
    FASTEMBED_CACHE_PATH=/app/.fastembed_cache

WORKDIR /app

# Dependencies resolve from the lock file only, so this layer caches across
# source edits. No dev group: the image does not run tests or linters.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
RUN uv sync --locked --no-dev

RUN uv run python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding('Qdrant/bm25')"

ENV PATH="/app/.venv/bin:$PATH"

# Fly sets PORT; the default matches fly.toml's internal_port.
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn src.service.app:app --host 0.0.0.0 --port ${PORT}"]
