# syntax=docker/dockerfile:1

# Pinned by digest so a moved tag cannot change the build; bump deliberately.
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim@sha256:7cf77f594be8042dab6daa9fe326f90962252268b4f120a7f5dccce4d947e6c1

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

# The server parses untrusted input; it does not need root. The fastembed
# cache was written by root above, so hand it over too.
RUN useradd --system --uid 1000 --no-create-home app && chown -R app:app /app
USER app

# Render sets PORT; the default is for local runs.
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn src.service.app:app --host 0.0.0.0 --port ${PORT}"]
