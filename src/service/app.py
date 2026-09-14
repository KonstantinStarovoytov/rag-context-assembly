"""HTTP surface: a REST endpoint and the MCP streamable-HTTP transport.

Both live in one deployment so a single URL and a single token serve both an
`/ask` caller and an MCP client pointed at `/mcp`.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import Response

from src.config import settings
from src.observability import flush, trace_url
from src.rag.pipeline import RetrievalStrategy
from src.service import auth, core
from src.service.mcp_server import build_mcp_server

PUBLIC_PATHS = frozenset({"/health", "/docs", "/openapi.json", "/redoc"})


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    strategy: RetrievalStrategy | None = None
    iterative: bool = False


class SourceResponse(BaseModel):
    citation: int
    title: str
    heading: str
    url: str
    rerank_score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    strategy: str
    iterative: bool
    trace_url: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int | None = Field(default=None, ge=1, le=50)


class PassageResponse(BaseModel):
    rank: int
    title: str
    heading: str
    url: str
    score: float
    content: str


class SearchResponse(BaseModel):
    passages: list[PassageResponse]


def _allowed_hosts() -> list[str]:
    return [
        host.strip() for host in settings.api_allowed_hosts.split(",") if host.strip()
    ]


LOCAL_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
LOCAL_ORIGINS = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]


def _transport_security() -> TransportSecuritySettings | None:
    """Declare the public hostname to the MCP transport.

    It answers 421 to unknown `Host` headers to block DNS rebinding, so a
    deployment behind a real domain must name that domain. Returning None keeps
    the library default, which allows localhost only.
    """
    hosts = _allowed_hosts()
    if not hosts:
        return None

    allowed_hosts = list(LOCAL_HOSTS)
    allowed_origins = list(LOCAL_ORIGINS)
    for host in hosts:
        # Fly sends the bare domain; a proxy may append a port.
        allowed_hosts += [host, f"{host}:*"]
        allowed_origins += [f"https://{host}", f"https://{host}:*"]

    return TransportSecuritySettings(
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
    )


def create_app() -> FastAPI:
    """Build a fresh app, including a fresh MCP session manager.

    The manager refuses to start twice, so it cannot be shared between app
    instances; anything that starts the app more than once needs its own.
    """
    # Stateless: no session state to lose across restarts or extra machines.
    mcp_app = build_mcp_server().streamable_http_app(
        streamable_http_path="/",
        stateless_http=True,
        transport_security=_transport_security(),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Fail at startup, not on the first paying request.
        auth.required_token()
        # The MCP transport keeps its own task group; it must be started here.
        async with mcp_app.router.lifespan_context(mcp_app):
            yield
        flush()

    app = FastAPI(
        title="rag-context-assembly",
        description=(
            "Hybrid RAG over the official Claude, Cursor, Codex and MCP "
            "documentation. The same pipeline is served as an MCP server at /mcp."
        ),
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def require_token(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path in PUBLIC_PATHS or request.method == "OPTIONS":
            return await call_next(request)
        if not auth.token_accepted(request.headers.get("authorization")):
            return JSONResponse(
                {"detail": "Missing or invalid bearer token"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Liveness only: it must not call paid models or Qdrant."""
        return {
            "status": "ok",
            "strategy": settings.retrieval_strategy,
            "generation_top_k": settings.generation_top_k,
            "retrieval_top_k": settings.retrieval_top_k,
        }

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        try:
            result = core.answer(
                request.question,
                strategy=request.strategy,
                iterative=request.iterative,
            )
        except core.QuestionRejected as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        finally:
            flush()

        return AskResponse(
            answer=result.answer,
            sources=[SourceResponse(**asdict(source)) for source in result.sources],
            strategy=result.strategy,
            iterative=result.iterative,
            trace_url=trace_url(),
        )

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        """Retrieval without generation, so a caller can inspect the evidence."""
        try:
            passages = core.search(request.query, limit=request.limit)
        except core.QuestionRejected as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        finally:
            flush()

        return SearchResponse(
            passages=[PassageResponse(**asdict(passage)) for passage in passages]
        )

    app.mount("/mcp", mcp_app)
    return app


app = create_app()
