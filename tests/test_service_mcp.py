"""MCP checks through a real client session, not by calling the tool functions."""

import socket
import threading
import time
from collections.abc import Iterator
from typing import Any

import httpx2
import pytest
import uvicorn
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from pydantic import SecretStr

from src.config import settings
from src.service import app as app_module
from src.service import core
from src.service.mcp_server import build_mcp_server

TOKEN = "test-token"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def stub_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    def answer(question: str, **_kwargs: Any) -> core.Answer:
        return core.Answer(
            answer=f"Answer to {question} [1].",
            sources=[
                core.Source(
                    citation=1,
                    title="Skills",
                    heading="Overview",
                    url="https://docs.test/skills.md",
                    rerank_score=0.91,
                )
            ],
            strategy="hybrid",
            iterative=False,
        )

    def search(query: str, *, limit: int | None = None) -> list[core.Passage]:
        return [
            core.Passage(
                rank=1,
                title="Skills",
                heading="Overview",
                url="https://docs.test/skills.md",
                score=0.5,
                content=f"passage for {query} limit={limit}",
            )
        ]

    monkeypatch.setattr(core, "answer", answer)
    monkeypatch.setattr(core, "search", search)


@pytest.mark.anyio
async def test_client_sees_both_tools(stub_pipeline: None) -> None:
    async with Client(build_mcp_server()) as client:
        tools = await client.list_tools()

    assert {tool.name for tool in tools.tools} == {"ask_docs", "search_docs"}


@pytest.mark.anyio
async def test_tools_advertise_runlayer_metadata(stub_pipeline: None) -> None:
    """Title, annotations, field docs and output schema must be on the wire."""
    async with Client(build_mcp_server()) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    ask = tools["ask_docs"]
    search = tools["search_docs"]

    assert ask.title == "Ask official agent docs"
    assert search.title == "Search official agent docs"
    assert ask.annotations is not None
    assert ask.annotations.read_only_hint is True
    assert ask.annotations.destructive_hint is False
    assert ask.annotations.idempotent_hint is True
    assert ask.annotations.open_world_hint is True
    assert "Do not use" in (ask.description or "")
    assert "Errors:" in (search.description or "")

    question = ask.input_schema["properties"]["question"]
    assert question["minLength"] == 1
    assert question["maxLength"] == 500
    assert "description" in question
    assert "description" in ask.input_schema["properties"]["iterative"]

    limit = search.input_schema["properties"]["limit"]
    assert limit["minimum"] == 1
    assert limit["maximum"] == 50
    assert ask.output_schema is not None
    assert "answer" in ask.output_schema["properties"]
    assert search.output_schema is not None
    assert "passages" in search.output_schema["properties"]


@pytest.mark.anyio
async def test_ask_docs_returns_answer_with_sources(stub_pipeline: None) -> None:
    async with Client(build_mcp_server()) as client:
        result = await client.call_tool("ask_docs", {"question": "How do skills work?"})

    payload = result.structured_content
    assert payload is not None
    assert payload["answer"] == "Answer to How do skills work? [1]."
    assert payload["sources"][0]["url"] == "https://docs.test/skills.md"


@pytest.mark.anyio
async def test_search_docs_passes_limit_through(stub_pipeline: None) -> None:
    async with Client(build_mcp_server()) as client:
        result = await client.call_tool("search_docs", {"query": "hooks", "limit": 3})

    payload = result.structured_content
    assert payload is not None
    assert payload["passages"][0]["content"] == "passage for hooks limit=3"


@pytest.mark.anyio
async def test_rejected_question_surfaces_as_tool_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*_args: Any, **_kwargs: Any) -> core.Answer:
        raise core.QuestionRejected("Question must not be empty")

    monkeypatch.setattr(core, "answer", reject)

    async with Client(build_mcp_server()) as client:
        result = await client.call_tool("ask_docs", {"question": " "})

    assert result.is_error is True


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture
def served_app(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Serve the real ASGI app, because the HTTP transport needs a real socket."""
    monkeypatch.setattr(settings, "api_token", SecretStr(TOKEN))
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(
            app_module.create_app(),
            host="127.0.0.1",
            port=port,
            log_level="warning",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        server.should_exit = True
        raise RuntimeError("test server did not start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.mark.anyio
async def test_mounted_http_transport_serves_tools(
    stub_pipeline: None, served_app: str
) -> None:
    """The deployed path: an MCP client over HTTP at /mcp, carrying the token."""
    async with httpx2.AsyncClient(
        headers={"Authorization": f"Bearer {TOKEN}"}
    ) as http_client:
        transport = streamable_http_client(
            f"{served_app}/mcp/", http_client=http_client
        )
        async with Client(transport) as client:
            tools = await client.list_tools()
            result = await client.call_tool(
                "ask_docs", {"question": "How do hooks work?"}
            )

    assert {tool.name for tool in tools.tools} == {"ask_docs", "search_docs"}
    payload = result.structured_content
    assert payload is not None
    assert payload["answer"] == "Answer to How do hooks work? [1]."


async def _mcp_post(host: str, body: dict[str, Any]) -> int:
    """POST to the mounted transport with a chosen Host header."""
    app = app_module.create_app()
    transport = httpx2.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx2.AsyncClient(transport=transport) as http_client:
            response = await http_client.post(
                f"http://{host}/mcp/",
                json=body,
                headers={
                    "Authorization": f"Bearer {TOKEN}",
                    "Accept": "application/json, text/event-stream",
                    "Content-Type": "application/json",
                },
            )
    return response.status_code


INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1"},
    },
}


@pytest.mark.anyio
async def test_declared_public_host_is_accepted(
    stub_pipeline: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deployment behind a domain must not answer 421 to its own hostname.

    The MCP transport rejects unknown Host headers to block DNS rebinding, so
    the public hostname has to be declared in API_ALLOWED_HOSTS.
    """
    monkeypatch.setattr(settings, "api_token", SecretStr(TOKEN))
    monkeypatch.setattr(settings, "api_allowed_hosts", "docs.example.test")

    assert await _mcp_post("docs.example.test", INITIALIZE) == 200


@pytest.mark.anyio
async def test_undeclared_host_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_token", SecretStr(TOKEN))
    monkeypatch.setattr(settings, "api_allowed_hosts", "docs.example.test")

    assert await _mcp_post("attacker.test", INITIALIZE) == 421


@pytest.mark.anyio
async def test_mounted_http_transport_rejects_missing_token(served_app: str) -> None:
    async with httpx2.AsyncClient() as http_client:
        response = await http_client.post(
            f"{served_app}/mcp/",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            headers={"Accept": "application/json, text/event-stream"},
        )

    assert response.status_code == 401
