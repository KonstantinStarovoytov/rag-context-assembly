"""MCP server exposing the RAG pipeline as tools.

Runs over stdio for a local editor, and over streamable HTTP when mounted into
the FastAPI app. Both transports call `src.service.core`, so a remote client and
a local one get the same answer.
"""

from dataclasses import asdict
from typing import Any

from mcp.server.mcpserver import MCPServer

from src.observability import flush
from src.service import core

INSTRUCTIONS = """
Answers questions about the official Claude, Cursor, Codex and MCP documentation
using hybrid retrieval over an indexed snapshot, with citations.

Use `ask_docs` for a written answer with sources. Use `search_docs` when you want
the documentation passages themselves and will reason over them yourself.
""".strip()


def ask_docs(question: str, iterative: bool = False) -> dict[str, Any]:
    """Answer a question about Claude, Cursor, Codex or MCP documentation.

    Returns a grounded answer with numbered citations and their source URLs.

    Args:
        question: The question, in any language; non-English is translated
            before retrieval.
        iterative: Run a second retrieval round when the first context has
            evidence gaps. Slower, and measured as unnecessary for most
            questions, so it is off by default.
    """
    try:
        return asdict(core.answer(question, iterative=iterative))
    finally:
        flush()


def search_docs(query: str, limit: int = 10) -> dict[str, Any]:
    """Retrieve documentation passages without generating an answer.

    Cheaper than ask_docs and lets the caller read the sources directly.

    Args:
        query: Search query.
        limit: How many passages to return.
    """
    try:
        passages = core.search(query, limit=limit)
    finally:
        flush()
    return {"passages": [asdict(passage) for passage in passages]}


def build_mcp_server() -> MCPServer[Any]:
    """A fresh server per app instance; its session manager cannot be reused."""
    mcp: MCPServer[Any] = MCPServer(
        name="rag-context-assembly",
        title="Agent documentation RAG",
        instructions=INSTRUCTIONS,
    )
    mcp.add_tool(ask_docs)
    mcp.add_tool(search_docs)
    return mcp


def main() -> None:
    """Entry point for the stdio transport (local editor configuration)."""
    build_mcp_server().run(transport="stdio")


if __name__ == "__main__":
    main()
