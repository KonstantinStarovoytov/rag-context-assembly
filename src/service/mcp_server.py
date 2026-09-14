"""MCP server exposing the RAG pipeline as tools.

Runs over stdio for a local editor, and over streamable HTTP when mounted into
the FastAPI app. Both transports call `src.service.core`, so a remote client and
a local one get the same answer.
"""

from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from src.config import settings
from src.observability import flush
from src.service import core

INSTRUCTIONS = """
Answers questions about the official Claude, Cursor, Codex and MCP documentation
using hybrid retrieval over an indexed snapshot, with citations.

Use `ask_docs` for a written answer with sources. Use `search_docs` when you want
the documentation passages themselves and will reason over them yourself.
""".strip()

READ_ONLY = ToolAnnotations(
    title=None,
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=True,
)

QUESTION_DESCRIPTION = (
    "Question about Claude, Cursor, Codex or MCP product behaviour. "
    "Any language; non-English is translated before retrieval. "
    "Not for repository code, private docs, or general programming advice."
)
ITERATIVE_DESCRIPTION = (
    "Run one extra retrieval round when the first context has evidence gaps. "
    "Slower and measured as unnecessary for most questions; leave false unless "
    "ask_docs already said the sources were incomplete."
)
QUERY_DESCRIPTION = (
    "Search query for the documentation index. Same language rules as ask_docs. "
    "Use this when you will read the passages yourself."
)
LIMIT_DESCRIPTION = (
    "How many passages to return. Higher costs more retrieval, not generation."
)


class SourceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation: int = Field(
        description="1-based citation number used in the answer, e.g. [1]."
    )
    title: str = Field(description="Document title.")
    heading: str = Field(description="Heading path inside the document.")
    url: str = Field(description="Source URL.")
    rerank_score: float = Field(description="Cohere rerank score for this chunk.")


class AskDocsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        description="Grounded answer. Factual claims cite sources as [1], [2]."
    )
    sources: list[SourceModel] = Field(
        description="Chunks the answer was generated from."
    )
    strategy: str = Field(description="Retrieval strategy that ran, usually hybrid.")
    iterative: bool = Field(
        description="Whether the extra retrieval round was requested."
    )


class PassageModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int = Field(description="1-based rank after hybrid retrieval.")
    title: str = Field(description="Document title.")
    heading: str = Field(description="Heading path inside the document.")
    url: str = Field(description="Source URL.")
    score: float = Field(description="Hybrid retrieval score.")
    content: str = Field(description="Chunk text.")


class SearchDocsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passages: list[PassageModel] = Field(
        description="Ranked documentation passages. Empty if nothing matched."
    )


def ask_docs(
    question: Annotated[
        str,
        Field(
            min_length=1,
            max_length=settings.api_max_question_chars,
            description=QUESTION_DESCRIPTION,
        ),
    ],
    iterative: Annotated[
        bool, Field(default=False, description=ITERATIVE_DESCRIPTION)
    ] = False,
) -> AskDocsResult:
    """Answer a question from the official Claude, Cursor, Codex or MCP docs.

    Use when the caller needs a written answer with citations. Do not use for
    repository code, private documents, or general programming advice outside
    those products. Prefer search_docs when you will read the sources yourself
    or want to avoid a generation charge.

    Errors: empty or overlong questions are rejected; an empty index returns
    an answer that says no relevant documentation was found.
    """
    try:
        result = core.answer(question, iterative=iterative)
    finally:
        flush()
    return AskDocsResult(
        answer=result.answer,
        sources=[
            SourceModel(
                citation=source.citation,
                title=source.title,
                heading=source.heading,
                url=source.url,
                rerank_score=source.rerank_score,
            )
            for source in result.sources
        ],
        strategy=result.strategy,
        iterative=result.iterative,
    )


def search_docs(
    query: Annotated[
        str,
        Field(
            min_length=1,
            max_length=settings.api_max_question_chars,
            description=QUERY_DESCRIPTION,
        ),
    ],
    limit: Annotated[
        int, Field(default=10, ge=1, le=50, description=LIMIT_DESCRIPTION)
    ] = 10,
) -> SearchDocsResult:
    """Retrieve official documentation passages without generating an answer.

    Use when you will reason over the sources yourself. Cheaper than ask_docs
    because it skips the generator. Do not use when the caller needs a finished
    cited answer — call ask_docs instead.

    Errors: empty or overlong queries are rejected; an empty index returns
    {"passages": []}.
    """
    try:
        passages = core.search(query, limit=limit)
    finally:
        flush()
    return SearchDocsResult(
        passages=[
            PassageModel(
                rank=passage.rank,
                title=passage.title,
                heading=passage.heading,
                url=passage.url,
                score=passage.score,
                content=passage.content,
            )
            for passage in passages
        ]
    )


def build_mcp_server() -> MCPServer[Any]:
    """A fresh server per app instance; its session manager cannot be reused."""
    mcp: MCPServer[Any] = MCPServer(
        name="rag-context-assembly",
        title="Agent documentation RAG",
        instructions=INSTRUCTIONS,
    )
    mcp.add_tool(
        ask_docs,
        title="Ask official agent docs",
        annotations=READ_ONLY.model_copy(update={"title": "Ask official agent docs"}),
        structured_output=True,
    )
    mcp.add_tool(
        search_docs,
        title="Search official agent docs",
        annotations=READ_ONLY.model_copy(
            update={"title": "Search official agent docs"}
        ),
        structured_output=True,
    )
    return mcp


def main() -> None:
    """Entry point for the stdio transport (local editor configuration)."""
    build_mcp_server().run(transport="stdio")


if __name__ == "__main__":
    main()
