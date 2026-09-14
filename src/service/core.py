"""Transport-independent request handling.

The HTTP endpoint and the MCP server must answer identically, so both go through
`answer` here rather than calling the pipeline with their own arguments.
"""

from dataclasses import dataclass

from src.config import settings
from src.rag.pipeline import RetrievalStrategy, answer_question
from src.rag.retriever import search_hybrid


class QuestionRejected(ValueError):
    """The request is malformed; the caller should not retry it unchanged."""


@dataclass(frozen=True, slots=True)
class Source:
    citation: int
    title: str
    heading: str
    url: str
    rerank_score: float


@dataclass(frozen=True, slots=True)
class Answer:
    answer: str
    sources: list[Source]
    strategy: str
    iterative: bool


@dataclass(frozen=True, slots=True)
class Passage:
    rank: int
    title: str
    heading: str
    url: str
    score: float
    content: str


def _validated_question(question: str) -> str:
    question = question.strip()
    if not question:
        raise QuestionRejected("Question must not be empty")
    limit = settings.api_max_question_chars
    if len(question) > limit:
        raise QuestionRejected(f"Question must be at most {limit} characters")
    return question


def answer(
    question: str,
    *,
    strategy: RetrievalStrategy | None = None,
    iterative: bool = False,
) -> Answer:
    question = _validated_question(question)
    result = answer_question(question, strategy=strategy, iterative=iterative)
    return Answer(
        answer=result.answer,
        sources=[
            Source(
                citation=source.citation,
                title=source.title,
                heading=source.heading,
                url=source.url,
                rerank_score=source.rerank_score,
            )
            for source in result.sources
        ],
        strategy=strategy or settings.retrieval_strategy,
        iterative=iterative,
    )


def search(question: str, *, limit: int | None = None) -> list[Passage]:
    """Retrieval without generation, for callers that want to read the sources."""
    question = _validated_question(question)
    k = limit if limit is not None else settings.retrieval_top_k
    if k < 1:
        raise QuestionRejected("limit must be positive")
    if k > 50:
        raise QuestionRejected("limit must be at most 50")

    passages = []
    for rank, result in enumerate(search_hybrid(query=question, k=k), start=1):
        metadata = result.document.metadata
        heading = " > ".join(
            value
            for value in (
                metadata.get("h1"),
                metadata.get("h2"),
                metadata.get("h3"),
            )
            if value
        )
        passages.append(
            Passage(
                rank=rank,
                title=metadata.get("title", "Untitled"),
                heading=heading,
                url=metadata.get("source", ""),
                score=result.score,
                content=result.document.page_content,
            )
        )
    return passages
