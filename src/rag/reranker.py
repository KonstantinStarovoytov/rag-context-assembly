from dataclasses import dataclass

import cohere
from langchain_core.documents import Document

from src.config import settings
from src.observability import traced
from src.rag.retriever import SearchResult


@dataclass(frozen=True, slots=True)
class RerankResult:
    document: Document

    retrieval_score: float
    rerank_score: float

    original_rank: int
    rerank_rank: int


def _rerank_trace_input(
    _self: object,
    query: str,
    results: list[SearchResult],
    top_n: int = 5,
) -> dict[str, object]:
    return {
        "query": query,
        "top_n": top_n,
        "candidates": [
            {
                "source": result.document.metadata.get("source"),
                "content": result.document.page_content,
                "title": result.document.metadata.get("title"),
                "retrieval_score": result.score,
                "original_rank": rank,
            }
            for rank, result in enumerate(results, start=1)
        ],
    }


def _rerank_trace_output(results: list[RerankResult]) -> list[dict[str, object]]:
    return [
        {
            "source": result.document.metadata.get("source"),
            "content": result.document.page_content,
            "title": result.document.metadata.get("title"),
            "retrieval_score": result.retrieval_score,
            "rerank_score": result.rerank_score,
            "original_rank": result.original_rank,
            "rerank_rank": result.rerank_rank,
        }
        for result in results
    ]


# Cohere's default is 300 s; reranking 20 chunks takes about a second.
RERANK_TIMEOUT_SECONDS = 30.0


class CohereReranker:
    def __init__(self) -> None:
        self.client = cohere.ClientV2(
            api_key=settings.cohere_api_key.get_secret_value(),
            timeout=RERANK_TIMEOUT_SECONDS,
            max_retries=1,
        )

        self.model = settings.cohere_rerank_model

    @traced(
        "rerank-candidates",
        "retriever",
        input_factory=_rerank_trace_input,
        metadata_factory=lambda self, *args, **kwargs: {"model": self.model},
        output_factory=_rerank_trace_output,
    )
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int = 5,
    ) -> list[RerankResult]:
        if not results:
            return []

        documents = [self._document_text(result.document) for result in results]

        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=min(top_n, len(documents)),
        )

        reranked: list[RerankResult] = []

        for rerank_position, item in enumerate(
            response.results,
            start=1,
        ):
            original_index = item.index
            original_result = results[original_index]

            reranked.append(
                RerankResult(
                    document=original_result.document,
                    retrieval_score=original_result.score,
                    rerank_score=item.relevance_score,
                    original_rank=original_index + 1,
                    rerank_rank=rerank_position,
                )
            )

        return reranked

    @staticmethod
    def _document_text(document: Document) -> str:
        metadata = document.metadata

        heading = " > ".join(
            value
            for value in (
                metadata.get("h1"),
                metadata.get("h2"),
                metadata.get("h3"),
            )
            if value
        )

        return (
            f"Title: {metadata.get('title', '')}\n"
            f"Section: {heading}\n"
            f"Product: {metadata.get('product', '')}\n\n"
            f"{metadata.get('raw_content', document.page_content)}"
        )
